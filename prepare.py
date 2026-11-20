import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


CLASSES = ["player_left", "player_right", "goalkeeper_left", "goalkeeper_right"]
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
SPLIT_PRIORITY = {"train": 0, "valid": 1, "test": 2}
ARCHIVES = {"train": "train.zip", "valid": "valid.zip", "test": "test.zip"}
CANVAS = (48, 96)
MIN_BOX = (8, 14)
FRAME_BUDGET = 96


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_text(value):
    return value.strip() if isinstance(value, str) else ""


def identity(record):
    info = record.get("info", {})
    game = safe_text(info.get("game_id"))
    clip = safe_text(info.get("id"))
    if not game or not clip:
        return None, None
    return game, clip


def read_records(root, audit):
    records = []
    for split, archive_name in ARCHIVES.items():
        archive_path = root / "archives" / archive_name
        if not archive_path.is_file():
            audit["archive_errors"].append({"split": split, "archive": archive_name, "reason": "missing_archive"})
            print(f"Scanned {split}: missing {archive_name}", flush=True)
            continue
        try:
            archive = zipfile.ZipFile(archive_path)
        except (OSError, zipfile.BadZipFile) as error:
            audit["archive_errors"].append({"split": split, "archive": archive_name, "reason": "unreadable_archive", "detail": str(error)})
            print(f"Scanned {split}: unreadable {archive_name}", flush=True)
            continue
        scanned = 0
        accepted_before = len(records)
        with archive:
            for member in sorted(archive.namelist()):
                if PurePosixPath(member).name != "Labels-GameState.json":
                    continue
                scanned += 1
                try:
                    record = json.loads(archive.read(member))
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, OSError) as error:
                    audit["label_errors"].append({"split": split, "member": member, "reason": "invalid_label_json", "detail": str(error)})
                    continue
                game, clip = identity(record)
                if game is None:
                    audit["missing_or_invalid_game_ids"].append({"split": split, "member": member, "clip_id": clip})
                    audit["exclusions"]["missing_or_invalid_game_id"] += 1
                    continue
                info = record.get("info")
                images = record.get("images")
                annotations = record.get("annotations")
                if not isinstance(info, dict) or not isinstance(images, list) or not isinstance(annotations, list):
                    audit["label_errors"].append({"split": split, "member": member, "game": game, "clip": clip, "reason": "missing_top_level_info_images_or_annotations"})
                    audit["exclusions"]["invalid_label_structure"] += 1
                    continue
                projected_info = {key: info.get(key) for key in ("game_id", "id", "game_time_start", "game_time_stop", "clip_start", "frame_rate", "im_dir")}
                projected_images = [{key: image.get(key) for key in ("image_id", "file_name", "is_labeled")} for image in images if isinstance(image, dict)]
                projected_annotations = []
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        continue
                    attributes = annotation.get("attributes") or {}
                    role = safe_text(attributes.get("role")).lower() if isinstance(attributes, dict) else ""
                    team = safe_text(attributes.get("team")).lower() if isinstance(attributes, dict) else ""
                    if role not in ("player", "goalkeeper") or team not in ("left", "right"):
                        audit["exclusions"]["irrelevant_annotation"] += 1
                        continue
                    projected_annotations.append({
                        "image_id": annotation.get("image_id"),
                        "track_id": annotation.get("track_id"),
                        "bbox_image": annotation.get("bbox_image"),
                        "attributes": {key: attributes.get(key) for key in ("role", "team", "jersey")},
                    })
                projected = {"info": projected_info, "images": projected_images, "annotations": projected_annotations}
                records.append({"split": split, "archive": archive_path, "member": member, "record": projected, "game": game, "clip": clip})
        print(f"Scanned {split}: labels={scanned}, retained={len(records) - accepted_before}", flush=True)
    return records


def assign_splits(records, audit):
    raw_games = {split: set() for split in ARCHIVES}
    by_game = defaultdict(list)
    for record in records:
        raw_games[record["split"]].add(record["game"])
        by_game[record["game"]].append(record)
    audit["raw_game_ids"] = {split: sorted(games) for split, games in raw_games.items()}
    audit["official_raw_overlap"] = {
        "train_valid": sorted(raw_games["train"] & raw_games["valid"]),
        "train_test": sorted(raw_games["train"] & raw_games["test"]),
        "valid_test": sorted(raw_games["valid"] & raw_games["test"]),
    }
    retained = []
    for game, game_records in sorted(by_game.items()):
        winning_split = max((entry["split"] for entry in game_records), key=SPLIT_PRIORITY.get)
        for entry in game_records:
            if entry["split"] == winning_split:
                retained.append(entry)
            else:
                audit["dropped_split_overlap"].append({"game": game, "clip": entry["clip"], "source_split": entry["split"], "retained_split": winning_split})
                audit["exclusions"]["lower_precedence_split"] += 1
    audit["retained_game_ids"] = {
        split: sorted({entry["game"] for entry in retained if entry["split"] == split})
        for split in ARCHIVES
    }
    audit["retained_clip_counts"] = dict(Counter(entry["split"] for entry in retained))
    return retained


def image_rows(record):
    rows = []
    seen = set()
    for source_index, row in enumerate(record["images"]):
        if not isinstance(row, dict):
            continue
        image_id = row.get("image_id")
        key = str(image_id)
        if image_id is None or key in seen:
            continue
        seen.add(key)
        rows.append((source_index, row))
    return rows


def frame_order(rows):
    def key(item):
        source_index, row = item
        value = row.get("image_id")
        try:
            return (0, int(value), source_index)
        except (TypeError, ValueError):
            return (1, source_index, source_index)
    return sorted(rows, key=key)


def sampled_indices(length, budget):
    count = min(length, budget)
    if count == length:
        return set(range(length))
    return {round(index * (length - 1) / (count - 1)) for index in range(count)} if count > 1 else {0}


def label_class(annotation, swap_teams=False):
    attributes = annotation.get("attributes") or {}
    role = safe_text(attributes.get("role")).lower()
    team = safe_text(attributes.get("team")).lower()
    if role not in ("player", "goalkeeper") or team not in ("left", "right"):
        return None
    if swap_teams:
        team = "right" if team == "left" else "left"
    return CLASS_INDEX[f"{role}_{team}"]


def crop_box(image, bbox):
    if not isinstance(bbox, dict):
        return None
    try:
        cx, cy = float(bbox["x_center"]), float(bbox["y_center"])
        width, height = float(bbox["w"]), float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([cx, cy, width, height]).all() or width <= 0 or height <= 0:
        return None
    left, top = max(0, int(np.floor(cx - width / 2))), max(0, int(np.floor(cy - height / 2)))
    right, bottom = min(image.width, int(np.ceil(cx + width / 2))), min(image.height, int(np.ceil(cy + height / 2)))
    if right - left < MIN_BOX[0] or bottom - top < MIN_BOX[1]:
        return None
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    scale = min(CANVAS[0] / crop.width, CANVAS[1] / crop.height)
    resized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", CANVAS, (0, 0, 0))
    canvas.paste(resized, ((CANVAS[0] - resized.width) // 2, (CANVAS[1] - resized.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def stable_unit_key(game, half):
    return hashlib.sha256(f"{game}:{half}".encode("utf-8")).hexdigest()[:20]


def unit_identity(record):
    info = record.get("info", {})
    match = re.match(r"^([12])\s*-\s*", safe_text(info.get("game_time_start")))
    stop_match = re.match(r"^([12])\s*-\s*", safe_text(info.get("game_time_stop")))
    try:
        clip_start = int(info["clip_start"])
        frame_rate = float(info["frame_rate"])
    except (KeyError, TypeError, ValueError):
        return None
    if not match or not stop_match or match.group(1) != stop_match.group(1) or frame_rate <= 0:
        return None
    return match.group(1), clip_start, frame_rate

def timestamp(frame, clip_start, frame_rate):
    match = re.search(r"(\d+)$", PurePosixPath(safe_text(frame.get("file_name"))).stem)
    if not match:
        return None
    return clip_start + round((int(match.group(1)) - 1) * 1000 / frame_rate)

def player_track(annotation, class_id):
    if class_id in (0, 1):
        jersey = safe_text((annotation.get("attributes") or {}).get("jersey"))
        if not jersey or jersey.lower() in {"unknown", "unk", "none", "-"}:
            return None
        try:
            jersey_number = int(jersey)
        except ValueError:
            return None
        return jersey_number if 0 <= jersey_number <= 99 else None
    try:
        return int(annotation["track_id"])
    except (KeyError, TypeError, ValueError):
        return None

def clip_jersey_sides(entry):
    sides = {"left": set(), "right": set()}
    for annotation in entry["record"]["annotations"]:
        if not isinstance(annotation, dict):
            continue
        class_id = label_class(annotation)
        if class_id not in (0, 1):
            continue
        jersey = player_track(annotation, class_id)
        if jersey is not None:
            sides["left" if class_id == 0 else "right"].add(jersey)
    shared = sides["left"] & sides["right"]
    sides["left"] -= shared
    sides["right"] -= shared
    return sides


def normalize_team_sides(groups, audit):
    failures = []
    for (split, game, half), entries in sorted(groups.items(), key=lambda item: (SPLIT_PRIORITY[item[0][0]], item[0][1], item[0][2])):
        ordered = sorted(entries, key=lambda entry: (entry["clip"], entry["member"], str(entry["archive"])))
        if len(ordered) > 16:
            reason = "clip_limit_exceeded"
            failures.append({"split": split, "game": game, "half": half, "reason": reason, "clip_count": len(ordered), "maximum_clips": 16})
            for entry in ordered:
                audit["team_normalization_assignments"].append({"split": split, "game": game, "half": half, "clip": entry["clip"], "orientation": "unresolved", "support": []})
            continue
        if len(ordered) == 1:
            signs = [1]
            pair_scores = {}
            best_score = 0
        else:
            side_sets = [clip_jersey_sides(entry) for entry in ordered]
            pair_scores = {}
            for left_index in range(len(ordered)):
                for right_index in range(left_index + 1, len(ordered)):
                    left = side_sets[left_index]
                    right = side_sets[right_index]
                    aligned = len(left["left"] & right["left"]) + len(left["right"] & right["right"])
                    crossed = len(left["left"] & right["right"]) + len(left["right"] & right["left"])
                    pair_scores[(left_index, right_index)] = {"aligned": aligned, "crossed": crossed, "score": aligned - crossed}
            best_score = None
            best_signs = []
            for mask in range(1 << (len(ordered) - 1)):
                candidate = [1] + [1 if mask & (1 << (index - 1)) else -1 for index in range(1, len(ordered))]
                score = sum(candidate[i] * candidate[j] * pair["score"] for (i, j), pair in pair_scores.items())
                if best_score is None or score > best_score:
                    best_score, best_signs = score, [candidate]
                elif score == best_score:
                    best_signs.append(candidate)
            if len(best_signs) != 1:
                failures.append({"split": split, "game": game, "half": half, "reason": "ambiguous_optimum", "optimal_assignment_count": len(best_signs), "objective": best_score})
                for index, entry in enumerate(ordered):
                    support = [{"clip": ordered[j if i == index else i]["clip"], **pair} for (i, j), pair in sorted(pair_scores.items()) if index in (i, j)]
                    audit["team_normalization_assignments"].append({"split": split, "game": game, "half": half, "clip": entry["clip"], "orientation": "unresolved", "support": support})
                continue
            signs = best_signs[0]
            unchanged = sum(sign == 1 for sign in signs)
            swapped = len(signs) - unchanged
            if unchanged == swapped:
                failures.append({"split": split, "game": game, "half": half, "reason": "tied_compatibility_majority", "objective": best_score})
                for index, entry in enumerate(ordered):
                    support = [{"clip": ordered[j if i == index else i]["clip"], **pair} for (i, j), pair in sorted(pair_scores.items()) if index in (i, j)]
                    audit["team_normalization_assignments"].append({"split": split, "game": game, "half": half, "clip": entry["clip"], "orientation": "unresolved", "support": support})
                continue
            if swapped > unchanged:
                signs = [-sign for sign in signs]
        for index, entry in enumerate(ordered):
            margin = sum(signs[index] * signs[other] * pair["score"] for (i, other), pair in pair_scores.items() if i == index)
            margin += sum(signs[other] * signs[index] * pair["score"] for (other, j), pair in pair_scores.items() if j == index)
            support = []
            for (left_index, right_index), pair in sorted(pair_scores.items()):
                if index in (left_index, right_index):
                    other = right_index if index == left_index else left_index
                    support.append({"clip": ordered[other]["clip"], **pair})
            orientation = "unchanged" if signs[index] == 1 else "swapped"
            entry["swap_teams"] = signs[index] == -1
            audit["team_normalization_assignments"].append({"split": split, "game": game, "half": half, "clip": entry["clip"], "orientation": orientation, "signed_aggregate_margin": margin, "support": support})
    audit["team_normalization_errors"].extend(failures)
    return failures


def collect_unit_frames(entries, audit):
    frames = []
    for entry in entries:
        record = entry["record"]
        half_data = unit_identity(record)
        if half_data is None or half_data[0] != entry["half"]:
            continue
        _, clip_start, frame_rate = half_data
        annotations_by_image = defaultdict(list)
        for annotation in record["annotations"]:
            if isinstance(annotation, dict):
                annotations_by_image[str(annotation.get("image_id"))].append(annotation)
        parent = str(PurePosixPath(entry["member"]).parent)
        im_dir = safe_text(record.get("info", {}).get("im_dir"))
        for _, row in frame_order(image_rows(record)):
            if row.get("is_labeled") is not True:
                continue
            frame_time = timestamp(row, clip_start, frame_rate)
            if frame_time is None:
                audit["exclusions"]["invalid_frame_number"] += 1
                continue
            filename = safe_text(row.get("file_name"))
            candidates = [str(PurePosixPath(parent) / filename), str(PurePosixPath(parent) / im_dir / filename) if im_dir else "", filename]
            frames.append({"entry": entry, "row": row, "time": frame_time, "annotations": annotations_by_image[str(row.get("image_id"))], "candidates": candidates})
    return sorted(frames, key=lambda frame: (frame["time"], frame["entry"]["clip"], str(frame["row"].get("image_id"))))

def selected_frame_indices(frames, budget):
    required = set()
    by_class = defaultdict(list)
    for index, frame in enumerate(frames):
        classes = {label_class(annotation, frame["entry"].get("swap_teams", False)) for annotation in frame["annotations"]}
        for class_id in (2, 3):
            if class_id in classes:
                by_class[class_id].append(index)
    for indices in by_class.values():
        if indices:
            required.update((indices[0], indices[-1]))
    if len(required) >= budget:
        return set(sorted(required)[:budget])
    return required | sampled_indices(len(frames), budget - len(required))

def eligible_cutoffs(observations):
    times = sorted({item["time"] for item in observations})
    eligible = []
    for left, right in zip(times, times[1:]):
        cutoff = (left + right) / 2
        early = [item for item in observations if item["time"] < cutoff]
        late = [item for item in observations if item["time"] > cutoff]
        if {item["class"] for item in early} != set(range(4)) or {item["class"] for item in late} != set(range(4)):
            continue
        early_tracks = defaultdict(set)
        late_tracks = defaultdict(set)
        for item in early:
            early_tracks[item["class"]].add(item["track"])
        for item in late:
            late_tracks[item["class"]].add(item["track"])
        if all(any(reference != query for reference in early_tracks[class_id] for query in late_tracks[class_id]) for class_id in (0, 1)):
            eligible.append(cutoff)
    return eligible

def process_unit(split, game, half, entries, work, audit):
    frames = collect_unit_frames(entries, audit)
    if not frames:
        return None, "no_timestamped_labeled_frames"
    selected = selected_frame_indices(frames, FRAME_BUDGET)
    archives = {}
    archive_names = {}
    observations = []
    crop_failures = 0
    try:
        for index in sorted(selected):
            frame = frames[index]
            path = frame["entry"]["archive"]
            archive = archives.get(path)
            if archive is None:
                archive = zipfile.ZipFile(path)
                archives[path] = archive
                archive_names[path] = set(archive.namelist())
            image_member = next((name for name in frame["candidates"] if name and name in archive_names[path]), None)
            if image_member is None:
                crop_failures += len(frame["annotations"])
                continue
            with archive.open(image_member) as stream, Image.open(stream) as source_image:
                image = source_image.convert("RGB")
                for annotation in frame["annotations"]:
                    class_id = label_class(annotation, frame["entry"].get("swap_teams", False))
                    if class_id is None:
                        continue
                    track = player_track(annotation, class_id)
                    if track is None:
                        if class_id in (0, 1):
                            audit["exclusions"]["unknown_jersey_player"] += 1
                        continue
                    crop = crop_box(image, annotation.get("bbox_image"))
                    if crop is None:
                        crop_failures += 1
                        continue
                    observations.append({"x": crop, "class": class_id, "time": frame["time"], "track": track, "clip": frame["entry"]["clip"]})
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        return None, f"image_read_error:{error}"
    finally:
        for archive in archives.values():
            archive.close()
    cutoffs = eligible_cutoffs(observations)
    if not cutoffs:
        return None, "missing_same_half_reference_query_coverage"
    cutoff = cutoffs[(len(cutoffs) - 1) // 2]
    observations.sort(key=lambda item: (item["time"], item["clip"], item["class"], item["track"]))
    arrays = {
        "x": np.stack([item["x"] for item in observations]).astype(np.uint8, copy=False),
        "y": np.asarray([item["class"] for item in observations], dtype=np.int64),
        "t": np.asarray([item["time"] for item in observations], dtype=np.int64),
        "track": np.asarray([item["track"] for item in observations], dtype=np.int64),
        "early": np.asarray([item["time"] < cutoff for item in observations], dtype=np.bool_),
    }
    digest = stable_unit_key(game, half)
    relative = Path("work") / "clips" / f"{split}_{digest}.npz"
    output = work.parent / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".npz", dir=output.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    unit_key = f"game-{game}-half-{half}"
    audit["units"].append({"split": split, "game": game, "half": half, "key": unit_key, "source_clip_ids": sorted({entry["clip"] for entry in entries}), "cutoff_ms": cutoff, "candidate_cutoff_count": len(cutoffs), "sampled_frames": len(selected), "samples": len(observations), "crop_failures": crop_failures})
    return {"split": split, "game": game, "clip": unit_key, "path": relative.as_posix()}, None

def new_audit():
    return {
        "version": 1,
        "classes": CLASSES,
        "canvas_wh": list(CANVAS),
        "minimum_clipped_box_wh": list(MIN_BOX),
        "frame_budget": FRAME_BUDGET,
        "frame_time_policy": "clip_start_ms + round((file_number - 1) * 1000 / frame_rate)",
        "unit_policy": "original game_id and matching game_time_start/game_time_stop half; team labels normalized to match-half slots from shared player jersey identities",
        "team_normalization": {"schema": 1, "protocol": "per-half exact signed pairwise jersey agreement; class strings are canonical match-half slots, not camera-relative left/right", "maximum_clips": 16},
        "class_mapping": {"role": ["player", "goalkeeper"], "team": ["left", "right"]},
        "team_normalization_assignments": [],
        "team_normalization_errors": [],
        "archive_errors": [],
        "label_errors": [],
        "missing_or_invalid_game_ids": [],
        "official_raw_overlap": {},
        "raw_game_ids": {},
        "retained_game_ids": {},
        "retained_clip_counts": {},
        "dropped_split_overlap": [],
        "exclusions": Counter(),
        "clip_exclusions": [],
        "units": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.root.expanduser().is_absolute():
        parser.error("--root must be absolute")
    root = args.root.expanduser().resolve()
    work = root / "work"
    audit = new_audit()
    records = read_records(root, audit)
    retained = assign_splits(records, audit)
    audit["retained_label_count"] = len(retained)
    for entry in retained:
        half_data = unit_identity(entry["record"])
        if half_data is None:
            audit["exclusions"]["invalid_half_or_clip_timing"] += 1
            audit["clip_exclusions"].append({"split": entry["split"], "game": entry["game"], "clip": entry["clip"], "reason": "invalid_half_or_clip_timing"})
            continue
        entry["half"] = half_data[0]
    retained = [entry for entry in retained if "half" in entry]
    groups = defaultdict(list)
    for entry in retained:
        groups[(entry["split"], entry["game"], entry["half"])].append(entry)
    normalization_failures = normalize_team_sides(groups, audit)
    if args.audit_only:
        audit["audit_only"] = True
        audit["exclusions"] = dict(audit["exclusions"])
        atomic_json(work / "audit.json", audit)
        if normalization_failures:
            raise SystemExit("team normalization unresolved; inspect work/audit.json")
        return 0
    if normalization_failures:
        audit["exclusions"] = dict(audit["exclusions"])
        atomic_json(work / "audit.json", audit)
        raise SystemExit("team normalization unresolved; inspect work/audit.json")
    manifest_clips = []
    ordered_groups = sorted(groups.items(), key=lambda item: (SPLIT_PRIORITY[item[0][0]], item[0][1], item[0][2]))
    for index, ((split, game, half), entries) in enumerate(ordered_groups, start=1):
        result, reason = process_unit(split, game, half, entries, work, audit)
        if result is None:
            audit["clip_exclusions"].append({"split": split, "game": game, "half": half, "source_clip_ids": sorted(entry["clip"] for entry in entries), "reason": reason})
            audit["exclusions"][reason] += 1
        else:
            manifest_clips.append(result)
        if index % 5 == 0 or index == len(ordered_groups):
            print(f"Preprocessed units: {index}/{len(ordered_groups)} ({len(manifest_clips)} eligible)", flush=True)
    audit["output_unit_counts"] = dict(Counter(unit["split"] for unit in manifest_clips))
    audit["crop_shape"] = [96, 48, 3]
    audit["sample_count"] = sum(unit["samples"] for unit in audit["units"])
    audit["exclusions"] = dict(audit["exclusions"])
    atomic_json(work / "audit.json", audit)
    output_games = {split: {unit["game"] for unit in manifest_clips if unit["split"] == split} for split in ARCHIVES}
    if any(not output_games[split] for split in ARCHIVES) or any(output_games[a] & output_games[b] for a in ARCHIVES for b in ARCHIVES if SPLIT_PRIORITY[a] < SPLIT_PRIORITY[b]):
        raise SystemExit("required split has no eligible disjoint game-half units; inspect work/audit.json")
    manifest = {"classes": CLASSES, "clips": manifest_clips, "audit": audit}
    atomic_json(work / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
