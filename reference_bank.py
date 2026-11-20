import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

import prepare

SPLITS = ("train", "valid", "test")
SAMPLE_INTERVAL_MS = 500


BANK_SCHEMA = 1
BANK_TEXT_SIZE = 64


def safe_text(value):
    return value.strip() if isinstance(value, str) else ""


def normalized_protocol_id(manifest):
    audit = manifest.get("audit", {})
    normalization = audit.get("team_normalization")
    if not isinstance(normalization, dict) or normalization.get("schema") != prepare.TEAM_NORMALIZATION_SCHEMA:
        raise ValueError("corrected team normalization is missing; rerun prepare.py before consuming raw clips")
    protocol = " ".join(safe_text(normalization.get("protocol")).split()).casefold()
    classes = manifest.get("classes")
    if not protocol or not isinstance(classes, list) or len(classes) != 4 or any(not isinstance(value, str) for value in classes):
        raise ValueError("invalid bank protocol or class mapping")
    identity = json.dumps({"protocol": protocol, "classes": classes}, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(identity.encode("ascii")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bank_identity(root, manifest, entry, bank_path):
    clips = [clip for clip in manifest.get("clips", [])
             if isinstance(clip, dict) and clip.get("split") == entry.get("split")
             and clip.get("game") == entry.get("game") and clip.get("clip") == entry.get("clip")]
    if (len(clips) != 1 or not clips[0].get("path") or not entry.get("path")
            or (Path(root) / clips[0]["path"]).resolve() != (Path(root) / entry["path"]).resolve()):
        raise ValueError(f"{entry.get('clip')}: prepared clip entry is missing or ambiguous")
    units = [unit for unit in manifest.get("audit", {}).get("units", [])
             if isinstance(unit, dict) and unit.get("split") == entry.get("split")
             and str(unit.get("game")) == str(entry.get("game"))
             and unit.get("key") == entry.get("clip")]
    if len(units) != 1:
        raise ValueError(f"{entry.get('clip')}: unit manifest entry is missing or ambiguous")
    unit = units[0]
    cutoff = unit.get("cutoff_ms")
    if type(cutoff) not in (int, float) or not np.isfinite(cutoff):
        raise ValueError(f"{entry.get('clip')}: unit cutoff must be a finite number")
    path = Path(bank_path)
    if path.name != f"{entry['clip']}.npz":
        raise ValueError(f"{path}: unexpected bank path for {entry['clip']}")
    prepared_path = Path(root) / entry["path"]
    if not prepared_path.is_file():
        raise ValueError(f"{prepared_path}: prepared clip is missing")
    return {"bank_schema": BANK_SCHEMA,
            "protocol_id": normalized_protocol_id(manifest),
            "prepared_sha256": sha256_file(prepared_path),
            "cutoff_ms": math.ceil(cutoff)}


def frame_rows(record):
    rows = []
    seen = set()
    for row in record.get("images", []):
        if not isinstance(row, dict) or row.get("is_labeled") is not True:
            continue
        image_id = row.get("image_id")
        if image_id is None or str(image_id) in seen:
            continue
        seen.add(str(image_id))
        rows.append(row)
    return rows


def bbox_values(bbox):
    if not isinstance(bbox, dict):
        return None
    try:
        cx, cy = float(bbox["x_center"]), float(bbox["y_center"])
        width, height = float(bbox["w"]), float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([cx, cy, width, height]).all() or width <= 0 or height <= 0:
        return None
    return cx, cy, width, height


def bbox_edges(bbox, image_size):
    values = bbox_values(bbox)
    if values is None:
        return None
    cx, cy, width, height = values
    image_width, image_height = image_size
    left = max(0.0, cx - width / 2)
    top = max(0.0, cy - height / 2)
    right = min(float(image_width), cx + width / 2)
    bottom = min(float(image_height), cy + height / 2)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def pitch_position(annotation):
    pitch = annotation.get("bbox_pitch")
    if not isinstance(pitch, dict):
        return None
    try:
        x = float(pitch["x_bottom_middle"]) / 105.0
        y = float(pitch["y_bottom_middle"]) / 68.0
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([x, y]).all():
        return None
    return float(np.clip(x + 0.5, 0, 1)), float(np.clip(y + 0.5, 0, 1))


def descriptor(annotation, all_annotations, image, bbox):
    position = pitch_position(annotation)
    edges = bbox_edges(bbox, image.size)
    values = bbox_values(bbox)
    if edges is None or values is None:
        return None
    left, top, right, bottom = edges
    torso = image.crop((left + (right - left) * 0.25,
                        top + (bottom - top) * 0.22,
                        left + (right - left) * 0.75,
                        top + (bottom - top) * 0.68)).convert("L")
    brightness = float(np.median(np.asarray(torso, dtype=np.uint8))) / 255.0
    subject_area = max(1.0, (right - left) * (bottom - top))
    overlap = 0.0
    for other in all_annotations:
        if other is annotation:
            continue
        other_edges = bbox_edges(other.get("bbox_image"), image.size)
        if other_edges is None:
            continue
        intersection_width = max(0.0, min(right, other_edges[2]) - max(left, other_edges[0]))
        intersection_height = max(0.0, min(bottom, other_edges[3]) - max(top, other_edges[1]))
        overlap = max(overlap, intersection_width * intersection_height / subject_area)
    _, _, width, height = values
    aspect = width / height
    aspect_proxy = aspect / (1.0 + aspect)
    pitch_x, pitch_y = (np.nan, np.nan) if position is None else position
    return np.asarray([pitch_x, pitch_y, brightness, overlap, aspect_proxy], dtype=np.float32)


def clip_identity(record):
    info = record.get("info", {})
    return safe_text(info.get("game_id")), safe_text(info.get("id"))


def team_swap_for_clip(manifest, split, game, half, clip):
    audit = manifest.get("audit", {})
    normalization = audit.get("team_normalization")
    if not isinstance(normalization, dict) or normalization.get("schema") != prepare.TEAM_NORMALIZATION_SCHEMA:
        raise ValueError("corrected team normalization is missing; rerun prepare.py before consuming raw clips")
    assignments = audit.get("team_normalization_assignments")
    if not isinstance(assignments, list):
        raise ValueError("team normalization assignments are missing; rerun prepare.py")
    matches = [assignment for assignment in assignments
               if isinstance(assignment, dict)
               and assignment.get("split") == split
               and str(assignment.get("game")) == str(game)
               and str(assignment.get("half")) == str(half)
               and assignment.get("clip") == clip]
    if not matches:
        raise ValueError(f"no team normalization assignment for {split}/{game}/half-{half}/{clip}; rerun prepare.py")
    if len(matches) != 1:
        raise ValueError(f"duplicate team normalization assignments for {split}/{game}/half-{half}/{clip}")
    orientation = matches[0].get("orientation")
    if orientation not in {"unchanged", "swapped"}:
        raise ValueError(f"unresolved team normalization assignment for {split}/{game}/half-{half}/{clip}; rerun prepare.py")
    return orientation == "swapped"

def read_unit_candidates(root, unit, prepared_times, audit, manifest):
    split = unit["split"]
    archive_path = root / "archives" / f"{split}.zip"
    candidates = []
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        labels = []
        folders = {f"SNGS-{clip.zfill(3)}" for clip in unit["source_clip_ids"]}
        for member in sorted(members):
            label_path = PurePosixPath(member)
            if label_path.name != "Labels-GameState.json" or label_path.parent.name not in folders:
                continue
            try:
                record = json.loads(archive.read(member))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            game, clip = clip_identity(record)
            if game != str(unit["game"]) or clip not in unit["source_clip_ids"]:
                continue
            timing = prepare.unit_identity(record)
            if timing is None or timing[0] != str(unit["half"]):
                continue
            swap_teams = team_swap_for_clip(manifest, split, unit["game"], timing[0], clip)
            labels.append((member, record, clip, timing, swap_teams))
        frames = []
        for member, record, clip, (_, clip_start, frame_rate), swap_teams in labels:
            annotations_by_image = defaultdict(list)
            for annotation in record.get("annotations", []):
                if isinstance(annotation, dict) and annotation.get("supercategory") == "object" and (annotation.get("attributes") or {}).get("role") in {"player", "goalkeeper", "referee", "other"}:
                    annotations_by_image[str(annotation.get("image_id"))].append(annotation)
            parent = PurePosixPath(member).parent
            im_dir = safe_text(record.get("info", {}).get("im_dir"))
            for row in frame_rows(record):
                frame_time = prepare.timestamp(row, clip_start, frame_rate)
                if frame_time is None or frame_time >= unit["cutoff_ms"]:
                    continue
                annotations = annotations_by_image[str(row.get("image_id"))]
                if not annotations:
                    continue
                filename = safe_text(row.get("file_name"))
                paths = [str(parent / filename), str(parent / im_dir / filename) if im_dir else "", filename]
                image_member = next((path for path in paths if path and path in members), None)
                if image_member is None:
                    audit["missing_image_members"] += 1
                    continue
                frames.append({"time": frame_time, "clip": clip, "annotations": annotations,
                               "image_member": image_member, "swap_teams": swap_teams})
        frames.sort(key=lambda frame: (frame["time"], frame["clip"], frame["image_member"]))
        sampled_slots = {}
        for index, frame in enumerate(frames):
            slot = frame["time"] // SAMPLE_INTERVAL_MS
            sampled_slots.setdefault(slot, index)
            if any(prepare.label_class(annotation, frame["swap_teams"]) in (2, 3) for annotation in frame["annotations"]):
                frame["goalkeeper_frame"] = True
            else:
                frame["goalkeeper_frame"] = False
        selected = set(sampled_slots.values())
        selected.update(index for index, frame in enumerate(frames) if frame["goalkeeper_frame"])
        selected.update(index for index, frame in enumerate(frames) if frame["time"] in prepared_times)
        seen = set()
        for index in sorted(selected):
            frame = frames[index]
            with archive.open(frame["image_member"]) as stream, Image.open(stream) as source:
                image = source.convert("RGB")
            for annotation in frame["annotations"]:
                class_id = prepare.label_class(annotation, frame["swap_teams"])
                if class_id is None:
                    continue
                track = prepare.player_track(annotation, class_id)
                if class_id in (0, 1) and track is None:
                    audit["unknown_jersey"] += 1
                    continue
                bbox = annotation.get("bbox_image")
                crop = prepare.crop_box(image, bbox)
                if crop is None:
                    audit["invalid_crop"] += 1
                    continue
                desc = descriptor(annotation, frame["annotations"], image, bbox)
                if desc is None:
                    audit["invalid_descriptor"] += 1
                    continue
                dedup = (frame["time"], class_id, track if class_id in (0, 1) else None)
                if dedup in seen:
                    continue
                seen.add(dedup)
                candidates.append({"x": crop, "t": frame["time"], "y": class_id,
                                   "track": -1 if track is None else track, "desc": desc,
                                   "clip": frame["clip"]})
            image.close()
    candidates.sort(key=lambda item: (item["t"], item["y"], item["track"], item["clip"]))
    return candidates, len(frames), len(selected)


def read_prepared_times(root, manifest, unit):
    times = set()
    for entry in manifest.get("clips", []):
        if entry.get("split") != unit["split"] or str(entry.get("game")) != str(unit["game"]):
            continue
        if entry.get("clip") != unit["key"]:
            continue
        path = root / entry["path"]
        if not path.is_file():
            continue
        with np.load(path) as clip:
            times.update(int(time) for time in clip["t"][clip["early"]])
    return times


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.root.expanduser().is_absolute():
        parser.error("--root must be absolute")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.root.expanduser().resolve()
    manifest = json.loads((root / "work" / "manifest.json").read_text(encoding="utf-8"))
    units = [unit for unit in manifest.get("audit", {}).get("units", []) if unit.get("split") in SPLITS]
    units.sort(key=lambda unit: (SPLITS.index(unit["split"]), str(unit["game"]), str(unit["half"])))
    if args.limit is not None:
        units = units[:args.limit]
    output = root / "work" / "references"
    audit = {"version": 1, "classes": manifest.get("classes", prepare.CLASSES),
             "references_per_class_target": 6, "units": [],
             "source_policy": "each session reads only its own split archive and only frames earlier than its cut-off",
             "sampling": {"interval_ms": SAMPLE_INTERVAL_MS,
                          "additional_frames": "all earlier frames containing a goalkeeper annotation and frames at exact earlier prepared-NPZ times"},
             "descriptors": {"pitch_x": "clip(bbox_pitch.x_bottom_middle / 105 + 0.5, 0, 1); optional, NaN when pitch position is missing",
                             "pitch_y": "clip(bbox_pitch.y_bottom_middle / 68 + 0.5, 0, 1); optional, NaN when pitch position is missing",
                             "pitch_consumers": "no current reference selection or evaluation consumer uses pitch_x or pitch_y",
                             "brightness": "median grayscale value in central torso crop, divided by 255; source crop excludes black padding",
                             "overlap": "maximum intersection area with another annotated human bbox divided by clipped subject bbox area",
                             "aspect_proxy": "bbox_image width/height transformed as r/(1+r)"},
             "exclusions": Counter()}
    for index, unit in enumerate(units, start=1):
        if not isinstance(unit.get("source_clip_ids"), list) or "cutoff_ms" not in unit or not unit.get("key"):
            raise SystemExit(f"invalid manifest unit: {unit.get('key')}")
        if not re.fullmatch(r"game-.+-half-[12]", str(unit["key"])):
            raise SystemExit(f"unexpected unit key: {unit['key']}")
        prepared_times = read_prepared_times(root, manifest, unit)
        candidates, frame_count, selected_count = read_unit_candidates(root, unit, prepared_times, audit["exclusions"], manifest)
        bank_path = output / f"{unit['key']}.npz"
        entries = [entry for entry in manifest.get("clips", [])
                   if entry.get("split") == unit["split"] and str(entry.get("game")) == str(unit["game"])
                   and entry.get("clip") == unit["key"]]
        if len(entries) != 1:
            raise ValueError(f"{unit['key']}: prepared clip entry is missing or ambiguous")
        identity = bank_identity(root, manifest, entries[0], bank_path)
        arrays = {"x": np.stack([item["x"] for item in candidates]).astype(np.uint8, copy=False) if candidates else np.empty((0, 96, 48, 3), dtype=np.uint8),
                  "t": np.asarray([item["t"] for item in candidates], dtype=np.int64),
                  "y": np.asarray([item["y"] for item in candidates], dtype=np.int64),
                  "track": np.asarray([item["track"] for item in candidates], dtype=np.int64),
                  "desc": np.stack([item["desc"] for item in candidates]).astype(np.float32, copy=False) if candidates else np.empty((0, 5), dtype=np.float32),
                  "bank_schema": np.asarray(identity["bank_schema"], dtype=np.int64),
                  "protocol_id": np.asarray(identity["protocol_id"], dtype=f"U{BANK_TEXT_SIZE}"),
                  "prepared_sha256": np.asarray(identity["prepared_sha256"], dtype=f"U{BANK_TEXT_SIZE}"),
                  "cutoff_ms": np.asarray(identity["cutoff_ms"], dtype=np.int64)}
        atomic_npz(bank_path, arrays)
        coverage = {str(class_id): int(np.count_nonzero(arrays["y"] == class_id)) for class_id in range(4)}
        audit["units"].append({"split": unit["split"], "game": unit["game"], "half": unit["half"], "key": unit["key"],
                               "cutoff_ms": unit["cutoff_ms"], "source_clip_ids": unit["source_clip_ids"],
                               "timestamped_early_frames": frame_count, "selected_frames": selected_count,
                               "candidate_count": len(candidates), "coverage_by_class": coverage,
                               "prepared_early_times": len(prepared_times),
                               "candidates_at_prepared_times": int(sum(item["t"] in prepared_times for item in candidates))})
        print(f"Reference bank {index}/{len(units)} {unit['key']}: frames={selected_count}/{frame_count} candidates={len(candidates)} classes={coverage}", flush=True)
    audit["exclusions"] = dict(audit["exclusions"])
    atomic_json(output / "audit.json", audit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
