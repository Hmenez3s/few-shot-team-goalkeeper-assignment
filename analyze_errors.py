import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import prepare
import train

BINS = ((0.0, 0.0, "zero"), (0.0, 0.1, "(0,0.1)"), (0.1, 0.3, "[0.1,0.3)"), (0.3, 0.6, "[0.3,0.6)"), (0.6, 1.0, "[0.6,1]"))


def box_edges(bbox, width, height):
    if not isinstance(bbox, dict):
        return None
    try:
        cx, cy, w, h = (float(bbox[key]) for key in ("x_center", "y_center", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite((cx, cy, w, h)).all() or w <= 0 or h <= 0:
        return None
    left, top = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    right, bottom = min(float(width), cx + w / 2), min(float(height), cy + h / 2)
    return (left, top, right, bottom) if right > left and bottom > top else None


def maximum_overlap(query_box, other_boxes):
    if query_box is None:
        raise ValueError("query annotation has invalid clipped geometry")
    x1, y1, x2, y2 = query_box
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        raise ValueError("query annotation has zero clipped area")
    best = 0.0
    for other in other_boxes:
        if other is None:
            continue
        ox1, oy1, ox2, oy2 = other
        intersection = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
        best = max(best, intersection / area)
    return best


def same_annotation(left, right):
    return left.get("image_id") == right.get("image_id") and left.get("track_id") == right.get("track_id") and left.get("bbox_image") == right.get("bbox_image")


def raw_annotations_by_frame(archive, member):
    data = json.loads(archive.read(member))
    if not isinstance(data.get("annotations"), list):
        raise ValueError(f"invalid raw annotations in {member}")
    frames = defaultdict(list)
    for annotation in data["annotations"]:
        if isinstance(annotation, dict):
            frames[annotation.get("image_id")].append(annotation)
    return frames

def frame_features(frame, image, raw_frames):
    entry = frame["entry"]
    image_id = frame["row"].get("image_id")
    raw_annotations = raw_frames[image_id]
    candidates = []
    for annotation in frame["annotations"]:
        class_id = prepare.label_class(annotation, entry["swap_teams"])
        if class_id is None:
            continue
        track = prepare.player_track(annotation, class_id)
        if track is None:
            continue
        crop = prepare.crop_box(image, annotation.get("bbox_image"))
        if crop is None:
            continue
        matches = [raw_annotation for raw_annotation in raw_annotations if same_annotation(annotation, raw_annotation)]
        if len(matches) != 1:
            raise ValueError(f"raw identity mismatch for {entry['clip']} image {image_id}, track {annotation.get('track_id')}: {len(matches)} matches")
        candidates.append((crop, class_id, track, frame["time"], annotation, matches[0]))
    rows = []
    for crop, class_id, track, time, annotation, raw_query in candidates:
        others = [box_edges(a.get("bbox_image"), image.width, image.height) for a in raw_annotations
                  if a is not raw_query and prepare.safe_text((a.get("attributes") or {}).get("role")).lower() in {"player", "goalkeeper", "referee", "other"}]
        qbox = box_edges(raw_query.get("bbox_image"), image.width, image.height)
        rows.append({"x": crop, "class": class_id, "track": track, "time": time, "entry_clip": entry["clip"], "source_clip": frame["entry"]["clip"],
                     "overlap": maximum_overlap(qbox, others),
                     "height": qbox[3] - qbox[1],
                     "truncated": qbox[0] <= 1 or qbox[1] <= 1 or qbox[2] >= image.width - 1 or qbox[3] >= image.height - 1})
    return rows


def reconstruct(groups):
    rows_by_unit = {}
    for key, entries in groups.items():
        frames = prepare.collect_unit_frames(entries, Counter())
        selected = prepare.selected_frame_indices(frames, prepare.FRAME_BUDGET)
        rows = []
        archives = {}
        raw_cache = {}
        try:
            for index in sorted(selected):
                frame = frames[index]
                path = frame["entry"]["archive"]
                if path not in archives:
                    archives[path] = zipfile.ZipFile(path)
                archive = archives[path]
                names = set(archive.namelist())
                key_member = (path, frame["entry"]["member"])
                if key_member not in raw_cache:
                    raw_cache[key_member] = raw_annotations_by_frame(archive, key_member[1])
                member = next((name for name in frame["candidates"] if name and name in names), None)
                if member is None:
                    raise ValueError(f"missing image for {frame['entry']['clip']} image {frame['row'].get('image_id')}")
                with archive.open(member) as stream, Image.open(stream) as source:
                    image = source.convert("RGB")
                    rows.extend(frame_features(frame, image, raw_cache[key_member]))
        finally:
            for archive in archives.values():
                archive.close()
        rows.sort(key=lambda row: (row["time"], row["entry_clip"], row["class"], row["track"]))
        rows_by_unit[key] = rows
    return rows_by_unit


def summarize(queries):
    truth = np.asarray([row["truth"] for row in queries], dtype=np.int64)
    pred = np.asarray([row["pred"] for row in queries], dtype=np.int64)
    cm = np.zeros((4, 4), dtype=np.int64)
    if len(queries):
        np.add.at(cm, (truth, pred), 1)
    wrong = truth != pred

    def metrics(mask):
        count = int(mask.sum())
        errors = int((wrong & mask).sum())
        return {"queries": count, "errors": errors, "error_rate": errors / count if count else None}

    by_overlap = []
    overlap = np.asarray([row["overlap"] for row in queries])
    for lo, hi, name in BINS:
        mask = overlap == 0 if name == "zero" else ((overlap > lo) & (overlap < hi) if lo == 0 else ((overlap >= lo) & (overlap < hi) if hi < 1 else (overlap >= lo) & (overlap <= hi)))
        by_overlap.append({"bin": name, **metrics(mask)})
    errors = [row for row, failed in zip(queries, wrong) if failed]
    zero_errors = [row for row in errors if row["overlap"] == 0]
    return {"query_count": len(queries), "errors": int(wrong.sum()), "confusion_matrix": cm.tolist(),
            "zero_overlap_error_confusion_matrix": confusion(zero_errors), "zero_overlap_errors_by_source_clip": dict(sorted(Counter(row["source_clip"] for row in zero_errors).items())),
            "by_overlap_proxy": by_overlap}


def confusion(rows):
    matrix = [[0] * 4 for _ in range(4)]
    for row in rows:
        matrix[row["truth"]][row["pred"]] += 1
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if not args.root.expanduser().is_absolute():
        parser.error("--root must be an absolute isolated run root")
    root = args.root.expanduser().resolve()
    work = root / "work"
    if not work.is_dir():
        parser.error("--root/work must contain corrected manifest, crops, and checkpoint")
    prepare.ARCHIVES = {"test": "test.zip"}
    manifest, splits, _ = train.load_manifest(root)
    if manifest.get("audit", {}).get("team_normalization", {}).get("schema") != prepare.TEAM_NORMALIZATION_SCHEMA:
        raise ValueError("rerun prepare.py with the current team normalization before evaluation")
    baseline_path = root / "results" / "result.json"
    if not baseline_path.is_file():
        baseline_path = work / "result.json"
    baseline = json.loads(baseline_path.read_text())
    reference_matrix = baseline["test"]["trained"]["confusion_matrix"]
    checkpoint = work / "checkpoint.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"corrected trained checkpoint not found: {checkpoint}")
    audit_assignments = manifest["audit"]["team_normalization_assignments"]
    assignments = [row for row in audit_assignments if row["split"] == "test"]
    orientations = {(row["game"], row["half"], row["clip"]): row["orientation"] for row in assignments}
    if len(orientations) != len(assignments) or set(orientations.values()) - {"unchanged", "swapped"}:
        raise ValueError("invalid test normalization assignments")
    records = prepare.read_records(root, prepare.new_audit())
    groups = defaultdict(list)
    found = set()
    for entry in records:
        half = prepare.unit_identity(entry["record"])
        if half is None:
            continue
        key = (entry["game"], half[0], entry["clip"])
        if key not in orientations:
            continue
        if key in found:
            raise ValueError(f"duplicate test source clip: {key}")
        found.add(key)
        entry["half"] = half[0]
        entry["swap_teams"] = orientations[key] == "swapped"
        groups[f"game-{entry['game']}-half-{half[0]}"].append(entry)
    if found != set(orientations):
        raise ValueError("some normalized test source clips could not be read")
    reconstructed = reconstruct(groups)
    device = torch.device(args.device)
    model = train.build_model(root / "inputs" / "resnet18.pth", device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.eval()
    cache = train.ClipCache(1)
    queries = []
    for entry in splits["test"]:
        clip = cache.get(entry["path"])
        rows = reconstructed.get(entry["clip"], [])
        if len(rows) != len(clip["y"]):
            raise ValueError(f"{entry['clip']}: reconstructed rows {len(rows)} != array rows {len(clip['y'])}")
        for index, row in enumerate(rows):
            if not np.array_equal(row["x"], clip["x"][index]) or row["class"] != int(clip["y"][index]) or row["time"] != int(clip["t"][index]) or row["track"] != int(clip["track"][index]):
                raise ValueError(f"{entry['clip']} row {index}: crop/label/track/time alignment mismatch")
        episode = train.fixed_episode(clip)
        if episode is None:
            raise ValueError(f"no fixed baseline episode for {entry['clip']}")
        predictions = train.predictions(model, clip, episode, device)
        for index, prediction in zip(episode[1], predictions):
            row = rows[index]
            queries.append({"game": entry["game"], "source_clip": row["source_clip"], "index": int(index), "truth": int(clip["y"][index]), "pred": int(prediction),
                            "track": int(clip["track"][index]), "time": int(clip["t"][index]), "overlap": row["overlap"], "height": row["height"], "truncated": row["truncated"]})
    result = summarize(queries)
    if result["confusion_matrix"] != reference_matrix:
        raise ValueError(f"baseline fixed-test confusion mismatch: produced {result['confusion_matrix']}, recorded {reference_matrix}")
    result.update({"version": 1, "metric_limit": "maximum same-frame annotated-human intersection divided by query box area; box-overlap proxy only, not physical occlusion or direction; no causal inference about overlapping class",
                   "baseline_reference": {"path": str(baseline_path), "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(), "confusion_matrix": reference_matrix},
                   "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                   "coverage": {"manifest_test_clips": len(splits["test"]), "aligned_test_queries": len(queries), "source_clips": sorted({row["source_clip"] for row in queries})},
                   "queries": queries})
    output = work / "corrected-error-analysis-v1.json"
    train.atomic_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "queries"}, indent=2))


if __name__ == "__main__":
    main()
