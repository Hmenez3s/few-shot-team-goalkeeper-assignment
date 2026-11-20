import argparse
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import prepare

BOOTSTRAP_FRACTION = 0.3
MAX_BOOTSTRAP_FRAMES = 192
MAX_QUERY_FRAMES = 96
ENCODE_BATCH_SIZE = 64
SUPPORT_COUNTS = {"one_reference": 1, "six_references": 6}


def timeline_cutoff(frames, bootstrap_fraction=BOOTSTRAP_FRACTION):
    times = sorted({int(frame["time"]) for frame in frames})
    if len(times) < 2:
        return None
    bootstrap_count = max(1, math.ceil(len(times) * bootstrap_fraction))
    if bootstrap_count >= len(times):
        return None
    return times[bootstrap_count - 1]


def query_frame_indices(frames, cutoff):
    if cutoff is None:
        return []
    return [index for index, frame in enumerate(frames) if int(frame["time"]) > cutoff]


def choose_support(candidates, cutoff, per_class):
    selected = {class_id: [] for class_id in range(len(prepare.CLASSES))}
    ordered = sorted(candidates, key=lambda row: (row["time"], row["clip"], str(row["image_id"]), row["annotation_index"]))
    for class_id in range(len(prepare.CLASSES)):
        seen_times = set()
        for candidate in ordered:
            if candidate["class_id"] != class_id or candidate["time"] > cutoff or not candidate.get("quality_valid", True):
                continue
            if candidate["time"] in seen_times:
                continue
            seen_times.add(candidate["time"])
            selected[class_id].append(candidate)
            if len(selected[class_id]) == per_class:
                break
    missing = [class_id for class_id, rows in selected.items() if len(rows) < per_class]
    return selected, missing


def load_unit_frames(manifest, records, audit):
    assignments = manifest["audit"]["team_normalization_assignments"]
    orientation = {}
    for row in assignments:
        if row.get("split") != "test":
            continue
        key = (row.get("game"), row.get("half"), row.get("clip"))
        value = row.get("orientation")
        if key in orientation or value not in {"unchanged", "swapped"}:
            raise ValueError(f"invalid test normalization assignment: {key}")
        orientation[key] = value
    groups = defaultdict(list)
    found = set()
    for entry in records:
        half_data = prepare.unit_identity(entry["record"])
        if half_data is None:
            continue
        key = (entry["game"], half_data[0], entry["clip"])
        if key not in orientation:
            continue
        if key in found:
            raise ValueError(f"duplicate test source clip: {key}")
        found.add(key)
        entry["half"] = half_data[0]
        entry["swap_teams"] = orientation[key] == "swapped"
        groups[(entry["game"], half_data[0])].append(entry)
    if found != set(orientation):
        raise ValueError("some normalized test source clips could not be read")
    units = {}
    for (game, half), entries in sorted(groups.items()):
        key = (game, half)
        units[key] = prepare.collect_unit_frames(entries, audit)
    return units


def extract_candidates(frames, selected_indices):
    archives = {}
    names = {}
    candidates = []
    valid_frames = set()
    try:
        for frame_index in sorted(selected_indices):
            frame = frames[frame_index]
            entry = frame["entry"]
            archive_path = entry["archive"]
            if archive_path not in archives:
                archives[archive_path] = zipfile.ZipFile(archive_path)
                names[archive_path] = set(archives[archive_path].namelist())
            member = next((name for name in frame["candidates"] if name and name in names[archive_path]), None)
            if member is None:
                continue
            with archives[archive_path].open(member) as stream, Image.open(stream) as source:
                image = source.convert("RGB")
                for annotation_index, annotation in enumerate(frame["annotations"]):
                    class_id = prepare.label_class(annotation, entry.get("swap_teams", False))
                    if class_id is None:
                        continue
                    crop = prepare.crop_box(image, annotation.get("bbox_image"))
                    if crop is None:
                        continue
                    row = {"x": crop, "class_id": class_id, "quality_valid": True, "time": int(frame["time"]), "frame_index": frame_index,
                           "clip": entry["clip"], "image_id": frame["row"].get("image_id"), "annotation_index": annotation_index}
                    candidates.append(row)
                    valid_frames.add(frame_index)
    finally:
        for archive in archives.values():
            archive.close()
    return candidates, valid_frames


def support_report(support, missing):
    return {"missing_class_ids": missing, "references": {str(class_id): [
        {"game_clip": row["clip"], "timestamp_ms": row["time"], "frame_index": row["frame_index"], "image_id": row["image_id"], "annotation_index": row["annotation_index"]}
        for row in rows] for class_id, rows in support.items()}}


def load_model(checkpoint_path, weights_path, device):
    import torch
    import train

    model = train.build_model(weights_path, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint must contain a model state dictionary: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def encode(model, crops, device):
    import torch
    import train

    images = torch.from_numpy(np.stack(crops).copy()).permute(0, 3, 1, 2).contiguous().float().div_(255)
    images = (images - train.MEAN) / train.STD
    with torch.no_grad():
        return torch.nn.functional.normalize(model(images.to(device)), dim=1).cpu().numpy()


def score_session(model, candidates, support, query_indices, device):
    if any(not support[class_id] for class_id in range(4)):
        return None
    support_rows = [row for class_id in range(4) for row in support[class_id]]
    support_features = encode(model, [row["x"] for row in support_rows], device)
    prototypes = np.stack([support_features[offset:offset + len(support[class_id])].mean(axis=0)
                           for class_id, offset in ((class_id, sum(len(support[j]) for j in range(class_id))) for class_id in range(4))])
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    query_rows = [row for row in candidates if row["frame_index"] in query_indices]
    if not query_rows:
        return []
    features = np.concatenate([encode(model, [row["x"] for row in query_rows[start:start + ENCODE_BATCH_SIZE]], device)
                               for start in range(0, len(query_rows), ENCODE_BATCH_SIZE)])
    scores = features @ prototypes.T
    output = []
    for row, logits in zip(query_rows, scores):
        output.append({"game_clip": row["clip"], "timestamp_ms": row["time"], "frame_index": row["frame_index"],
                       "image_id": row["image_id"], "annotation_index": row["annotation_index"], "truth_class_id": row["class_id"],
                       "prediction_class_id": int(np.argmax(logits)), "scores": logits.tolist()})
    return output


def summarize_predictions(rows):
    confusion = [[0] * 4 for _ in range(4)]
    for row in rows:
        confusion[row["truth_class_id"]][row["prediction_class_id"]] += 1
    count = len(rows)
    return {"query_count": count, "accuracy": sum(confusion[i][i] for i in range(4)) / count if count else None,
            "confusion_matrix": confusion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint-one", required=True, type=Path)
    parser.add_argument("--checkpoint-six", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.root.expanduser().is_absolute():
        parser.error("--root must be an absolute isolated run root")
    root = args.root.expanduser().resolve()
    if not (root / "work" / "manifest.json").is_file():
        parser.error("--root/work/manifest.json is required")
    output = (args.out or root / "work" / "prospective-evaluation-v1.json").resolve()
    if not output.is_relative_to((root / "work").resolve()) or output.exists():
        parser.error("--out must be a new file under --root/work")
    manifest = json.loads((root / "work" / "manifest.json").read_text())
    if manifest.get("audit", {}).get("team_normalization", {}).get("schema") != prepare.TEAM_NORMALIZATION_SCHEMA:
        raise ValueError("rerun prepare.py with the current team normalization before evaluation")
    if manifest.get("classes") != prepare.CLASSES:
        raise ValueError("manifest classes do not match prepare.py")
    test_games = {row["game"] for row in manifest.get("clips", []) if row.get("split") == "test"}
    other_games = {row["game"] for row in manifest.get("clips", []) if row.get("split") in {"train", "valid"}}
    if not test_games or test_games & other_games:
        raise ValueError("test games must be nonempty and unseen in train/valid")
    weights = (args.weights or root / "inputs" / "resnet18.pth").expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    prepare.ARCHIVES = {"test": "test.zip"}
    audit = prepare.new_audit()
    records = prepare.read_records(root, audit)
    units = load_unit_frames(manifest, records, audit)
    import torch
    import train

    torch.set_num_threads(4)
    device = torch.device(args.device)
    checkpoint_paths = {"one_reference_trained": args.checkpoint_one.expanduser().resolve()}
    if args.checkpoint_six:
        checkpoint_paths["six_reference_trained"] = args.checkpoint_six.expanduser().resolve()
    if any(not path.is_relative_to(root) for path in checkpoint_paths.values()):
        raise ValueError("checkpoint paths must be inside the isolated --root")
    models = {name: load_model(path, weights, device) for name, path in checkpoint_paths.items()}
    all_results = {name: {arm: {"sessions": [], "predictions": []} for arm in SUPPORT_COUNTS} for name in models}
    for (game, half), frames in units.items():
        cutoff = timeline_cutoff(frames)
        bootstrap = [index for index, frame in enumerate(frames) if cutoff is not None and frame["time"] <= cutoff]
        bootstrap_indices = [bootstrap[index] for index in sorted(prepare.sampled_indices(len(bootstrap), MAX_BOOTSTRAP_FRAMES))]
        later = query_frame_indices(frames, cutoff)
        query_indices = [later[index] for index in sorted(prepare.sampled_indices(len(later), MAX_QUERY_FRAMES))]
        candidates, valid_crop_frames = extract_candidates(frames, set(bootstrap_indices) | set(query_indices))
        early_candidates = [row for row in candidates if cutoff is not None and row["time"] <= cutoff]
        for arm, count in SUPPORT_COUNTS.items():
            support, missing = choose_support(early_candidates, cutoff, count)
            for name, model in models.items():
                row_predictions = score_session(model, candidates, support, set(query_indices), device) if not missing else None
                result = all_results[name][arm]
                if row_predictions is not None:
                    result["predictions"].extend([{**row, "game": game, "half": half} for row in row_predictions])
                result["sessions"].append({"game": game, "half": half, "cutoff_timestamp_ms": cutoff,
                                 "bootstrap_window": {"fraction": BOOTSTRAP_FRACTION, "inclusive_end_timestamp_ms": cutoff,
                                                      "available_frames": len(bootstrap), "sampled_frames": len(bootstrap_indices)},
                                 "available_query_frames": len(later), "query_frame_indices": query_indices,
                                 "valid_query_frame_indices": [i for i in query_indices if i in valid_crop_frames],
                                 "coverage": "covered" if not missing else "missing_early_reference_coverage",
                                 "reference": support_report(support, missing), "prediction_count": len(row_predictions or [])})
        print(f"Evaluated {game}/{half}: bootstrap frames={len(bootstrap_indices)}, query frames={len(query_indices)}", flush=True)
    for arms in all_results.values():
        for result in arms.values():
            result["metrics"] = summarize_predictions(result["predictions"])
            result["covered_sessions"] = sum(row["coverage"] == "covered" for row in result["sessions"])
            result["session_count"] = len(result["sessions"])
    paired = {}
    for checkpoint_name, arms in all_results.items():
        left = { (row["game"], row["half"]): row for row in arms["one_reference"]["sessions"] }
        right = { (row["game"], row["half"]): row for row in arms["six_references"]["sessions"] }
        common = sorted(key for key in left.keys() & right.keys() if left[key]["coverage"] == right[key]["coverage"] == "covered")
        common_set = set(common)
        one_rows = [row for row in arms["one_reference"]["predictions"] if (row["game"], row["half"]) in common_set]
        six_rows = [row for row in arms["six_references"]["predictions"] if (row["game"], row["half"]) in common_set]
        paired[checkpoint_name] = {"common_covered_sessions": [list(key) for key in common],
                                   "common_session_count": len(common),
                                   "one_reference": summarize_predictions(one_rows),
                                   "six_references": summarize_predictions(six_rows)}
    result = {"version": 1, "evaluation": "prospective_fixed_replay_unseen_games", "split": "test", "classes": prepare.CLASSES,
              "bootstrap_fraction": BOOTSTRAP_FRACTION, "support_policy": "first chronological quality-valid recrops from early labeled annotations; one or six distinct early timestamps per class",
              "query_policy": "all valid crops in up to 96 uniformly sampled later labeled frames per half; frame indices selected by time, not class or jersey",
              "bootstrap_sampling": "up to 192 uniform frames from the first 30% of labeled timestamps per half",
              "checkpoint_paths": {name: str(path) for name, path in checkpoint_paths.items()},
              "checkpoint_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in checkpoint_paths.items()},
              "results": all_results, "paired_common_coverage": paired}
    train.atomic_json(output, result)
    print(json.dumps({"output": str(output), "paired_common_coverage": paired}, indent=2))


if __name__ == "__main__":
    main()
