import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experiments import analyze_errors
import prepare
import train

ROLES = {"player", "goalkeeper", "referee", "other"}
SLOTS = {"player": (0, 1), "goalkeeper": (2, 3)}


def test_groups(root):
    prepare.ARCHIVES = {"test": "test.zip"}
    manifest, splits, _ = train.load_manifest(root)
    assignments = [row for row in manifest["audit"]["team_normalization_assignments"] if row["split"] == "test"]
    orientation = {(r["game"], r["half"], r["clip"]): r["orientation"] for r in assignments}
    records = prepare.read_records(root, prepare.new_audit())
    groups, entries = defaultdict(list), {}
    for entry in records:
        half = prepare.unit_identity(entry["record"])
        if half is None:
            continue
        key = (entry["game"], half[0], entry["clip"])
        if key in orientation:
            entry["half"] = half[0]
            entry["swap_teams"] = orientation[key] == "swapped"
            groups[f"game-{entry['game']}-half-{half[0]}"].append(entry)
            entries[entry["clip"]] = entry
    if set(entries) != {key[2] for key in orientation}:
        raise ValueError("test clip set does not align with normalized assignments")
    return manifest, splits["test"], groups, entries


def extract(root, output):
    root = Path(root).resolve()
    manifest, split_entries, groups, entries = test_groups(root)
    reconstructed = analyze_errors.reconstruct(groups)
    rows = []
    query_by_clip = {}
    cache = train.ClipCache(1)
    for entry in split_entries:
        clip = cache.get(entry["path"])
        fixed = train.fixed_episode(clip)
        if fixed is None:
            raise ValueError(f"no fixed episode for {entry['clip']}")
        query_by_clip[entry["clip"]] = {row_index: q for q, row_index in enumerate(fixed[1])}
        aligned = reconstructed.get(entry["clip"], [])
        if len(aligned) != len(clip["y"]):
            raise ValueError(f"{entry['clip']}: prepared row alignment mismatch")
        for index, row in enumerate(aligned):
            if row["class"] != int(clip["y"][index]) or not np.array_equal(row["x"], clip["x"][index]) or row["time"] != int(clip["t"][index]) or row["track"] != int(clip["track"][index]):
                raise ValueError(f"{entry['clip']} row {index}: crop/label/track/time alignment mismatch")
    manifest_assignments = {(r["game"], r["half"], r["clip"]): r["orientation"] == "swapped" for r in manifest["audit"]["team_normalization_assignments"] if r["split"] == "test"}
    for unit_entries in groups.values():
        frames = prepare.collect_unit_frames(unit_entries, Counter())
        selected = prepare.selected_frame_indices(frames, prepare.FRAME_BUDGET)
        archives, names, raw_cache = {}, {}, {}
        try:
            for fi in sorted(selected):
                frame = frames[fi]
                entry = frame["entry"]
                path = entry["archive"]
                archive = archives.setdefault(path, zipfile.ZipFile(path))
                if path not in names:
                    names[path] = set(archive.namelist())
                key = (path, entry["member"])
                if key not in raw_cache:
                    raw_cache[key] = analyze_errors.raw_annotations_by_frame(archive, entry["member"])
                image_name = next((n for n in frame["candidates"] if n and n in names[path]), None)
                if image_name is None:
                    raise ValueError(f"missing image {entry['clip']} {frame['row'].get('image_id')}")
                with archive.open(image_name) as stream, Image.open(stream) as source:
                    image = source.convert("RGB")
                    raw = raw_cache[key][frame["row"].get("image_id")]
                    for annotation in raw:
                        role = prepare.safe_text((annotation.get("attributes") or {}).get("role")).lower()
                        if role not in ROLES:
                            continue
                        box = analyze_errors.box_edges(annotation.get("bbox_image"), image.width, image.height)
                        if box is None:
                            continue
                        left, top, right, bottom = box
                        crop = np.asarray(image.crop((math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom))), dtype=np.uint8)
                        match_rows = [i for i, a in enumerate(frame["annotations"]) if analyze_errors.same_annotation(a, annotation)]
                        # Prepared indices are resolved against the exact source-frame row order.
                        prepared_index, query_index, canonical = -1, -1, None
                        if len(match_rows) == 1:
                            # The feature rows are ordered by timestamp, clip, class, track; find by annotation identity.
                            class_id = prepare.label_class(annotation, entry.get("swap_teams", False))
                            if class_id is not None:
                                track = prepare.player_track(annotation, class_id)
                                if track is not None:
                                    candidates = [(i, r) for grouped_rows in reconstructed.values() for i, r in enumerate(grouped_rows) if r["source_clip"] == entry["clip"] and r["time"] == frame["time"] and r["track"] == track and r["class"] == class_id]
                                    if len(candidates) == 1:
                                        prepared_index = candidates[0][0]
                                        prepared_clip = f"game-{entry['game']}-half-{entry['half']}"
                                        query_index = query_by_clip[prepared_clip].get(prepared_index, -1)
                                        canonical = class_id
                        pitch = annotation.get("bbox_pitch") or {}
                        try:
                            x_pitch = float(pitch["x_bottom_middle"])
                        except (KeyError, TypeError, ValueError):
                            x_pitch = float("nan")
                        rows.append((crop, {"clip": entry["clip"], "game": entry["game"], "session": frame["entry"]["half"], "image_id": frame["row"].get("image_id"), "track_id": annotation.get("track_id"), "role": role, "class": canonical, "pitch_x": x_pitch, "query_index": query_index, "prepared_index": prepared_index}))
        finally:
            for archive in archives.values():
                archive.close()
    query_rows = [r for _, r in rows if r["query_index"] >= 0]
    expected = sum(len(queries) for queries in query_by_clip.values())
    if len(query_rows) != expected:
        raise ValueError(f"query alignment count {len(query_rows)} != {expected}")
    lengths = np.array([len(crop) for crop, _ in rows], dtype=np.int64)
    widths = np.array([crop.shape[1] for crop, _ in rows], dtype=np.int32)
    heights = np.array([crop.shape[0] for crop, _ in rows], dtype=np.int32)
    offsets = np.r_[0, np.cumsum(lengths * widths * 3)]
    pixels = np.concatenate([crop.reshape(-1) for crop, _ in rows]).astype(np.uint8)
    metadata = [r for _, r in rows]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "extract.npz", pixels=pixels, offsets=offsets, widths=widths, heights=heights, metadata=np.asarray(json.dumps(metadata)))
    print(json.dumps({"crop_count": len(rows), "query_count": len(query_rows), "clips": sorted({r["clip"] for r in metadata}), "output": str(output / "extract.npz")}, indent=2))


def _kmeans_two(vectors):
    try:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=2, random_state=0).fit_predict(vectors)
    except ImportError:
        pass
    rng = np.random.RandomState(0)
    centers = [vectors[rng.randint(len(vectors))]]
    distances = np.sum((vectors - centers[0]) ** 2, axis=1)
    total = distances.sum()
    if total == 0:
        second = next((i for i, vector in enumerate(vectors) if not np.array_equal(vector, centers[0])), 0)
    else:
        second = int(np.searchsorted(np.cumsum(distances), rng.random_sample() * total, side="right"))
        second = min(second, len(vectors) - 1)
    centers.append(vectors[second])
    centers = np.asarray(centers, dtype=float)
    for _ in range(300):
        labels = np.sum((vectors[:, None, :] - centers[None, :, :]) ** 2, axis=2).argmin(axis=1)
        updated = np.stack([vectors[labels == c].mean(axis=0) if np.any(labels == c) else centers[c] for c in (0, 1)])
        if np.array_equal(updated, centers):
            break
        centers = updated
    return labels


def track_role_votes(rows, role_ids, ground_truth=False):
    role_names = {0: "ball", 1: "goalkeeper", 2: "other", 3: "player", 4: "referee"}
    by_track = defaultdict(list)
    for i, row in enumerate(rows):
        by_track[(row["clip"], str(row["track_id"]))].append((i, row["role"] if ground_truth else role_ids[i]))
    votes = {}
    for observations in by_track.values():
        role = Counter(value for _, value in observations).most_common(1)[0][0]
        for i, _ in observations:
            votes[i] = role if ground_truth else role_names[int(role)]
    return votes


def cluster_tracks(rows, embeddings, role_ids, ground_truth=False, oracle=False):
    by_clip = defaultdict(list)
    roles = track_role_votes(rows, role_ids, ground_truth)
    for i, row in enumerate(rows):
        by_clip[row["clip"]].append((i, row, roles[i]))
    predictions = {}
    for clip, items in by_clip.items():
        track_rows = defaultdict(list)
        for i, row, role in items:
            track_rows[str(row["track_id"])].append((i, row, role))
        players = [(track, observations) for track, observations in track_rows.items() if observations[0][2] == "player"]
        if not players:
            continue
        vectors = np.stack([embeddings[[o[0] for o in obs]].mean(axis=0) for _, obs in players])
        labels = np.zeros(len(players), dtype=np.int64) if len(players) == 1 else _kmeans_two(vectors)
        means = [np.nanmean([o[1]["pitch_x"] for (_, obs), lab in zip(players, labels) if lab == c for o in obs]) for c in (0, 1)]
        cluster_side = {0: 0 if means[0] < means[1] else 1, 1: 1 if means[0] < means[1] else 0}
        if oracle:
            costs = [[0, 0], [0, 0]]
            for (track, obs), label in zip(players, labels):
                truth = Counter(o[1]["class"] % 2 for o in obs if o[1]["class"] in (0, 1)).most_common(1)
                if truth:
                    costs[label][truth[0][0]] += len(obs)
            if costs[0][0] + costs[1][1] < costs[0][1] + costs[1][0]:
                cluster_side = {0: 1, 1: 0}
        for (track, obs), label in zip(players, labels):
            for i, row, _ in obs:
                predictions[i] = 0 if cluster_side[label] == 0 else 1
        for i, row, role in items:
            if role == "goalkeeper":
                predictions[i] = 0 if not np.isfinite(row["pitch_x"]) or row["pitch_x"] <= 0 else 1
    return predictions, roles


def evaluate(extract_path, embed_path, output):
    data = np.load(extract_path, allow_pickle=False)
    rows = json.loads(str(data["metadata"]))
    pixels, offsets = data["pixels"], data["offsets"]
    embeddings_file = np.load(embed_path, allow_pickle=False)
    embeddings, logits = embeddings_file["embeddings"], embeddings_file["role_logits"]
    embeddings = embeddings.reshape((len(embeddings), -1))
    if len(rows) != len(embeddings) or len(rows) != len(logits):
        raise ValueError("extract/embed crop counts differ")
    role_ids = logits.argmax(axis=1)
    role_names = {0: "ball", 1: "goalkeeper", 2: "other", 3: "player", 4: "referee"}
    variants = {}
    for name, gt_roles, oracle in (("V1", False, False), ("V2", True, False), ("V3", True, True)):
        assigned, voted_roles = cluster_tracks(rows, embeddings, role_ids, gt_roles, oracle)
        records = []
        for i, row in enumerate(rows):
            if row["role"] not in {"player", "goalkeeper"} or row["class"] is None:
                continue
            if row["query_index"] < 0:
                continue
            pred_role = voted_roles[i]
            if pred_role not in SLOTS or i not in assigned or not np.isfinite(row["pitch_x"]):
                pred = None
            elif pred_role == "goalkeeper":
                pred = 2 + assigned[i]
            else:
                pred = assigned[i]
            records.append((row, int(row["class"]), pred))
        all_records = []
        for i, row in enumerate(rows):
            if row["role"] not in {"player", "goalkeeper"} or row["class"] is None:
                continue
            pred_role = voted_roles[i]
            if pred_role not in SLOTS or i not in assigned or not np.isfinite(row["pitch_x"]):
                pred = None
            elif pred_role == "goalkeeper":
                pred = 2 + (1 if row["pitch_x"] > 0 else 0)
            else:
                pred = assigned[i]
            all_records.append((row, int(row["class"]), pred))
        variants[name] = {"scored_queries": _metrics(records), "all_player_goalkeeper": _metrics(all_records)}
    weights = Path("/Users/j.henrique/soccer-prtreid-baseline.3o5PLhhC/weights")
    result = {"variants": variants, "counts": {"extracted": len(rows), "query_count": sum(r["query_index"] >= 0 for r in rows), "missing_pitch_x": sum(not np.isfinite(r["pitch_x"]) for r in rows), "clips": sorted({r["clip"] for r in rows}), "sessions": {f"{r['game']}-{r['session']}": sum(x['game'] == r['game'] and x['session'] == r['session'] for x in rows) for r in rows}}, "weights_md5": {name: hashlib.md5((weights / name).read_bytes()).hexdigest() for name in ("prtreid-soccernet-baseline.pth.tar", "hrnetv2_w32_imagenet_pretrained.pth")}, "source_commits": {"src-prtreid": "30617a7", "src-sn-gamestate": "1c95834"}}
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def _metrics(records):
    matrix = np.zeros((4, 4), dtype=np.int64)
    truth_counts = np.zeros(4, dtype=np.int64)
    missing = 0
    by_session = defaultdict(lambda: {"total": 0, "matrix": np.zeros((4, 4), dtype=np.int64), "truth_counts": np.zeros(4, dtype=np.int64)})
    for row, truth, pred in records:
        session = by_session[f"{row['game']}-{row['session']}"]
        session["total"] += 1
        truth_counts[truth] += 1
        session["truth_counts"][truth] += 1
        if pred is None or pred not in range(4):
            missing += 1
            continue
        matrix[truth, pred] += 1
        session["matrix"][truth, pred] += 1
    recalls = [matrix[i, i] / truth_counts[i] if truth_counts[i] else 0.0 for i in range(4)]
    per_session = {}
    for key, session in sorted(by_session.items()):
        session_matrix = session["matrix"]
        session_recall = [session_matrix[i, i] / session["truth_counts"][i] if session["truth_counts"][i] else 0.0 for i in range(4)]
        per_session[key] = {"count": session["total"], "classified": int(session_matrix.sum()), "accuracy": float(session_matrix.trace() / session["total"]), "macro_recall": float(np.mean(session_recall)), "confusion_matrix": session_matrix.tolist()}
    return {"count": len(records), "classified": int(matrix.sum()), "missing_or_wrong": missing, "accuracy": float(matrix.trace() / len(records)) if records else None, "macro_recall": float(np.mean(recalls)), "confusion_matrix": matrix.tolist(), "per_session": per_session}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--root", required=True, type=Path)
    extract_parser.add_argument("--out", required=True, type=Path)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--extract", required=True, type=Path)
    eval_parser.add_argument("--embed", required=True, type=Path)
    eval_parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.root, args.out)
    else:
        evaluate(args.extract, args.embed, args.out)


if __name__ == "__main__":
    main()
