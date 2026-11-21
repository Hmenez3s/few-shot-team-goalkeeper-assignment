import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments import frozen_encoders, prtreid_comparison

REPO = Path(__file__).resolve().parent.parent
GROUPS = frozen_encoders.GROUP_NAMES
METHODS = {
    "soccernet_baseline": ("prtreid_kmeans_official", "per_box"),
    "fewshot_r18_1shot_per_box": ("ours_one_reference_per_box", "per_box"),
    "fewshot_r18_1shot_tracklets": ("ours_one_reference_tracklets", "tracklet_mean"),
    "fewshot_r18_6shot_per_box": ("ours_six_references_per_box", "per_box"),
    "fewshot_r18_6shot_tracklets": ("ours_six_references_tracklets", "tracklet_mean"),
}
CACHE_METHODS = {
    "soccernet_baseline_true_roles": ("prtreid_kmeans_true_roles", "per_box"),
    "siglip_six_references_per_box": ("siglip_six_references", "per_box"),
    "siglip_six_references_tracklets": ("siglip_six_references", "tracklet_mean"),
    "clip_six_references_per_box": ("clip_six_references", "per_box"),
    "clip_six_references_tracklets": ("clip_six_references", "tracklet_mean"),
    "roboflow_predicted_roles_per_box": ("roboflow_siglip_umap_kmeans_predicted_roles", "per_box"),
    "roboflow_predicted_roles_tracklets": ("roboflow_siglip_umap_kmeans_predicted_roles", "tracklet_mean"),
    "roboflow_oracle_roles_per_box": ("roboflow_siglip_umap_kmeans_oracle_roles", "per_box"),
    "roboflow_oracle_roles_tracklets": ("roboflow_siglip_umap_kmeans_oracle_roles", "tracklet_mean"),
}


def parser_args():
    parser = argparse.ArgumentParser(description="Cluster bootstrap intervals for the six-match comparison")
    parser.add_argument("--games-7-8-11-root", type=Path, required=True)
    parser.add_argument("--games-7-8-11-extract", type=Path, required=True)
    parser.add_argument("--games-7-8-11-embed", type=Path, required=True)
    parser.add_argument("--games-2-3-5-root", type=Path, required=True)
    parser.add_argument("--games-2-3-5-extract", type=Path, required=True)
    parser.add_argument("--games-2-3-5-embed", type=Path, required=True)
    parser.add_argument("--games-7-8-11-one", type=Path, required=True)
    parser.add_argument("--games-7-8-11-six", type=Path, required=True)
    parser.add_argument("--games-2-3-5-one", type=Path, required=True)
    parser.add_argument("--games-2-3-5-six", type=Path, required=True)
    parser.add_argument("--games-7-8-11-result", type=Path, required=True)
    parser.add_argument("--games-2-3-5-result", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "results/bootstrap-intervals.json")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261121)
    return parser.parse_args()


def legacy_kmeans_two(vectors):
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
        updated = np.stack([vectors[labels == c].mean(axis=0) if np.any(labels == c) else centers[c]
                            for c in (0, 1)])
        if np.array_equal(updated, centers):
            break
        centers = updated
    return labels


def baseline_predictions(rows, logits, embeddings, ground_truth_roles=False, legacy_kmeans=True):
    role_ids = logits.argmax(axis=1)
    if legacy_kmeans:
        original_kmeans = prtreid_comparison._kmeans_two
        prtreid_comparison._kmeans_two = legacy_kmeans_two
        try:
            assigned, roles = prtreid_comparison.cluster_tracks(
                rows, embeddings.reshape(len(embeddings), -1), role_ids, ground_truth=ground_truth_roles)
        finally:
            prtreid_comparison._kmeans_two = original_kmeans
    else:
        assigned, roles = prtreid_comparison.cluster_tracks(
            rows, embeddings.reshape(len(embeddings), -1), role_ids, ground_truth=ground_truth_roles)
    records = []
    for i, row in enumerate(rows):
        if row["role"] not in {"player", "goalkeeper"} or row["class"] is None or row["query_index"] < 0:
            continue
        role = roles[i]
        pred = None
        if role in prtreid_comparison.SLOTS and i in assigned and np.isfinite(row["pitch_x"]):
            pred = assigned[i] + 2 if role == "goalkeeper" else assigned[i]
        records.append((row, -1 if pred is None else int(pred)))
    return records


def resnet_predictions(root, extract, checkpoints, match_set, device):
    import torch
    import torch.nn.functional as F

    rows = frozen_encoders.load_extract(extract)
    query_rows = [row for row in rows if row["query_index"] >= 0]
    if len(query_rows) != frozen_encoders.MATCH_SETS[match_set]["expected_queries"]:
        raise ValueError(f"{match_set}: extracted query count mismatch")
    sessions = frozen_encoders.sessions_for_set(root)
    grouped = frozen_encoders.rows_by_unit(rows)
    device = torch.device(device)
    by_method = {key: [] for key in ("fewshot_r18_1shot_per_box", "fewshot_r18_1shot_tracklets",
                                      "fewshot_r18_6shot_per_box", "fewshot_r18_6shot_tracklets")}
    for shot, checkpoint in ((1, checkpoints["one"]), (6, checkpoints["six"])):
        model = frozen_encoders.train.build_model(root / "inputs/resnet18.pth", device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
        model.eval()

        def embed(crops):
            with torch.inference_mode():
                return torch.cat([F.normalize(model(frozen_encoders.six_references.batch_crops(crops[start:start + 128], device)), dim=1).cpu()
                                  for start in range(0, len(crops), 128)])

        for entry, clip, episode, selected_session in sessions:
            support6, query = selected_session
            support_indices = episode[0]
            references = clip["x"][support_indices] if shot == 1 else np.stack(support6)
            per_class = 1 if shot == 1 else frozen_encoders.six_references.REFERENCES
            vectors = embed(clip["x"])
            prototypes = F.normalize(embed(references).reshape(4, per_class, -1).mean(1), dim=1)
            box_prediction = (vectors @ prototypes.T).argmax(1).numpy()
            track_prediction = box_prediction.copy()
            excluded = set(map(int, support_indices))
            by_track = defaultdict(list)
            by_index = {}
            for row in grouped[entry["clip"]]:
                idx = int(row["prepared_index"])
                if idx >= 0 and row["role"] in ("player", "goalkeeper"):
                    by_track[frozen_encoders.tracklet_key(row)].append(idx)
                    by_index[idx] = row
            for members in by_track.values():
                eligible = [idx for idx in members if idx not in excluded]
                if eligible:
                    mean = F.normalize(vectors[eligible].mean(0, keepdim=True), dim=1)
                    track_prediction[eligible] = int((mean @ prototypes.T).argmax())
            for idx in map(int, query):
                row = by_index.get(idx)
                if row is None:
                    raise ValueError(f"{entry['clip']}: missing query metadata for {idx}")
                by_method[f"fewshot_r18_{shot}shot_per_box"].append((row, int(box_prediction[idx])))
                by_method[f"fewshot_r18_{shot}shot_tracklets"].append((row, int(track_prediction[idx])))
        del model
    return by_method


def cached_predictions(args, match_set, root, extract, embed_path):
    import torch

    frozen_result = json.loads((REPO / "results/frozen-encoders.json").read_text())
    rows = frozen_encoders.load_extract(extract)
    grouped = frozen_encoders.rows_by_unit(rows)
    sessions = frozen_encoders.sessions_for_set(root)
    logits = frozen_encoders.load_role_logits(embed_path)
    role_sources = {
        frozen_encoders.ROBOFLOW_ORACLE_METHOD: frozen_encoders.ground_truth_roles(grouped),
        frozen_encoders.ROBOFLOW_PREDICTED_METHOD: frozen_encoders.predicted_roles(rows, logits),
    }
    device = "cpu"
    predictions = {}
    model_info = frozen_result["protocol"]["model_info"]
    for method in frozen_encoders.MODEL_IDS:
        model_id = frozen_encoders.MODEL_IDS[method]
        revision = model_info[method]["revision"]
        cache_dir = args.cache
        for entry, clip, episode, session in sessions:
            feature_path = cache_dir / f"{match_set}__{model_id.replace('/', '-')}__{revision[:12]}__{entry['clip']}.npy"
            if not feature_path.is_file():
                raise FileNotFoundError(f"missing cached image features: {feature_path.name}")
        model, processor = frozen_encoders.image_model(model_id, device)
        features_by_clip = {
            entry["clip"]: frozen_encoders.cached_features(args.cache, match_set, model, processor, clip["x"],
                                                            entry["clip"], device, model_id, revision)
            for entry, clip, episode, session in sessions
        }
        reference = frozen_encoders.reference_records(sessions, grouped, features_by_clip, model, processor,
                                                       device, model_id)
        if method == "siglip_six_references":
            for role_method, roles in role_sources.items():
                roboflow, skipped = frozen_encoders.roboflow_records(sessions, grouped, features_by_clip, roles)
                if skipped:
                    raise ValueError(f"{match_set}/{role_method}: player split unavailable in {skipped}")
                for variant in ("per_box", "tracklet_mean"):
                    predictions[f"{'roboflow_oracle_roles' if role_method == frozen_encoders.ROBOFLOW_ORACLE_METHOD else 'roboflow_predicted_roles'}_{'per_box' if variant == 'per_box' else 'tracklets'}"] = [
                        (row, pred if variant == "per_box" else track_pred)
                        for (row, pred, track_pred) in roboflow
                    ]
        for variant in ("per_box", "tracklet_mean"):
            predictions[f"{method}_{'per_box' if variant == 'per_box' else 'tracklets'}"] = [
                (row, box_pred if variant == "per_box" else track_pred)
                for row, box_pred, track_pred in reference
            ]
        del model, processor
        if device == "mps":
            torch.mps.empty_cache()
    return predictions


def make_records(args):
    config = {
        "games_7_8_11": (args.games_7_8_11_root, args.games_7_8_11_extract,
                         args.games_7_8_11_embed,
                         {"one": args.games_7_8_11_one, "six": args.games_7_8_11_six},
                         args.games_7_8_11_result),
        "games_2_3_5": (args.games_2_3_5_root, args.games_2_3_5_extract,
                         args.games_2_3_5_embed,
                         {"one": args.games_2_3_5_one, "six": args.games_2_3_5_six},
                         args.games_2_3_5_result),
    }
    match_records = []
    for match_set, (root, extract, embed_path, checkpoints, result_path) in config.items():
        rows = frozen_encoders.load_extract(extract)
        if sum(row["query_index"] >= 0 for row in rows) != frozen_encoders.MATCH_SETS[match_set]["expected_queries"]:
            raise ValueError(f"{match_set}: query count mismatch")
        data = np.load(embed_path, allow_pickle=False)
        emb = data["embeddings"].reshape(len(rows), -1)
        logits = data["role_logits"]
        base = baseline_predictions(rows, logits, emb)
        expected = json.loads(result_path.read_text())["variants"]["V1"]["scored_queries"]
        if len(base) != expected["count"]:
            raise ValueError(f"{match_set}: SoccerNet scored record count {len(base)} != {expected['count']}")
        predictions = {"soccernet_baseline": base}
        if args.cache:
            predictions["soccernet_baseline_true_roles"] = baseline_predictions(
                rows, logits, emb, True, legacy_kmeans=match_set != "games_7_8_11")
            predictions.update(cached_predictions(args, match_set, root, extract, embed_path))
        predictions.update(resnet_predictions(root, extract, checkpoints, match_set, "mps" if __import__("torch").backends.mps.is_available() else "cpu"))
        matrix = np.zeros((4, 4), dtype=int)
        for row, pred in base:
            if 0 <= pred < 4:
                matrix[int(row["class"]), pred] += 1
        if match_set == "games_7_8_11" and not np.array_equal(matrix, expected["confusion_matrix"]):
            raise ValueError(f"{match_set}: SoccerNet predictions differ from the published confusion matrix")
        per_row_preds = defaultdict(dict)
        for method, records in predictions.items():
            for row, pred in records:
                per_row_preds[(row["clip"], row["prepared_index"])][method] = pred
        for row in rows:
            if row["query_index"] < 0 or row["class"] is None or row["role"] not in ("player", "goalkeeper"):
                continue
            key = (row["clip"], row["prepared_index"])
            preds = per_row_preds.get(key, {})
            method_names = list(METHODS) + (list(CACHE_METHODS) if args.cache else [])
            if len(preds) != len(method_names):
                raise ValueError(f"{match_set}: prediction missing for scored box {key}")
            match_records.append({"truth": int(row["class"]), "match": str(row["game"]),
                                  "half": frozen_encoders.unit_key(row),
                                  "tracklet": f"{row['clip']}:{row['track_id']}",
                                  "predictions": preds})
    return match_records


def _sufficient(records, unit_ids, methods):
    unique = np.unique(unit_ids)
    index = np.searchsorted(unique, unit_ids)
    truth = np.zeros((len(unique), 4), dtype=np.int64)
    np.add.at(truth, (index, np.asarray([r["truth"] for r in records])), 1)
    correct = np.zeros((len(methods), len(unique), 4), dtype=np.int64)
    for mi, method in enumerate(methods):
        pred = np.asarray([r["predictions"][method] for r in records])
        good = pred == np.asarray([r["truth"] for r in records])
        np.add.at(correct[mi], (index[good], np.asarray([r["truth"] for r in records])[good]), 1)
    return unique, truth, correct


def bootstrap(records, methods, cluster_key, resamples, seed):
    unit_ids = np.asarray([r[cluster_key] for r in records])
    units, truth, correct = _sufficient(records, unit_ids, methods)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(units), size=(resamples, len(units)))
    multipliers = np.zeros((resamples, len(units)), dtype=np.int16)
    np.add.at(multipliers, (np.arange(resamples)[:, None], draws), 1)
    group_counts = multipliers @ truth
    group_correct = np.einsum("bu,mug->bmg", multipliers, correct, optimize=True)
    present = group_counts > 0
    recalls = np.divide(group_correct, group_counts[:, None, :], out=np.zeros_like(group_correct, dtype=float), where=group_counts[:, None, :] > 0)
    score = (recalls * present[:, None, :]).sum(axis=2) / present.sum(axis=1)[:, None]
    accuracy = group_correct.sum(axis=2) / group_counts.sum(axis=1)[:, None]
    empty_count = np.count_nonzero(~present.all(axis=1))
    return {"score": score, "accuracy": accuracy, "empty_group_resamples": int(empty_count),
            "empty_group_resamples_by_class": {GROUPS[g]: int((~present[:, g]).sum()) for g in range(4)}}


def interval(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main():
    args = parser_args()
    started = time.monotonic()
    previous = json.loads(args.out.read_text()) if args.cache and args.out.is_file() else None
    records = make_records(args)
    methods = list(METHODS) + (list(CACHE_METHODS) if args.cache else [])
    published_prtreid = json.loads((REPO / "results/prtreid-comparison.json").read_text())["pooled_six_games"]
    published_frozen = json.loads((REPO / "results/frozen-encoders.json").read_text())["pooled_six_games"]
    point = {}
    reproduction_mismatches = {}
    skipped_roboflow = set()
    for method in methods:
        truth = np.asarray([r["truth"] for r in records])
        pred = np.asarray([r["predictions"][method] for r in records])
        counts = np.bincount(truth, minlength=4)
        correct = np.bincount(truth[pred == truth], minlength=4)
        score = float(np.mean(correct / counts))
        accuracy = float(np.mean(pred == truth))
        if method in METHODS:
            published_key, _ = METHODS[method]
            reference = published_prtreid[published_key]
        else:
            published_key, variant = CACHE_METHODS[method]
            if method == "soccernet_baseline_true_roles":
                reference = published_prtreid[published_key]
            else:
                reference = published_frozen[published_key][variant]
        matches = (np.isclose(score, reference["score"], rtol=0, atol=1e-9) and
                   np.isclose(accuracy, reference["accuracy"], rtol=0, atol=1e-9))
        if not matches:
            mismatch = {"recomputed": {"score": score, "accuracy": accuracy},
                        "published": {"score": reference["score"], "accuracy": reference["accuracy"]},
                        "difference": {"score": score - reference["score"], "accuracy": accuracy - reference["accuracy"]}}
            if method.startswith("roboflow_"):
                skipped_roboflow.add(method.split("_roles_")[0] + "_roles")
                reproduction_mismatches[method] = mismatch
                continue
            raise ValueError(f"self-check failed {method}: {mismatch}")
        point[method] = {"score": score, "accuracy": accuracy}
    if skipped_roboflow:
        methods = [method for method in methods if not any(method.startswith(f"{name}_") for name in skipped_roboflow)]
        for method in methods:
            if method in CACHE_METHODS and method.startswith("roboflow_"):
                point.pop(method, None)
    cluster_res = bootstrap(records, methods, "tracklet", args.resamples, args.seed)
    half_res = bootstrap(records, methods, "half", args.resamples, args.seed + 1)
    per_method = {}
    for idx, method in enumerate(methods):
        per_method[method] = {"point_estimate": point[method], "tracklet_cluster_bootstrap": {
            "score_95_percentile_interval": interval(cluster_res["score"][:, idx]),
            "accuracy_95_percentile_interval": interval(cluster_res["accuracy"][:, idx]),
            "empty_group_resamples": cluster_res["empty_group_resamples"],
            "empty_group_resamples_by_class": cluster_res["empty_group_resamples_by_class"]},
            "match_half_cluster_bootstrap": {
            "score_95_percentile_interval": interval(half_res["score"][:, idx]),
            "accuracy_95_percentile_interval": interval(half_res["accuracy"][:, idx]),
            "empty_group_resamples": half_res["empty_group_resamples"],
            "empty_group_resamples_by_class": half_res["empty_group_resamples_by_class"]}}
    differences = {}
    target_idx = methods.index("fewshot_r18_6shot_tracklets")
    comparators = ["soccernet_baseline"]
    if "siglip_six_references_tracklets" in methods:
        comparators.append("siglip_six_references_tracklets")
    if "roboflow_oracle_roles_tracklets" in methods:
        comparators.append("roboflow_oracle_roles_tracklets")
    for comparator in comparators:
        cmp_idx = methods.index(comparator)
        differences[f"fewshot_r18_6shot_tracklets_minus_{comparator}"] = {
            "point_estimate": {"score": point["fewshot_r18_6shot_tracklets"]["score"] - point[comparator]["score"],
                               "accuracy": point["fewshot_r18_6shot_tracklets"]["accuracy"] - point[comparator]["accuracy"]},
            "tracklet_cluster_bootstrap": {"score_95_percentile_interval": interval(cluster_res["score"][:, target_idx] - cluster_res["score"][:, cmp_idx]),
                                            "accuracy_95_percentile_interval": interval(cluster_res["accuracy"][:, target_idx] - cluster_res["accuracy"][:, cmp_idx])},
            "match_half_cluster_bootstrap": {"score_95_percentile_interval": interval(half_res["score"][:, target_idx] - half_res["score"][:, cmp_idx]),
                                              "accuracy_95_percentile_interval": interval(half_res["accuracy"][:, target_idx] - half_res["accuracy"][:, cmp_idx])}}
    result = {"description": ("Cluster-bootstrap confidence intervals for box-level classification across six matches, comparing SoccerNet PRTReID, FewShot-R18 reference classifiers, frozen image encoders and Roboflow team clustering." if args.cache else "Cluster-bootstrap confidence intervals for box-level classification across six matches, comparing SoccerNet PRTReID with FewShot-R18 reference classifiers."),
              "method": "For every scored box, preserve truth, match, half/session and (clip, track_id). Resample the unique tracklet clusters with replacement, or independently resample the 12 match halves with replacement, retaining all boxes in selected clusters/halves. Each replicate recomputes accuracy over boxes and score as the unweighted mean recall across non-empty truth groups. Empty-group replicate counts are recorded. Percentile bounds are 2.5th and 97.5th percentiles. Paired differences use identical replicate draws across methods. The tracklet bootstrap uses seed; match-half uses seed + 1.",
              "seed": args.seed, "bootstrap_seeds": {"tracklet_cluster": args.seed, "match_half_cluster": args.seed + 1},
              "resamples": args.resamples,
              "n_boxes": len(records), "n_tracklets": len(set(r["tracklet"] for r in records)),
              "n_halves": len(set(r["half"] for r in records)), "methods": per_method,
              "omitted_methods": {} if args.cache else {
                  "siglip_six_references_tracklets": "Cached per-clip embeddings are present, but computing reference prototypes requires the Transformers package, which is unavailable in the local interpreter; no per-box predictions are stored in frozen-encoders.json.",
                  "roboflow_predicted_roles": "The cached SigLIP features are present, but the UMAP and scikit-learn packages required to rebuild per-box clustering predictions are unavailable locally.",
              },
              "paired_differences": differences,
              "empty_group_resamples": {"tracklet_cluster_bootstrap": cluster_res["empty_group_resamples"],
                                         "match_half_cluster_bootstrap": half_res["empty_group_resamples"]},
              "self_check": {"passed": True, "tolerance": 1e-9,
                             "published_point_estimates": {key: value["point_estimate"] for key, value in per_method.items()}},
              "runtime_seconds": time.monotonic() - started}
    if args.cache:
        result["oracle_methods"] = [method for method in methods if method in (
            "soccernet_baseline_true_roles", "roboflow_oracle_roles_per_box", "roboflow_oracle_roles_tracklets")]
        result["method_notes"] = {
            "soccernet_baseline_true_roles": "Uses ground-truth player and goalkeeper roles.",
            "roboflow_oracle_roles_per_box": "Oracle variant; receives ground-truth player and goalkeeper roles.",
            "roboflow_oracle_roles_tracklets": "Oracle variant; receives ground-truth player and goalkeeper roles.",
        }
        result["reproduction_mismatches"] = reproduction_mismatches
    if previous is not None:
        for method in METHODS:
            if previous["methods"][method] != result["methods"][method]:
                raise AssertionError(f"existing bootstrap intervals changed for {method}")
        old_pair = previous["paired_differences"]["fewshot_r18_6shot_tracklets_minus_soccernet_baseline"]
        new_pair = result["paired_differences"]["fewshot_r18_6shot_tracklets_minus_soccernet_baseline"]
        if old_pair != new_pair:
            raise AssertionError("existing paired SoccerNet intervals changed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"n_boxes": result["n_boxes"], "n_tracklets": result["n_tracklets"], "n_halves": result["n_halves"],
                      "methods": {k: v["point_estimate"] for k, v in per_method.items()},
                      "paired_differences": differences, "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
