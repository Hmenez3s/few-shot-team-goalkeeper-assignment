import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import frozen_encoders, research_references, transfer_datasets

METHODS = (
    "fewshot_r18_1shot",
    "fewshot_r18_6shot",
    "imagenet_r18",
    "siglip",
    "clip",
    "osnet",
    "hsv_torso",
    "roboflow_siglip_umap_kmeans",
)
EMBEDDING_METHODS = METHODS[:6]
GROUP_NAMES = ("team_0", "team_1")
MODEL_CHECKPOINTS = {
    "fewshot_r18_1shot": REPO / "models/one-reference.pth",
    "fewshot_r18_6shot": REPO / "models/six-references.pth",
}
MODEL_IDS = {
    "siglip": frozen_encoders.MODEL_IDS["siglip_six_references"],
    "clip": frozen_encoders.MODEL_IDS["clip_six_references"],
}


def load_clips(work_dir):
    clips = []
    for path in sorted((work_dir / "wide_view").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            required = ("x", "y", "track", "frame", "frame_count", "box_height")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{path}: missing crop arrays: {', '.join(missing)}")
            clip = {key: data[key] for key in required}
        crops = clip["x"]
        count = len(crops)
        if crops.dtype != np.uint8 or crops.shape != (count, 96, 48, 3):
            raise ValueError(f"{path}: x must be uint8 [N,96,48,3]")
        for key in ("y", "track", "frame", "box_height"):
            if len(clip[key]) != count:
                raise ValueError(f"{path}: {key} length does not match x")
        if count == 0 or int(clip["frame_count"]) < 1:
            raise ValueError(f"{path}: clip has no crops or invalid frame_count")
        if not np.isin(clip["y"], (0, 1)).all():
            raise ValueError(f"{path}: y must contain only team labels 0 and 1")
        clips.append({"name": path.stem, **clip})
    if not clips:
        raise FileNotFoundError(f"no cached clips found in {work_dir / 'wide_view'}")
    return clips


def make_draws(clip, draw_count, rng):
    labels = np.asarray(clip["y"], dtype=np.int64)
    tracks = np.asarray(clip["track"], dtype=np.int64)
    frames = np.asarray(clip["frame"], dtype=np.int64)
    early = frames < 0.30 * int(clip["frame_count"])
    later = ~early
    eligible_by_team = {}
    early_by_team = {}
    query_by_team = {}
    for team in (0, 1):
        early_tracks = sorted(np.unique(tracks[early & (labels == team)]).tolist())
        eligible = [track_id for track_id in early_tracks
                    if np.count_nonzero(early & (labels == team) & (tracks == track_id)) >= 6]
        eligible_by_team[team] = eligible
        early_by_team[team] = int(np.count_nonzero(early & (labels == team)))
        query_by_team[team] = int(np.count_nonzero(later & (labels == team)))

    if any(not eligible_by_team[team] for team in (0, 1)):
        return None, {
            "clip": clip["name"],
            "early_boxes_by_team": [early_by_team[0], early_by_team[1]],
            "eligible_athletes_by_team": [len(eligible_by_team[0]), len(eligible_by_team[1])],
            "query_boxes_by_team_before_reference_exclusion": [query_by_team[0], query_by_team[1]],
            "reason": "team_without_athlete_with_six_early_boxes",
        }

    draws = []
    for _ in range(draw_count):
        ref_tracks = {}
        refs_six = {}
        refs_one = {}
        for team in (0, 1):
            ref_track = int(rng.choice(eligible_by_team[team]))
            candidates = np.flatnonzero(early & (labels == team) & (tracks == ref_track))
            six = np.asarray(rng.choice(candidates, size=6, replace=False), dtype=np.int64)
            one = int(rng.choice(six))
            ref_tracks[team] = ref_track
            refs_six[team] = six
            refs_one[team] = np.asarray([one], dtype=np.int64)
        excluded = ((labels == 0) & (tracks == ref_tracks[0])) | ((labels == 1) & (tracks == ref_tracks[1]))
        query = np.flatnonzero(later & ~excluded)
        draw = {
            "reference_tracks": [ref_tracks[0], ref_tracks[1]],
            "references_six": [refs_six[0], refs_six[1]],
            "references_one": [refs_one[0], refs_one[1]],
            "query": query,
            "query_boxes_by_team": [int(np.count_nonzero(labels[query] == team)) for team in (0, 1)],
        }
        draws.append(draw)
    used_query_by_team = draws[0]["query_boxes_by_team"]
    return {
        "early": early,
        "later": later,
        "draws": draws,
        "query_boxes_by_team": used_query_by_team,
        "eligible_by_team": [len(eligible_by_team[0]), len(eligible_by_team[1])],
        "early_boxes_by_team": [early_by_team[0], early_by_team[1]],
    }, None


def torch_device(torch):
    return transfer_datasets.preferred_device(torch)


def load_embedding_methods(methods, osnet_path, device):
    models = {}
    metadata = {}
    require_siglip = "roboflow_siglip_umap_kmeans" in methods
    transformer_methods = [name for name in ("siglip", "clip") if name in methods]
    if require_siglip and "siglip" not in transformer_methods:
        transformer_methods.append("siglip")
    imagenet_path = REPO / "data/inputs/resnet18.pth"
    for method in ("fewshot_r18_1shot", "fewshot_r18_6shot", "imagenet_r18"):
        if method in methods:
            checkpoint = MODEL_CHECKPOINTS.get(method)
            models[method] = transfer_datasets.load_encoder(checkpoint, imagenet_path, device)
            metadata[method] = {
                "checkpoint": str(checkpoint) if checkpoint else str(imagenet_path),
                "model_id": "torchvision/resnet18",
                "revision": None,
            }
    if "osnet" in methods and osnet_path is not None:
        models["osnet"] = research_references.reid_encoder(osnet_path, device)
        metadata["osnet"] = {"checkpoint": str(osnet_path.resolve()), "model_id": None, "revision": None}
    elif "osnet" in methods:
        metadata["osnet"] = {"status": "omitted", "reason": "--osnet was not provided"}
    if transformer_methods:
        os.environ["HF_HUB_OFFLINE"] = "1"
        for method in transformer_methods:
            model_id = MODEL_IDS[method]
            revision = frozen_encoders.model_revision(model_id)
            model, processor = frozen_encoders.image_model(model_id, device)
            models[method] = (model, processor, model_id)
            metadata[method] = {"model_id": model_id, "revision": revision}
    return models, metadata


def encode_method(method, model, crops, device):
    if method.startswith("fewshot_r18_") or method == "imagenet_r18":
        return transfer_datasets.embed_crops(model, crops, device)
    if method in MODEL_IDS:
        encoder, processor, model_id = model
        return frozen_encoders.encode(encoder, processor, crops, device, model_id)
    if method == "osnet":
        return research_references.encode_reid(model, crops, device)
    raise ValueError(f"unsupported embedding method: {method}")


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def prototypes(features, reference_indices):
    return normalize_rows(np.stack([
        normalize_rows(features[indices]).mean(axis=0) for indices in reference_indices
    ]))


def tracklet_means(features, query_indices, tracks):
    query_tracks = tracks[query_indices]
    groups = []
    for track_id in np.unique(query_tracks):
        indices = query_indices[query_tracks == track_id]
        groups.append((indices, normalize_rows(features[indices]).mean(axis=0)))
    means = np.stack([mean for _, mean in groups])
    return groups, normalize_rows(means)


def assign_clusters(cluster_means, reference_prototypes):
    similarities = normalize_rows(cluster_means) @ normalize_rows(reference_prototypes).T
    direct = similarities[0, 0] + similarities[1, 1]
    swapped = similarities[0, 1] + similarities[1, 0]
    return np.asarray([0, 1] if direct >= swapped else [1, 0], dtype=np.int64)


def metric(truth, prediction):
    counts = np.bincount(truth, minlength=2)
    correct = np.bincount(truth[prediction == truth], minlength=2)
    per_team = np.divide(correct, counts, out=np.zeros(2, dtype=float), where=counts > 0)
    return {
        "score": float(per_team.mean()),
        "accuracy": float(correct.sum() / len(truth)) if len(truth) else 0.0,
        "per_team_accuracy": {GROUP_NAMES[i]: float(per_team[i]) for i in (0, 1)},
        "counts": {GROUP_NAMES[i]: int(counts[i]) for i in (0, 1)},
    }


def evaluate_methods(clips, plans, methods, models, device, draw_count):
    predictions = {
        method: {k: {variant: [[] for _ in range(draw_count)]
                     for variant in ("per_box", "tracklet_mean")} for k in (1, 6)}
        for method in methods
    }
    cluster_state = {}
    for clip, plan in zip(clips, plans):
        if plan is None:
            continue
        crops = clip["x"]
        tracks = np.asarray(clip["track"], dtype=np.int64)
        labels = np.asarray(clip["y"], dtype=np.int64)
        features = {}
        methods_to_encode = list(methods)
        if "roboflow_siglip_umap_kmeans" in methods and "siglip" not in methods_to_encode:
            methods_to_encode.append("siglip")
        for method in methods_to_encode:
            if method in EMBEDDING_METHODS:
                features[method] = encode_method(method, models[method], crops, device)
        colors = None
        if "hsv_torso" in methods:
            colors = np.stack([research_references.color_vector(crop) for crop in crops])
        if "roboflow_siglip_umap_kmeans" in methods:
            import umap
            from sklearn.cluster import KMeans

            siglip = features["siglip"]
            later_indices = np.flatnonzero(plan["later"])
            groups = [later_indices[tracks[later_indices] == track_id]
                      for track_id in np.unique(tracks[later_indices])]
            track_vectors = np.stack([siglip[indices].mean(axis=0) for indices in groups])
            reducer = umap.UMAP(n_components=3, random_state=0)
            projection = reducer.fit_transform(track_vectors)
            cluster_ids = KMeans(n_clusters=2, random_state=0).fit_predict(projection)
            cluster_state[clip["name"]] = {
                int(cluster): normalize_rows(np.mean(track_vectors[cluster_ids == cluster], axis=0, keepdims=True))[0]
                for cluster in (0, 1)
            }
            track_to_cluster = {
                int(tracks[indices[0]]): int(cluster)
                for indices, cluster in zip(groups, cluster_ids)
            }
            cluster_state[clip["name"]]["track_to_cluster"] = track_to_cluster

        for draw_index, draw in enumerate(plan["draws"]):
            query = draw["query"]
            truth = labels[query]
            for k in (1, 6):
                reference_indices = draw["references_one"] if k == 1 else draw["references_six"]
                for method in methods:
                    if method == "roboflow_siglip_umap_kmeans":
                        cluster_means = np.stack([
                            cluster_state[clip["name"]][cluster] for cluster in (0, 1)
                        ])
                        mapping = assign_clusters(cluster_means, prototypes(features["siglip"], reference_indices))
                        track_to_cluster = cluster_state[clip["name"]]["track_to_cluster"]
                        pred = np.asarray([
                            mapping[track_to_cluster[int(track_id)]] for track_id in tracks[query]
                        ], dtype=np.int64)
                        records = {"per_box": pred, "tracklet_mean": pred}
                    elif method == "hsv_torso":
                        centers = np.stack([colors[indices].mean(axis=0) for indices in reference_indices])
                        per_box = ((colors[query, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
                        by_track = [query[tracks[query] == track_id] for track_id in np.unique(tracks[query])]
                        track_pred = np.empty(len(query), dtype=np.int64)
                        for indices in by_track:
                            mean = colors[indices].mean(axis=0)
                            team = int(((mean[None, :] - centers) ** 2).sum(axis=1).argmin())
                            track_pred[np.isin(query, indices)] = team
                        records = {"per_box": per_box, "tracklet_mean": track_pred}
                    else:
                        vectors = features[method]
                        proto = prototypes(vectors, reference_indices)
                        per_box = (normalize_rows(vectors[query]) @ proto.T).argmax(axis=1)
                        track_pred = np.empty(len(query), dtype=np.int64)
                        by_track, means = tracklet_means(vectors, query, tracks)
                        mean_predictions = (means @ proto.T).argmax(axis=1)
                        for (indices, _), prediction in zip(by_track, mean_predictions):
                            track_pred[np.isin(query, indices)] = prediction
                        records = {"per_box": per_box, "tracklet_mean": track_pred}
                    for variant in ("per_box", "tracklet_mean"):
                        predictions[method][k][variant][draw_index].append(
                            (truth.copy(), records[variant].copy())
                        )
    draw_metrics = {
        method: {k: {variant: [] for variant in ("per_box", "tracklet_mean")} for k in (1, 6)}
        for method in methods
    }
    for method, shot_predictions in predictions.items():
        for k, variants in shot_predictions.items():
            for variant, draws in variants.items():
                for rows in draws:
                    truth = np.concatenate([row[0] for row in rows]) if rows else np.empty(0, dtype=np.int64)
                    prediction = np.concatenate([row[1] for row in rows]) if rows else np.empty(0, dtype=np.int64)
                    draw_metrics[method][k][variant].append(metric(truth, prediction))
    return draw_metrics


def summarize_metrics(draw_metrics):
    results = {}
    for method, shot_records in draw_metrics.items():
        results[method] = {}
        for k, variants in shot_records.items():
            results[method][f"{k}_shot"] = {}
            for variant, records in variants.items():
                scores = [record["score"] for record in records]
                accuracies = [record["accuracy"] for record in records]
                results[method][f"{k}_shot"][variant] = {
                    "score_mean": float(np.mean(scores)),
                    "score_sample_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                    "accuracy_mean": float(np.mean(accuracies)),
                    "accuracy_sample_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
                    "draws": records,
                }
    return results


def package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def parser_args(argv=None):
    parser = argparse.ArgumentParser(description="Paired SoccerTrack transfer baseline evaluation")
    parser.add_argument("--work-dir", type=Path, required=True, help="directory containing wide_view/*.npz")
    parser.add_argument("--output", type=Path, default=Path("results/transfer-soccertrack.json"))
    parser.add_argument("--osnet", type=Path)
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    args = parser.parse_args(argv)
    if args.draws < 2:
        parser.error("--draws must be at least 2 to report sample standard deviations")
    args.methods = list(dict.fromkeys(args.methods or METHODS))
    if args.osnet is not None and not args.osnet.is_file():
        parser.error("--osnet must name an existing TorchScript file")
    return args


def main(argv=None):
    args = parser_args(argv)
    started = time.monotonic()
    import torch

    device = torch_device(torch)
    clips = load_clips(args.work_dir.resolve())
    rng = np.random.default_rng(args.seed)
    plans, skipped = [], []
    used = []
    query_counts_by_draw = np.zeros((args.draws, 2), dtype=np.int64)
    for clip in clips:
        plan, skip = make_draws(clip, args.draws, rng)
        if skip:
            skipped.append(skip)
            plans.append(None)
            continue
        plans.append(plan)
        for draw_index, draw in enumerate(plan["draws"]):
            query_counts_by_draw[draw_index] += draw["query_boxes_by_team"]
        used.append({
            "clip": clip["name"],
            "early_boxes_by_team": plan["early_boxes_by_team"],
            "eligible_athletes_by_team": plan["eligible_by_team"],
            "query_boxes_by_team": {
                GROUP_NAMES[team]: {
                    "mean_across_draws": float(np.mean([draw["query_boxes_by_team"][team]
                                                         for draw in plan["draws"]])),
                    "min": min(draw["query_boxes_by_team"][team] for draw in plan["draws"]),
                    "max": max(draw["query_boxes_by_team"][team] for draw in plan["draws"]),
                } for team in (0, 1)
            },
            "draws": [{
                "reference_tracks": draw["reference_tracks"],
                "query_boxes_by_team": draw["query_boxes_by_team"],
                "references_one": [indices.tolist() for indices in draw["references_one"]],
                "references_six": [indices.tolist() for indices in draw["references_six"]],
                "query_indices": draw["query"].tolist(),
            } for draw in plan["draws"]],
        })
    methods = list(args.methods)
    models, model_info = load_embedding_methods(methods, args.osnet, device)
    evaluated_methods = [method for method in methods
                         if method not in EMBEDDING_METHODS or method in models]
    draw_metrics = evaluate_methods(clips, plans, evaluated_methods, models, device, args.draws)
    results = summarize_metrics(draw_metrics)
    if "osnet" in model_info and model_info["osnet"].get("status") == "omitted":
        results["osnet"] = {"status": "omitted", "reason": "--osnet was not provided"}
    else:
        model_info.setdefault("osnet", {"status": "not_requested"})
    output = {
        "description": "Athlete-level transfer comparison of FewShot-R18 and appearance/vision baselines on SoccerTrack v1 wide-view clips.",
        "provenance": {
            "dataset": "SoccerTrack v1",
            "url": "https://github.com/AtomScott/SoccerTrack",
            "license": "LGPL-3.0 per Kaggle listing",
            "recording_structure": "One recording split into 30-second clips; wide 8K fisheye view.",
            "clip_count_expected": 12,
            "goalkeeper_labels": False,
            "median_box_height": float(np.median(np.concatenate([clip["box_height"] for clip in clips]))),
            "box_height_unit": "source-image pixels, as stored in cached crop metadata",
        },
        "protocol": "Per clip, early boxes have frame < 0.30 * frame_count and all remaining boxes are later. For each seeded draw, choose one athlete uniformly per team among tracks with at least six early boxes; choose six distinct early boxes from that athlete and one of those six for the nested 1-shot reference. Queries are all later boxes of every track except the two selected reference athletes, identical for 1-shot and 6-shot. Draw indices are generated once and reused for every method. Clips missing an eligible athlete on either team are skipped and reported with counts. Embedding prototypes average L2-normalised reference vectors and are renormalised; cosine argmax predicts team. HSV uses the nearest mean torso histogram by squared L2, with tracklet predictions based on the mean histogram per later query track. Tracklet-mean variants average L2-normalised later-box embeddings per track and score that prediction over the track's boxes. Within each draw, counts are pooled across clips before score (mean of the two team accuracies) and plain accuracy are computed; reported means and sample standard deviations are over draws. Roboflow-style UMAP(3, random_state=0) and KMeans(2, random_state=0) are fit once per clip to later query-track mean SigLIP features; because SoccerTrack has no pitch coordinates, cluster labels are named separately for each draw and shot count by the 2x2 assignment maximising cosine similarity to the two SigLIP reference prototypes. Roboflow is tracklet-level only and its output is reported under both variants.",
        "clips_used": used,
        "clips_skipped": skipped,
        "query_boxes_per_team": {
            GROUP_NAMES[i]: {
                "mean_across_draws": float(query_counts_by_draw[:, i].mean()),
                "min": int(query_counts_by_draw[:, i].min()),
                "max": int(query_counts_by_draw[:, i].max()),
            } for i in (0, 1)
        },
        "query_boxes_per_team_by_draw": query_counts_by_draw.tolist(),
        "per_draw_query_boxes_per_team": "Counts pooled over usable clips for each draw; identical across methods and k.",
        "results": results,
        "models": model_info,
        "versions": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface-hub"),
            "umap-learn": package_version("umap-learn"),
            "scikit-learn": package_version("scikit-learn"),
            "pillow": package_version("Pillow"),
        },
        "runtime_seconds": time.monotonic() - started,
        "device": str(device),
        "draw_count": args.draws,
        "seed": args.seed,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
