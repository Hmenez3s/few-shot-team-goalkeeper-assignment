import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import six_references
import train
from experiments import prtreid_comparison

REPO = Path(__file__).resolve().parent.parent
GROUP_NAMES = ("team_a", "team_b", "goalkeeper_a", "goalkeeper_b")
MODEL_IDS = {
    "siglip_six_references": "google/siglip-base-patch16-224",
    "clip_six_references": "openai/clip-vit-base-patch32",
}
ROBOFLOW_ORACLE_METHOD = "roboflow_siglip_umap_kmeans_oracle_roles"
ROBOFLOW_PREDICTED_METHOD = "roboflow_siglip_umap_kmeans_predicted_roles"
ROBOFLOW_SPORTS_COMMIT = "42c80c06b6b65a7f89455b89fe31cdf4c38ba227"
CACHE_DIR = Path("/Users/j.henrique/soccer-frozen-encoders/cache")
MATCH_SETS = {
    "games_7_8_11": {
        "root": Path("/Users/j.henrique/soccer-gsr-role-audited.jfj8rWrx"),
        "extract": Path("/Users/j.henrique/soccer-prtreid-baseline.3o5PLhhC/run-compare/extract/extract.npz"),
        "embed": Path("/Users/j.henrique/soccer-prtreid-baseline.3o5PLhhC/run-compare/embed.npz"),
        "expected_queries": 2659,
        "resnet_checkpoint": REPO / "models/six-references.pth",
        "resnet_expected": {"per_box": "ours_six_references_per_box", "tracklet_mean": "ours_six_references_tracklets"},
    },
    "games_2_3_5": {
        "root": Path("/Users/j.henrique/soccer-gsr-fold1-compare.8rkNtZ2O"),
        "extract": Path("/Users/j.henrique/soccer-gsr-fold1-compare.8rkNtZ2O/compare/extract/extract.npz"),
        "embed": Path("/Users/j.henrique/soccer-gsr-fold1-compare.8rkNtZ2O/compare/embed.npz"),
        "expected_queries": 3183,
        "resnet_checkpoint": Path("/Users/j.henrique/soccer-gsr-fold1-compare.8rkNtZ2O/work/six-references/checkpoint.pth"),
        "resnet_expected": {"per_box": "ours_six_references_per_box", "tracklet_mean": "ours_six_references_tracklets"},
    },
}
PUBLISHED_COMPARISON = REPO / "results/prtreid-comparison.json"


def unit_key(row):
    return f"game-{row['game']}-half-{row['session']}"


def tracklet_key(row):
    return (row["clip"], str(row["track_id"]))


def cluster_means(cluster_ids, pitch_x):
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    pitch_x = np.asarray(pitch_x, dtype=float)
    return np.asarray([
        np.nanmean(pitch_x[(cluster_ids == cluster) & np.isfinite(pitch_x)])
        if np.any((cluster_ids == cluster) & np.isfinite(pitch_x)) else np.nan
        for cluster in (0, 1)
    ])

def sides_from_means(cluster_ids, means):
    left_cluster = int(np.argmin(means))
    return np.where(np.asarray(cluster_ids, dtype=np.int64) == left_cluster, 0, 1)

def map_player_clusters(cluster_ids, pitch_x):
    means = cluster_means(cluster_ids, pitch_x)
    if not np.isfinite(means).all():
        raise ValueError("both player clusters need finite pitch positions")
    return sides_from_means(cluster_ids, means), means


def assign_goalkeepers(pitch_x, team_centroids):
    pitch_x = np.asarray(pitch_x, dtype=float)
    team_centroids = np.asarray(team_centroids, dtype=float)
    predicted = np.full(len(pitch_x), -1, dtype=np.int64)
    valid = np.isfinite(pitch_x) & np.isfinite(team_centroids).all()
    predicted[valid] = np.abs(pitch_x[valid, None] - team_centroids[None, :]).argmin(axis=1)
    return predicted


def score_predictions(truth, prediction):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if truth.shape != prediction.shape or np.any((truth < 0) | (truth > 3)):
        raise ValueError("truth and prediction must be aligned class ids in [0, 3]")
    matrix = np.zeros((4, 4), dtype=np.int64)
    counts = np.bincount(truth, minlength=4)
    correct = np.zeros(4, dtype=np.int64)
    for target, predicted in zip(truth, prediction):
        if 0 <= predicted < 4:
            matrix[target, predicted] += 1
            correct[target] += target == predicted
    accuracies = np.divide(correct, counts, out=np.zeros(4, dtype=float), where=counts > 0)
    return {
        "score": float(accuracies.mean()),
        "accuracy": float(correct.sum() / len(truth)) if len(truth) else 0.0,
        "per_group": dict(zip(GROUP_NAMES, accuracies.tolist())),
        "truth_counts": dict(zip(GROUP_NAMES, counts.astype(int).tolist())),
        "correct": dict(zip(GROUP_NAMES, correct.astype(int).tolist())),
        "confusion_matrix": matrix.tolist(),
        "unassigned_boxes": int(np.count_nonzero((prediction < 0) | (prediction > 3))),
    }


def summarize(records, variant):
    truth, prediction = [], []
    for row, box_prediction, tracklet_prediction in records:
        truth.append(row["class"])
        prediction.append(box_prediction if variant == "per_box" else tracklet_prediction)
    return score_predictions(truth, prediction)


def load_extract(path):
    data = np.load(path, allow_pickle=False)
    return json.loads(str(data["metadata"]))


def sessions_for_set(root):
    _, splits, _ = train.load_manifest(root)
    cache = train.ClipCache(capacity=12)
    sessions = six_references.sessions_for(splits["test"], cache, root / "work" / "references")
    rng = random.Random(six_references.EVALUATION_SEED)
    return [(entry, clip, episode, six_references.make_session(clip, bank, episode, rng, True))
            for entry, clip, bank, episode in sessions]


def rows_by_unit(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[unit_key(row)].append(row)
    return grouped


def image_model(model_id, device):
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection, SiglipVisionModel

    if model_id == MODEL_IDS["siglip_six_references"]:
        model = SiglipVisionModel.from_pretrained(model_id).to(device).eval()
    else:
        model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device).eval()
    return model, AutoImageProcessor.from_pretrained(model_id)


def model_revision(model_id):
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    revision = (cache / "refs" / "main").read_text().strip()
    if not (cache / "snapshots" / revision).is_dir():
        raise ValueError(f"{model_id}: cached snapshot for {revision} is missing")
    return revision


def encode(model, processor, crops, device, model_id, batch_size=64):
    import torch
    import torch.nn.functional as F

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = [Image.fromarray(crop) for crop in crops[start:start + batch_size]]
            encoded = model(**processor(images=batch, return_tensors="pt").to(device))
            features = encoded.last_hidden_state.mean(dim=1) if model_id == MODEL_IDS["siglip_six_references"] else encoded.image_embeds
            chunks.append(F.normalize(features, dim=1).cpu().numpy())
    if not chunks:
        raise ValueError("no crops to encode")
    return np.concatenate(chunks, axis=0)


def cached_features(cache_dir, match_set, model, processor, crops, clip_id, device, model_id, revision):
    path = cache_dir / f"{match_set}__{model_id.replace('/', '-')}__{revision[:12]}__{clip_id}.npy"
    if path.is_file():
        features = np.load(path)
        if features.shape[0] != len(crops):
            raise ValueError(f"{path}: cached embeddings do not match {len(crops)} crops")
        return features
    features = encode(model, processor, crops, device, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as stream:
        np.save(stream, features)
    pending.replace(path)
    return features

def ground_truth_roles(grouped):
    roles = {}
    for clip, rows in grouped.items():
        roles[clip] = {}
        for row in rows:
            index = int(row["prepared_index"])
            if index < 0:
                continue
            if index in roles[clip]:
                raise ValueError(f"{clip}: duplicate extraction row for prepared index {index}")
            roles[clip][index] = row["role"]
    return roles

def predicted_roles(rows, role_logits):
    if len(role_logits) != len(rows):
        raise ValueError("role logits and extraction metadata have different row counts")
    votes = prtreid_comparison.track_role_votes(rows, role_logits.argmax(axis=1))
    roles = defaultdict(dict)
    for position, row in enumerate(rows):
        index = int(row["prepared_index"])
        if index < 0:
            continue
        clip = unit_key(row)
        if index in roles[clip]:
            raise ValueError(f"{clip}: duplicate extraction row for prepared index {index}")
        roles[clip][index] = votes[position]
    return roles

def load_role_logits(path):
    return np.load(path, allow_pickle=False)["role_logits"]

def reference_records(sessions, grouped, features_by_clip, model, processor, device, model_id):
    records = []
    for entry, clip, episode, session in sessions:
        support, query = session
        features = features_by_clip[entry["clip"]]
        support_features = encode(model, processor, np.asarray(support), device, model_id)
        prototypes = support_features.reshape(4, six_references.REFERENCES, -1).mean(axis=1)
        prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
        box_prediction = (features @ prototypes.T).argmax(axis=1)
        excluded = set(map(int, episode[0]))
        by_track = defaultdict(list)
        by_index = {}
        for row in grouped[entry["clip"]]:
            index = int(row["prepared_index"])
            if index < 0 or row["role"] not in ("player", "goalkeeper"):
                continue
            by_track[tracklet_key(row)].append(index)
            by_index[index] = row
        tracklet_prediction = box_prediction.copy()
        for members in by_track.values():
            eligible = [index for index in members if index not in excluded]
            if eligible:
                mean = features[eligible].mean(axis=0)
                tracklet_prediction[eligible] = np.argmax((mean / np.linalg.norm(mean)) @ prototypes.T)
        for index in map(int, query):
            row = by_index.get(index)
            if row is None:
                raise ValueError(f"{entry['clip']}: no extraction metadata for prepared query {index}")
            records.append((row, int(box_prediction[index]), int(tracklet_prediction[index])))
    return records


def roboflow_records(sessions, grouped, features_by_clip, roles_by_clip):
    import umap
    from sklearn.cluster import KMeans

    records, undivided = [], []
    for entry, clip, episode, session in sessions:
        features = features_by_clip[entry["clip"]]
        roles = roles_by_clip[entry["clip"]]
        by_index = {}
        player_tracks = defaultdict(list)
        for row in grouped[entry["clip"]]:
            index = int(row["prepared_index"])
            if index < 0:
                continue
            by_index[index] = row
            if roles.get(index) == "player":
                player_tracks[tracklet_key(row)].append(index)
        track_vectors, track_x = [], []
        for indices in player_tracks.values():
            visible = [index for index in indices if np.isfinite(by_index[index]["pitch_x"])]
            if visible:
                track_vectors.append(features[visible].mean(axis=0))
                track_x.append(float(np.mean([by_index[index]["pitch_x"] for index in visible])))
        predicted = {}
        if len(track_vectors) >= 2:
            reducer = umap.UMAP(n_components=3, random_state=0)
            projection = reducer.fit_transform(np.asarray(track_vectors))
            cluster_model = KMeans(n_clusters=2, random_state=0)
            track_clusters = cluster_model.fit_predict(projection)
            means = cluster_means(track_clusters, np.asarray(track_x))
            if not np.isfinite(means).all():
                undivided.append(entry["clip"])
            else:
                track_sides = sides_from_means(track_clusters, means)
                side_by_cluster = np.asarray([track_sides[np.flatnonzero(track_clusters == cluster)[0]] for cluster in (0, 1)])
                box_clusters = cluster_model.predict(reducer.transform(features))
                player_indices = np.asarray([index for index, row in by_index.items()
                                             if roles.get(index) == "player" and np.isfinite(row["pitch_x"])],
                                            dtype=np.int64)
                player_sides = side_by_cluster[box_clusters[player_indices]]
                centroids = np.asarray([
                    np.mean([by_index[index]["pitch_x"] for index, side in zip(player_indices, player_sides) if side == team])
                    if np.any(player_sides == team) else np.nan
                    for team in (0, 1)
                ])
                goalkeeper_indices = np.asarray([index for index in by_index if roles.get(index) == "goalkeeper"],
                                                dtype=np.int64)
                goalkeeper_sides = assign_goalkeepers(
                    np.asarray([by_index[index]["pitch_x"] for index in goalkeeper_indices]), centroids)
                predicted = {int(index): int(side) for index, side in zip(player_indices, player_sides)}
                predicted.update({int(index): int(side + 2) if side >= 0 else -1
                                  for index, side in zip(goalkeeper_indices, goalkeeper_sides)})
        else:
            undivided.append(entry["clip"])
        for index in map(int, session[1]):
            row = by_index.get(index)
            if row is None:
                raise ValueError(f"{entry['clip']}: no extraction metadata for prepared query {index}")
            records.append((row, predicted.get(index, -1), predicted.get(index, -1)))
    return records, undivided


def resnet_records(root, checkpoint, sessions, grouped, device):
    import torch
    import torch.nn.functional as F

    model = train.build_model(root / "inputs/resnet18.pth", device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.eval()

    def embed(crops):
        with torch.inference_mode():
            return torch.cat([F.normalize(model(six_references.batch_crops(crops[start:start + 128], device)), dim=1).cpu()
                              for start in range(0, len(crops), 128)])

    records = []
    for entry, clip, episode, session in sessions:
        support, query = session
        prototypes = F.normalize(embed(np.stack(support)).reshape(4, six_references.REFERENCES, -1).mean(1), dim=1)
        vectors = embed(clip["x"])
        box_prediction = (vectors @ prototypes.T).argmax(1).numpy()
        excluded = set(map(int, episode[0]))
        by_track = defaultdict(list)
        by_index = {}
        for row in grouped[entry["clip"]]:
            index = int(row["prepared_index"])
            if index < 0 or row["role"] not in ("player", "goalkeeper"):
                continue
            by_track[tracklet_key(row)].append(index)
            by_index[index] = row
        tracklet_prediction = box_prediction.copy()
        for members in by_track.values():
            eligible = [index for index in members if index not in excluded]
            if eligible:
                mean = F.normalize(vectors[eligible].mean(0, keepdim=True), dim=1)
                tracklet_prediction[eligible] = int((mean @ prototypes.T).argmax())
        for index in map(int, query):
            row = by_index.get(index)
            if row is None:
                raise ValueError(f"{entry['clip']}: no extraction metadata for prepared query {index}")
            records.append((row, int(box_prediction[index]), int(tracklet_prediction[index])))
    return records


def published_expectations(match_set):
    published = json.loads(PUBLISHED_COMPARISON.read_text())["match_sets"][match_set]["methods"]
    return {variant: published[name] for variant, name in MATCH_SETS[match_set]["resnet_expected"].items()}


def verify_resnet(match_set, paths, grouped, sessions, device):
    expected = published_expectations(match_set)
    records = resnet_records(paths["root"], paths["resnet_checkpoint"], sessions, grouped, device)
    result = {}
    for variant, published in expected.items():
        stats = summarize(records, variant)
        matched = (stats["confusion_matrix"] == published["confusion_matrix"]
                   and np.isclose(stats["score"], published["score"]))
        result[variant] = {"published_method": MATCH_SETS[match_set]["resnet_expected"][variant],
                           "score": stats["score"], "accuracy": stats["accuracy"],
                           "per_group": stats["per_group"], "confusion_matrix": stats["confusion_matrix"],
                           "matches_published": bool(matched)}
        if not matched:
            raise ValueError(f"{match_set} {variant}: ResNet six-reference replay does not match "
                             f"{result[variant]['published_method']}")
    return result


def pooled(match_results, method, variant):
    parts = [match_results[name]["methods"][method][variant] for name in match_results]
    truth_counts = {group: sum(part["truth_counts"][group] for part in parts) for group in GROUP_NAMES}
    correct = {group: sum(part["correct"][group] for part in parts) for group in GROUP_NAMES}
    matrix = np.sum([np.asarray(part["confusion_matrix"], dtype=np.int64) for part in parts], axis=0)
    total = sum(truth_counts.values())
    per_group = {group: correct[group] / truth_counts[group] if truth_counts[group] else 0.0 for group in GROUP_NAMES}
    return {
        "score": float(np.mean(list(per_group.values()))),
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "per_group": per_group,
        "truth_counts": truth_counts,
        "correct": correct,
        "confusion_matrix": matrix.tolist(),
        "unassigned_boxes": sum(part["unassigned_boxes"] for part in parts),
    }


def run(output, cache_dir=CACHE_DIR):
    import sklearn
    import torch
    import transformers
    import umap

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    started = time.monotonic()
    model_info = {method: {"model_id": model_id, "revision": model_revision(model_id)}
                  for method, model_id in MODEL_IDS.items()}
    match_results, self_check, undivided = {}, {}, {}
    for match_set, paths in MATCH_SETS.items():
        rows = load_extract(paths["extract"])
        queries = sum(int(row["query_index"]) >= 0 for row in rows)
        if queries != paths["expected_queries"]:
            raise ValueError(f"{match_set}: query count {queries} != expected {paths['expected_queries']}")
        grouped = rows_by_unit(rows)
        sessions = sessions_for_set(paths["root"])
        self_check[match_set] = verify_resnet(match_set, paths, grouped, sessions, device)
        role_sources = {
            ROBOFLOW_ORACLE_METHOD: ground_truth_roles(grouped),
            ROBOFLOW_PREDICTED_METHOD: predicted_roles(rows, load_role_logits(paths["embed"])),
        }
        methods = {}
        for method, model_id in MODEL_IDS.items():
            revision = model_info[method]["revision"]
            model, processor = image_model(model_id, device)
            features_by_clip = {
                entry["clip"]: cached_features(cache_dir, match_set, model, processor, clip["x"], entry["clip"],
                                               device, model_id, revision)
                for entry, clip, episode, session in sessions
            }
            records = reference_records(sessions, grouped, features_by_clip, model, processor, device, model_id)
            methods[method] = {variant: summarize(records, variant) for variant in ("per_box", "tracklet_mean")}
            if method == "siglip_six_references":
                for roboflow_method, roles_by_clip in role_sources.items():
                    records, skipped = roboflow_records(sessions, grouped, features_by_clip, roles_by_clip)
                    undivided[f"{match_set}/{roboflow_method}"] = skipped
                    methods[roboflow_method] = {variant: summarize(records, variant)
                                                for variant in ("per_box", "tracklet_mean")}
            del model, processor
            if device == "mps":
                torch.mps.empty_cache()
        truth = methods["siglip_six_references"]["per_box"]["truth_counts"]
        match_results[match_set] = {
            "boxes": queries,
            "boxes_per_group": truth,
            "methods": methods,
        }
        print(f"{match_set}: {queries} boxes; " + "; ".join(
            f"{method} {variant}={methods[method][variant]['score']:.4f}"
            for method in methods for variant in ("per_box", "tracklet_mean")), flush=True)
    result = {
        "description": "Frozen SigLIP and CLIP encoders and two Roboflow sports TeamClassifier variants, compared with the reference classifier on the same six reference crops per group, query boxes, tracklets and pitch coordinates. Every method consumes the prepared uint8 96x48 crops, which the Hugging Face image processors resize to 224x224. Score is the mean of the four per-group accuracies with truth counts from all scored boxes; unassigned predictions count as wrong. The Roboflow variant named ..._oracle_roles is an oracle that reads ground-truth roles and tracklets; ..._predicted_roles uses the SoccerNet PRTReID role head for roles instead, so it is the unsupervised baseline.",
        "protocol": {
            "references_per_group": six_references.REFERENCES,
            "reference_selection": "six_references.sessions_for and six_references.make_session with EVALUATION_SEED, the same sessions as experiments/tracklet_comparison.py",
            "pixels": "prepared uint8 96x48 crops for references, query boxes and clustering inputs",
            "prototypes": "cosine prototypes from the mean of the six reference embeddings per group, matching the reference classifier",
            "tracklet_mean": "ground-truth (clip, track_id) grouping over prepared indices, reference anchors excluded, mean embedding per tracklet",
            "roboflow": "SigLIP features of player tracklets per clip -> UMAP(n_components=3, random_state=0) -> KMeans(n_clusters=2, random_state=0); clusters map to teams by mean pitch_x (left = smaller mean x); each goalkeeper takes the team whose player centroid is nearest in pitch_x; a missing pitch_x prediction is unassigned and wrong. The pipeline predicts every box, so its per_box and tracklet_mean entries are identical.",
            "roboflow_roles": {
                ROBOFLOW_ORACLE_METHOD: "ORACLE: player and goalkeeper roles and the (clip, track_id) grouping come from the ground-truth annotations, which favours this variant.",
                ROBOFLOW_PREDICTED_METHOD: "SoccerNet PRTReID role head logits from the official baseline's embed.npz, majority-voted per (clip, track_id) with experiments/prtreid_comparison.py track_role_votes. Tracklets voted player are clustered, tracklets voted goalkeeper use the goalkeeper rule, and boxes voted referee, other or ball stay unassigned and count as wrong.",
            },
            "roboflow_embeddings": f"SigLIP image embeddings cached per clip in {CACHE_DIR} and reused across the reference prototype and both Roboflow variants",
            "roboflow_source": {"repo": "https://github.com/roboflow/sports", "commit": ROBOFLOW_SPORTS_COMMIT,
                                "files": ["sports/common/team.py", "examples/soccer/main.py"]},
            "roboflow_deviations": [
                "TeamClassifier feeds detected crops through the SigLIP processor and averages last_hidden_state tokens; that feature recipe is preserved, but the inputs are the prepared 96x48 crops so every method sees identical pixels.",
                "TeamClassifier fits on detected player crops and predicts per crop; both variants here fit on player tracklet features (mean over the tracklet's boxes) and predict every box, so the unit of scoring matches the other methods. Roles and tracklets come from ground truth in the oracle variant and from the PRTReID role head in the predicted variant; neither variant trains anything.",
                "UMAP and KMeans random_state are fixed to 0; every other hyperparameter stays at the library default of the recorded versions.",
                "Goalkeeper distance is one-dimensional in pitch_x because only pitch_x is available, unlike the example's two-dimensional image coordinates.",
            ],
            "resnet_self_check": "ours ResNet-18 six-reference per-box and tracklet numbers recomputed with this file's own grouping and compared against results/prtreid-comparison.json",
            "device": device,
            "model_info": model_info,
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                         "umap_learn": umap.__version__, "transformers": transformers.__version__},
        },
        "match_sets": match_results,
        "clips_without_player_split": undivided,
        "pooled_six_games": {method: {variant: pooled(match_results, method, variant)
                                      for variant in ("per_box", "tracklet_mean")}
                             for method in match_results[next(iter(match_results))]["methods"]},
        "resnet_self_check": self_check,
        "runtime_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare frozen SigLIP, CLIP and the Roboflow team pipeline with the reference classifier")
    parser.add_argument("--out", type=Path, default=REPO / "results/frozen-encoders.json")
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    args = parser.parse_args()
    result = run(args.out, args.cache)
    print(json.dumps({name: {method: {variant: round(stats["score"], 4) for variant, stats in variants.items()}
                             for method, variants in data["methods"].items()}
                      for name, data in result["match_sets"].items()}, indent=2))


if __name__ == "__main__":
    main()
