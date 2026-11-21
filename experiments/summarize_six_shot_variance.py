import argparse
import importlib.metadata
import json
import platform
import random
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import six_references
import train


def load(path):
    return json.loads(Path(path).read_text())


def embed(model, crops, device):
    features = []
    with torch.inference_mode():
        for start in range(0, len(crops), 128):
            output = model(six_references.batch_crops(crops[start:start + 128], device))
            features.append(F.normalize(output, dim=1).cpu())
    return torch.cat(features)


def statistics_for(truth, prediction):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    counts = matrix.sum(axis=1)
    recalls = np.divide(matrix.diagonal(), counts, out=np.zeros(4, dtype=float), where=counts > 0)
    return {"score": float(recalls.mean()), "accuracy": float(matrix.diagonal().sum() / len(truth)),
            "boxes": len(truth), "confusion_matrix": matrix.tolist()}


def tracklet_statistics(root, run, extract_path):
    checkpoint = Path(run) / "checkpoint.pth"
    rows = json.loads(str(np.load(extract_path, allow_pickle=False)["metadata"]))
    _, splits, _ = train.load_manifest(Path(root).resolve())
    cache = train.ClipCache(12)
    sessions = six_references.sessions_for(splits["test"], cache, Path(root) / "work" / "references")
    rng = random.Random(six_references.EVALUATION_SEED)
    supports = {}
    for entry, clip, bank, episode in sessions:
        supports[entry["clip"]] = (clip, episode, six_references.make_session(clip, bank, episode, rng, True)[0])
    tracks = {}
    for row in rows:
        if row["prepared_index"] >= 0 and row["role"] in ("player", "goalkeeper"):
            key = f"game-{row['game']}-half-{row['session']}"
            tracks[(key, int(row["prepared_index"]))] = (row["clip"], str(row["track_id"]))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = train.build_model(Path(root) / "inputs" / "resnet18.pth", device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.eval()
    box_records, track_records = [], []
    for entry in splits["test"]:
        clip, episode, support = supports[entry["clip"]]
        support_indices, query = episode
        prototypes = F.normalize(embed(model, np.asarray(support), device).reshape(4, six_references.REFERENCES, -1).mean(1), dim=1)
        vectors = embed(model, clip["x"], device)
        box_prediction = (vectors @ prototypes.T).argmax(1).numpy()
        track_prediction = box_prediction.copy()
        excluded = set(map(int, support_indices))
        groups = {}
        for index in range(len(clip["y"])):
            track = tracks.get((entry["clip"], index))
            if track is not None and index not in excluded:
                groups.setdefault(track, []).append(index)
        for members in groups.values():
            mean = F.normalize(vectors[members].mean(0, keepdim=True), dim=1)
            track_prediction[members] = int((mean @ prototypes.T).argmax())
        box_records.append({"truth": clip["y"][query], "pred": box_prediction[query]})
        track_records.append({"truth": clip["y"][query], "pred": track_prediction[query]})
    return {
        "per_box": statistics_for(np.concatenate([row["truth"] for row in box_records]),
                                  np.concatenate([row["pred"] for row in box_records])),
        "tracklet_mean": statistics_for(np.concatenate([row["truth"] for row in track_records]),
                                        np.concatenate([row["pred"] for row in track_records])),
    }


def summarize_run(root, run, extract):
    result = load(Path(run) / "result.json")
    trained = result["test"]
    per_box = {"score": trained["macro_class_accuracy"], "accuracy": trained["accuracy"],
               "boxes": trained["count"], "confusion_matrix": trained["confusion_matrix"]}
    tracklet = tracklet_statistics(root, run, extract)
    if per_box["confusion_matrix"] != tracklet["per_box"]["confusion_matrix"]:
        raise ValueError(f"{run}: per-box test result differs from fixed-reference inference")
    return {"selected_epoch": result["best_epoch"], "per_box": per_box,
            "tracklet_mean": tracklet["tracklet_mean"]}


def pooled(results):
    matrix = [[sum(row["confusion_matrix"][i][j] for row in results) for j in range(4)] for i in range(4)]
    boxes = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(4))
    score = statistics.mean(matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else 0 for i in range(4))
    return {"boxes": boxes, "score": score, "accuracy": correct / boxes,
            "confusion_matrix": matrix}


def metric_summary(seeds, metric):
    return {"mean": statistics.mean(row[metric] for row in seeds),
            "sample_sd": statistics.stdev(row[metric] for row in seeds)}


def main():
    parser = argparse.ArgumentParser(description="Summarize six-reference seed and rotated-fold variance")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed-run", nargs=2, action="append", metavar=("SEED", "OUTPUT"), required=True)
    parser.add_argument("--fold-run", nargs=4, action="append", metavar=("FOLD", "ROOT", "OUTPUT", "EXTRACT"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seed_runs = []
    for seed_text, output in args.seed_run:
        seed = int(seed_text)
        seed_runs.append({"seed": seed, **summarize_run(args.root, output, args.fold_run[0][3])})
    folds = []
    for fold_text, root, output, extract in args.fold_run:
        result = load(Path(output) / "result.json")
        folds.append({"fold": int(fold_text), "test_games": result["split_identity"]["test"],
                      **summarize_run(root, output, extract)})
    games = [game for fold in folds for game in fold["test_games"]]
    if len(games) != len(set(games)):
        raise ValueError("folds must hold out disjoint matches")
    metric_names = ("score", "accuracy")
    seed_summary = {variant: {metric: metric_summary([row[variant] for row in seed_runs], metric)
                              for metric in metric_names}
                    for variant in ("per_box", "tracklet_mean")}
    fold_pooled = {variant: pooled([fold[variant] for fold in folds])
                   for variant in ("per_box", "tracklet_mean")}
    reproduction = seed_runs[next(i for i, row in enumerate(seed_runs) if row["seed"] == 25)]
    difference = abs(reproduction["per_box"]["score"] * 100 - 93.81)
    if difference > 0.01:
        raise ValueError(f"seed 25 per-box score differs from published 93.81% by {difference:.4f} percentage points")
    packages = {}
    for name in ("torch", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "protocol": "six references per class; fixed test queries and references; best validation macro recall checkpoint; tracklet means use ground-truth (clip, track_id) groups excluding reference anchors",
        "evaluation_references": {"seed": six_references.EVALUATION_SEED, "fixed_across_training_seeds": True},
        "units": {"score": "mean of four per-group accuracies", "accuracy": "fraction of correctly classified boxes"},
        "seeds": seed_runs,
        "seed_summary": seed_summary,
        "folds": folds,
        "fold_pooled": {"games": sorted(games), **fold_pooled},
        "reproduction_check": {"seed": 25, "expected_per_box_score_percent": 93.81,
                               "actual_per_box_score_percent": reproduction["per_box"]["score"] * 100,
                               "absolute_difference_percentage_points": difference,
                               "tolerance_percentage_points": 0.01, "passed": True},
        "versions": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages},
        "commands": [
            "python six_references.py --root $ROOT --seed $SEED --out $OUTPUT",
            "python experiments/prtreid_comparison.py extract --root $FOLD_ROOT --out $EXTRACT_OUTPUT",
            "python experiments/summarize_six_shot_variance.py --root $ROOT --seed-run SEED OUTPUT --fold-run FOLD ROOT OUTPUT EXTRACT --out results/six-shot-variance.json",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed_summary": seed_summary, "fold_pooled": fold_pooled,
                      "reproduction_check": result["reproduction_check"]}, indent=2))


if __name__ == "__main__":
    main()
