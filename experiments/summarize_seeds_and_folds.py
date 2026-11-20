import argparse
import json
import statistics
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description="Summarize training seeds and rotated whole-match folds")
    parser.add_argument("--official", type=Path, required=True, help="data root of the official split (seed 25, fold 0)")
    parser.add_argument("--seed-roots", type=Path, nargs="+", required=True, help="data roots named seed-N, trained with train.py --seed N")
    parser.add_argument("--fold-roots", type=Path, nargs="+", required=True, help="rotated fold roots trained with seed 25")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = load(args.official / "work" / "result.json")
    seeds = [{"seed": 25, "test": base["test"]["trained"], "selected_epoch": base["best_epoch"]}]
    for root in args.seed_roots:
        result = load(root / "work" / "result.json")
        seeds.append({"seed": int(root.name.rsplit("-", 1)[-1]),
                      "test": result["test"]["trained"], "selected_epoch": result["best_epoch"]})
    folds = [{"fold": 0, "test_games": base["split_identity"]["test"], "test": base["test"]["trained"], "selected_epoch": base["best_epoch"]}]
    for index, root in enumerate(args.fold_roots, start=1):
        result = load(root / "work" / "result.json")
        folds.append({"fold": index, "test_games": result["split_identity"]["test"], "test": result["test"]["trained"], "selected_epoch": result["best_epoch"]})
    games = [game for fold in folds for game in fold["test_games"]]
    if len(games) != len(set(games)):
        raise ValueError("folds must hold out disjoint matches")
    matrix = [[sum(fold["test"]["confusion_matrix"][i][j] for fold in folds) for j in range(4)] for i in range(4)]
    boxes = sum(sum(row) for row in matrix)
    if boxes != sum(fold["test"]["count"] for fold in folds):
        raise ValueError("pooled confusion matrix does not cover every scored box")
    summary = {metric: {"mean": statistics.mean(seed["test"][metric] for seed in seeds),
                        "sample_sd": statistics.stdev(seed["test"][metric] for seed in seeds)}
               for metric in ("accuracy", "macro_class_accuracy")}
    result = {"protocol": "retrospective role-audited team slots; official held-out split for seeds, same-corpus rotating whole-game folds",
              "seeds": seeds, "seed_summary": summary, "folds": folds,
              "fold_pooled": {"boxes": boxes, "games": sorted(games),
                              "accuracy": sum(matrix[i][i] for i in range(4)) / boxes,
                              "macro_class_accuracy": statistics.mean(matrix[i][i] / sum(matrix[i]) for i in range(4)),
                              "confusion_matrix": matrix}}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed_summary": summary, "fold_pooled": {k: v for k, v in result["fold_pooled"].items() if k != "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
