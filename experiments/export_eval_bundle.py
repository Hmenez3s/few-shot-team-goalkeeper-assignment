import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_formats
import quantize
import train


def calibration_crops(root):
    _, splits, games = train.load_manifest(root)
    rng = np.random.default_rng(quantize.CALIBRATION_SEED)
    crops = []
    for entry in splits["train"]:
        clip = train.ClipCache(1).get(entry["path"])
        picks = [rng.choice(np.flatnonzero(clip["y"] == cls), quantize.CALIBRATION_PER_CLASS, replace=False) for cls in range(4)]
        crops.append(clip["x"][np.stack(picks, axis=1).reshape(-1)])
    return np.concatenate(crops), sorted(games["train"])


def main():
    parser = argparse.ArgumentParser(description="Write the fixed test sessions and training calibration crops to one portable file")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sessions = evaluate_formats.test_sessions(args.root.resolve())
    calibration, games = calibration_crops(args.root.resolve())
    arrays = {"calibration": calibration, "mean": train.MEAN.numpy().reshape(3), "std": train.STD.numpy().reshape(3)}
    meta = []
    for index, session in enumerate(sessions):
        arrays[f"support_{index}"] = session["support"]
        arrays[f"crops_{index}"] = session["crops"]
        arrays[f"truth_{index}"] = session["truth"]
        meta.append({"game": session["game"], "clip": session["clip"], "boxes": int(len(session["truth"]))})
    arrays["sessions"] = np.asarray(json.dumps({"sessions": meta, "calibration_games": games, "references_per_class": 6}))
    np.savez_compressed(args.out, **arrays)
    print(json.dumps({"sessions": len(meta), "boxes": sum(m["boxes"] for m in meta), "calibration_crops": len(calibration)}))


if __name__ == "__main__":
    main()
