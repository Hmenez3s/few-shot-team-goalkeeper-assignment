import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
import six_references
import train


def embed(model, crops, device):
    out = []
    with torch.inference_mode():
        for start in range(0, len(crops), 128):
            out.append(F.normalize(model(six_references.batch_crops(crops[start:start + 128], device)), dim=1).cpu())
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser(description="Score the reference classifier per box and with ground-truth tracklet averaging")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--extract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--one", type=Path, default=REPO / "models/one-reference.pth")
    parser.add_argument("--six", type=Path, default=REPO / "models/six-references.pth")
    parser.add_argument("--expected-one", type=Path, default=REPO / "results/result.json")
    parser.add_argument("--expected-six", type=Path, default=REPO / "results/six-references.json")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    device = torch.device(args.device)
    rows = json.loads(str(np.load(args.extract, allow_pickle=False)["metadata"]))
    manifest, splits, _ = train.load_manifest(ROOT)
    cache = train.ClipCache(12)
    sessions = six_references.sessions_for(splits["test"], cache, ROOT / "work" / "references")
    rng = random.Random(six_references.EVALUATION_SEED)
    supports = {}
    for entry, clip, bank, episode in sessions:
        supports[entry["clip"]] = (clip, episode, six_references.make_session(clip, bank, episode, rng, True)[0])
    tracks = {}
    for row in rows:
        if row["prepared_index"] >= 0 and row["role"] in ("player", "goalkeeper"):
            key = f"game-{row['game']}-half-{row['session']}"
            tracks[(key, row["prepared_index"])] = (row["clip"], str(row["track_id"]))
    results = {}
    for name, path, per_class in (("one_reference", args.one, 1), ("six_references", args.six, 6)):
        model = train.build_model(ROOT / "inputs/resnet18.pth", device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
        model.eval()
        per_box, per_track = [], []
        for entry in splits["test"]:
            clip, episode, six_support = supports[entry["clip"]]
            support_index, query = episode
            refs = clip["x"][support_index] if per_class == 1 else np.stack(six_support)
            prototypes = F.normalize(embed(model, refs, device).reshape(4, per_class, -1).mean(1), dim=1)
            vectors = embed(model, clip["x"], device)
            box_pred = (vectors @ prototypes.T).argmax(1).numpy()
            excluded = set(map(int, support_index))
            groups = defaultdict(list)
            for index in range(len(clip["y"])):
                key = tracks.get((entry["clip"], index))
                if key is not None and index not in excluded:
                    groups[key].append(index)
            track_pred = box_pred.copy()
            for members in groups.values():
                mean = F.normalize(vectors[members].mean(0, keepdim=True), dim=1)
                track_pred[members] = int((mean @ prototypes.T).argmax())
            truth = clip["y"][query]
            per_box.append({"game": entry["game"], "clip": entry["clip"], "truth": truth, "pred": box_pred[query]})
            per_track.append({"game": entry["game"], "clip": entry["clip"], "truth": truth, "pred": track_pred[query]})
        box_stats, track_stats = train.statistics(per_box), train.statistics(per_track)
        published = json.loads((args.expected_one if per_class == 1 else args.expected_six).read_text())
        expected = published["test"]["trained"] if per_class == 1 else published["test"]
        assert box_stats["confusion_matrix"] == expected["confusion_matrix"], (name, box_stats["confusion_matrix"], expected["confusion_matrix"])
        results[name] = {"per_box": box_stats, "tracklet_mean": track_stats}
        print(name, "per box", round(box_stats["accuracy"] * 100, 2), round(box_stats["macro_class_accuracy"] * 100, 2),
              "tracklet", round(track_stats["accuracy"] * 100, 2), round(track_stats["macro_class_accuracy"] * 100, 2), track_stats["confusion_matrix"], flush=True)
    args.out.write_text(json.dumps(results, indent=2, default=lambda x: x.tolist()))


if __name__ == "__main__":
    main()
