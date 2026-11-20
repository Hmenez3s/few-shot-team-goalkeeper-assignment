import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

import train


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Check checkpoint identities and that no identical crop appears in two splits")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads((root / "work" / "result.json").read_text())
    manifest, splits, games = train.load_manifest(root)
    if result["audit"] != manifest["audit"]:
        raise ValueError("result.json was not produced from this manifest")
    hashes, checks = {}, {}
    for split, entries in splits.items():
        crops = []
        for entry in entries:
            with np.load(entry["path"], allow_pickle=False) as clip:
                train.validate_clip(clip, entry["path"])
                crops.extend(hashlib.sha256(crop).digest() for crop in clip["x"])
        hashes[split] = set(crops)
        checks[split] = {"games": sorted(games[split]), "crops": len(crops), "unique_pixel_sha256": len(hashes[split])}
    overlaps = {f"{a}_{b}": len(hashes[a] & hashes[b]) for a, b in itertools.combinations(hashes, 2)}
    verification = {"checkpoint_sha256": sha256(root / "work" / "checkpoint.pth"), "best_epoch": result["best_epoch"],
                    "test_boxes": result["test"]["trained"]["count"], "prepared_split_checks": checks,
                    "cross_split_identical_crops": overlaps,
                    "six_reference_checkpoint_sha256": sha256(root / "work" / "six-references" / "checkpoint.pth")}
    args.out.write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))
    if any(overlaps.values()):
        raise SystemExit("identical crops found in two splits")


if __name__ == "__main__":
    main()
