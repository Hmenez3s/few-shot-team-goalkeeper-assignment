#!/bin/bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "usage: experiments/rotated_fold.sh DATA_ROOT NEW_FOLD_ROOT [SHIFT=1]" >&2
    exit 2
fi
data=$(cd "$1" && pwd -P)
mkdir "$2"
out=$(cd "$2" && pwd -P)
shift_by=${3:-1}
python=${PYTHON:-python}
splits=(train valid test)
mkdir -p "$out/archives" "$out/work/clips" "$out/work/references"
ln -s "$data/inputs" "$out/inputs"
for i in 0 1 2; do
    ln -s "$data/archives/${splits[$i]}.zip" "$out/archives/${splits[$(( (i + shift_by) % 3 ))]}.zip"
done
ln "$data"/work/clips/*.npz "$out/work/clips/"
ln "$data"/work/references/*.npz "$out/work/references/"
cp "$data/work/references/audit.json" "$out/work/references/"
"$python" - "$data/work/manifest.json" "$out/work/manifest.json" "$shift_by" <<'PY'
import json
import sys
from pathlib import Path

source, destination, shift = sys.argv[1], sys.argv[2], int(sys.argv[3])
manifest = json.loads(Path(source).read_text())
order = ["train", "valid", "test"]
rotation = {split: order[(index + shift) % 3] for index, split in enumerate(order)}
for clip in manifest["clips"]:
    clip["split"] = rotation[clip["split"]]
audit = manifest["audit"]
audit["original_to_fold_split"] = rotation
for bucket in ("units", "team_normalization_assignments"):
    for row in audit[bucket]:
        row["split"] = rotation[row["split"]]
for key in ("retained_game_ids", "retained_clip_counts", "output_unit_counts"):
    audit[key] = {rotation[split]: value for split, value in audit[key].items()}
Path(destination).write_text(json.dumps(manifest) + "\n")
PY
echo "fold root: $out"
