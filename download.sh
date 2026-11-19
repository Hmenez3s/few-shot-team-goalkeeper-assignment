#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
revision=ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9
dataset="https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/$revision"
weights=https://download.pytorch.org/models/resnet18-f37072fd.pth
mkdir -p data/archives data/inputs
fetch() {
  local url=$1 final=$2
  if [ ! -e "$final" ]; then
    echo "GET $final"
    curl -fL --connect-timeout 20 --speed-time 120 --speed-limit 1024 --retry 4 --retry-delay 5 -C - -o "$final.part" "$url"
    mv "$final.part" "$final"
  fi
  echo "READY $final"
}
for split in train valid test; do
  fetch "$dataset/$split.zip?download=true" "data/archives/$split.zip"
done
fetch "$weights" data/inputs/resnet18.pth
