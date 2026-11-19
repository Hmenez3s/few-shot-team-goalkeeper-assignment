#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
python=${PYTHON:-python}
(cd data && shasum -a 256 -c ../checksums.sha256)
"$python" -u prepare.py --root "$PWD/data"
"$python" -u train.py --root "$PWD/data"
