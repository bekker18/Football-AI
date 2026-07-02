#!/usr/bin/env bash
# Downloads the three YOLO checkpoints (player / pitch / ball) into <DST>/models
# and one sample broadcast clip into <DST>. Runs inside the container; <DST> is
# bind-mounted, so assets persist on the host and are fetched only once.
set -euo pipefail

DST="${1:-/data}"
MODELS="$DST/models"
mkdir -p "$MODELS"

dl () {  # dl <output-path-relative-to-DST> <gdrive-id>
  if [[ -s "$DST/$1" ]]; then
    echo "✓ $1 already present, skipping"
  else
    echo "↓ downloading $1 ..."
    gdown -O "$DST/$1" "https://drive.google.com/uc?id=$2"
  fi
}

# Checkpoints (from roboflow/sports examples/soccer/setup.sh) -> models/
dl models/football-player-detection.pt 17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q
dl models/football-pitch-detection.pt  1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf
dl models/football-ball-detection.pt   1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V

# One sample broadcast clip to test end to end -> DST root (it's data, not a model)
dl 2e57b9_0.mp4 19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf

echo "All assets under $DST:"
ls -lhR "$DST"
