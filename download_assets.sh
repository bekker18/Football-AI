#!/usr/bin/env bash
# Downloads the three YOLO checkpoints (player / pitch / ball) into <ROOT>/models
# and one sample broadcast clip into <ROOT>/data/raw -- the repo's committed
# layout (models/ at the project root, footage under data/raw/).
#
# ROOT is the first argument and defaults to this script's own directory (the
# project root), so it works regardless of the current working directory:
#   bash download_assets.sh /kaggle/working/Football-AI
#
# The two destinations can also be overridden independently via MODELS_DIR /
# RAW_DIR -- Docker uses this to keep its flat /data/{models,raw} layout.
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODELS="${MODELS_DIR:-$ROOT/models}"    # checkpoints
RAW="${RAW_DIR:-$ROOT/data/raw}"        # input footage (data)
mkdir -p "$MODELS" "$RAW"

dl () {  # dl <output-path> <gdrive-id>
  if [[ -s "$1" ]]; then
    echo "✓ $1 already present, skipping"
  else
    echo "↓ downloading $1 ..."
    gdown -O "$1" "https://drive.google.com/uc?id=$2"
  fi
}

# Checkpoints (from roboflow/sports examples/soccer/setup.sh) -> models/
dl "$MODELS/football-player-detection.pt" 17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q
dl "$MODELS/football-pitch-detection.pt"  1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf
dl "$MODELS/football-ball-detection.pt"   1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V

# One sample broadcast clip to test end to end -> data/raw/ (input data)
dl "$RAW/2e57b9_0.mp4" 19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf

echo "Checkpoints in $MODELS, footage in $RAW:"
ls -lhR "$MODELS" "$RAW"
