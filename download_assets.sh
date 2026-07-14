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

# The fourth checkpoint, and the only one that is not a .pt: the SigLIP vision
# encoder behind roboflow/sports' TeamClassifier. It is NOT on Drive -- sports
# pulls it from the HF Hub by name, and it does so lazily, when the classifier is
# CONSTRUCTED. Left to itself that means an 813 MB download starting minutes into
# a run, in the middle of phase 1, with the video already half-decoded. Pull it
# here instead, into the cache src/cv/cli.py points HF_HOME at, so the run is
# offline from there on.
#
# HF_HUB_DISABLE_XET: Xet is the Hub's chunked backend for large files; on Kaggle
# it stalls at 0 B/s (the plain CDN, which serves the small config.json, is fine).
# Opting out here is what makes this download finish there.
HF_CACHE="${HF_HOME:-$MODELS/hf_cache}"
SIGLIP="google/siglip-base-patch16-224"
if python -c "import huggingface_hub" 2>/dev/null; then
  echo "↓ caching $SIGLIP into $HF_CACHE ..."
  HF_HOME="$HF_CACHE" HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  SIGLIP="$SIGLIP" python - <<'PY'
import os

from huggingface_hub import snapshot_download

# safetensors only: the repo also ships .bin / flax / tf weights, which
# transformers will not touch and which would triple the download.
path = snapshot_download(
    os.environ["SIGLIP"],
    allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
)
print(f"✓ {path}")
PY
else
  echo "! huggingface_hub not importable -- skipping $SIGLIP."
  echo "  (Layer 2 image: expected. Layer 1: the first run will fetch it itself.)"
fi

echo "Checkpoints in $MODELS, footage in $RAW:"
ls -lhR "$MODELS" "$RAW"
