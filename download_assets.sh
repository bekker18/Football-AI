#!/usr/bin/env bash
# Downloads the three YOLO checkpoints (player / pitch / ball) and one sample
# broadcast clip into /data. Runs inside the container; /data is bind-mounted,
# so assets persist on the host and are fetched only once.
set -euo pipefail

DST="${1:-/data}"
mkdir -p "$DST"

dl () {  # dl <output> <gdrive-id>
  if [[ -s "$DST/$1" ]]; then
    echo "✓ $1 already present, skipping"
  else
    echo "↓ downloading $1 ..."
    gdown -O "$DST/$1" "https://drive.google.com/uc?id=$2"
  fi
}

# Checkpoints (from roboflow/sports examples/soccer/setup.sh)
dl football-player-detection.pt 17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q
dl football-pitch-detection.pt  1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf
dl football-ball-detection.pt   1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V

# One sample broadcast clip to test end to end
dl 2e57b9_0.mp4 19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf

echo "All assets in $DST:"
ls -lh "$DST"
