# ---------------------------------------------------------------------------
# Layer 1 (CV extraction) container. CPU by default so it runs with nothing but
# Docker installed. GPU instructions are in the README.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Pin the upstream repo so builds are reproducible. Override at build time with
#   docker compose build --build-arg SPORTS_REF=<commit-sha>
ARG SPORTS_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # writable config dirs (container may run read-only-ish; keep these off /data)
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/mpl \
    # cache SigLIP weights into the mounted /data volume so they download once
    HF_HOME=/data/hf_cache

# ffmpeg + libgl/glib cover OpenCV video IO and any GL-linked transitive deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned deps first (better layer caching).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install the `sports` package WITHOUT its (unpinned) deps so our pins stand.
RUN git clone https://github.com/roboflow/sports.git /opt/sports \
    && cd /opt/sports && git checkout "$SPORTS_REF" \
    && pip install --no-deps -e /opt/sports

COPY extract_game_state.py download_assets.sh ./

# Disable ultralytics analytics/telemetry chatter.
RUN yolo settings sync=False 2>/dev/null || true

ENTRYPOINT ["python", "extract_game_state.py"]
