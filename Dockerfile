# ---------------------------------------------------------------------------
# One Dockerfile, two images, selected with `--target`.
#
#   target: l2   (~950 MB)  game state -> SPADL actions.   Tables in, tables out.
#   target: l1   (~3.7 GB)  video      -> game state.      [DEFAULT]
#
# **l1 is built FROM l2.** That is not a packaging trick, it is the actual
# relationship: Layer 1 needs everything Layer 2 needs (it writes the very parquet
# Layer 2 reads back) and then torch + YOLO + SigLIP + the roboflow/sports checkout
# on top, because it is the only half that has to look at pixels. Expressing it as
# `FROM l2 AS l1` means the shared pins are installed exactly once, in one place,
# and the two images cannot drift apart on them.
#
# Why keep the small one at all: Layer 2 never decodes a frame, so it needs no GPU,
# no checkpoints and no video — it runs the whole event pipeline in seconds. You
# can re-tune a threshold and re-run eventing without going near the slow half.
#
# `l1` is deliberately the LAST stage, so a bare `docker build .` (no --target)
# still produces the Layer 1 image, exactly as it always did. docker-compose.yml
# names the target explicitly for every service.
# ---------------------------------------------------------------------------


# --- l2: game state -> SPADL actions ---------------------------------------- #
FROM python:3.11-slim AS l2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # writable config dirs (container may run read-only-ish; keep these off /data)
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/mpl \
    # cache SigLIP weights into the mounted /data volume so they download once
    HF_HOME=/data/hf_cache

WORKDIR /app

# No apt packages needed: opencv-python-headless links no libGL, which is exactly
# why the `review_*` overlay videos can be rendered from this image without
# dragging in the X stack.
COPY requirements.txt .
RUN pip install -r requirements.txt

# The `src` package is BIND-MOUNTED at run time (see docker-compose.yml), not
# COPYed, so a code edit is live without a rebuild — the whole ergonomic point of
# the small image. The stages are run as modules:
#
#   python -m src.prerequisites run_prerequisites --in ... --out ...
#   python -m src.possession    detect_possession --in ... --out ...
#   python -m src.actions       detect_actions    --in ... --out ...
#
# Default command is the test suite, so `docker compose run --rm l2` runs it.
CMD ["python", "-m", "pytest", "-q"]


# --- l1: video -> game state (DEFAULT TARGET) ------------------------------- #
FROM l2 AS l1

# Pin the upstream repo so builds are reproducible. Override at build time with
#   docker compose build --build-arg SPORTS_REF=<commit-sha>
ARG SPORTS_REF=main

# ffmpeg + libgl/glib cover OpenCV video IO and any GL-linked transitive deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch / YOLO / SigLIP — the half that looks at pixels. requirements-cv.txt starts
# by including requirements.txt, so pip resolves BOTH layers' pins in one pass and
# a conflict fails the build here rather than at import time in production.
COPY requirements-cv.txt .
RUN pip install -r requirements-cv.txt

# Install the `sports` package WITHOUT its (unpinned) deps so our pins stand.
RUN git clone https://github.com/roboflow/sports.git /opt/sports \
    && cd /opt/sports && git checkout "$SPORTS_REF" \
    && pip install --no-deps -e /opt/sports

# Install our package (registers the `football-ai` console script). --no-deps
# keeps the pinned requirements above untouched (pyproject declares no deps).
COPY pyproject.toml ./
COPY src ./src
COPY main.py download_assets.sh ./
RUN pip install --no-deps .

# Disable ultralytics analytics/telemetry chatter.
RUN yolo settings sync=False 2>/dev/null || true

# `football-ai --source ... --out-dir ...` (the download service overrides this).
ENTRYPOINT ["football-ai"]
