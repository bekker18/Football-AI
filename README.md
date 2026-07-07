# Layer 1 — CV extraction (video → game state), fully Dockerised

Turns a soccer video into **structured per-frame game state**: tracked players /
goalkeepers / referees (+ optional ball), each with a team label and a position
in **pitch metres**. Built on `roboflow/sports` (YOLOv8 detection, ByteTrack,
SigLIP+UMAP+KMeans team assignment, homography). You only need Docker.

The upstream repo only *renders annotated video*; this project adds a custom
extractor that writes the actual data (parquet / csv / jsonl) plus an optional
annotated video for eyeballing.

## Quick start

```bash
mkdir -p data/raw data/gamestate    # bind-mount targets (all under ./data)
docker compose build
docker compose run --rm download         # ~one-time: checkpoints → ./data/models, sample clip → ./data/raw
docker compose run --rm extract          # runs on the sample clip → ./data/gamestate
```

Run on your own footage (drop the file in `./data/raw` first):

```bash
docker compose run --rm extract \
  --source /data/raw/mygame.mp4 \
  --out-dir /data/gamestate \
  --save-video /data/gamestate/annotated.mp4   # optional sanity-check overlay
```

Fast smoke test (cap frames, coarser fit sampling):

```bash
docker compose run --rm extract --source /data/raw/mygame.mp4 --max-frames 300 --stride-fit 30
```

Enable ball detection (off by default — slow on CPU, and Layer 2 / Door 2 is
ball-free-friendly):

```bash
docker compose run --rm extract --source /data/raw/mygame.mp4 --ball
```

## Outputs (in `./data/gamestate`)

| file | shape | contents |
|------|-------|----------|
| `tracking.parquet` / `tracking.csv` | one row per (frame, object) | the main artifact |
| `frames.jsonl` | one line per frame | same data, objects nested per frame |
| `meta.json` | — | fps, resolution, pitch dims, pinned versions, run args |
| `annotated.mp4` | — | only with `--save-video` |

Per-row columns: `frame, time_s, object_id, role, team, img_x, img_y,
pitch_x_m, pitch_y_m, pitch_valid, bbox_x1..bbox_y2`.

- **Coordinates**: `pitch_x_m ∈ [0,120]` (length), `pitch_y_m ∈ [0,70]` (width),
  origin top-left, metres. `pitch_valid=false` when a frame had too few pitch
  keypoints for a homography (zoom-ins, replays) — those rows still carry image
  pixels.
- **`object_id`**: integer id, stable within a clip. `>= 1` is a ByteTrack id
  for a person; `0` is the ball (one reserved track); `-1` is a detection not yet
  confirmed by the tracker. No nulls — select the ball with `object_id == 0`.
- **`team`**: `0`/`1` are **arbitrary KMeans clusters, not stable across clips**
  and not tied to home/away. Map them to real teams downstream.

## Key knobs

`--device cpu|cuda|mps` · `--imgsz 1280` (player model) · `--ball-imgsz 640` ·
`--stride-fit 60` (crop sampling for the team classifier) · `--max-frames 0`
(0 = whole video) · `--ball` · `--save-video PATH`.

### Speed knobs

Team colour (SigLIP + UMAP) is the per-frame bottleneck, so it's predicted on a
stride and majority-voted, not recomputed every frame:

- `--team-stride 10` — frames between team-colour predictions (labels carried
  forward per track in between). Lower if short tracks come out with null `team`.
- `--pitch-stride 1` — frames between homography recomputes (reused in between).
  Raise to ~3–5 to skip redundant pitch detection; higher gets staler on pans.
- FP16 (`half`) inference is auto-enabled on `--device cuda`.

On CPU, `--imgsz 960` and dropping `--ball` (the tiled slicer is many inferences
per frame) are the biggest additional wins.

## GPU (optional)

The default image ships **CPU** torch so it runs anywhere. CPU is fine for
testing but slow (1280px YOLO + SigLIP). For real runs use a GPU:

1. Install the **NVIDIA Container Toolkit** on the host (this is more than "just
   Docker").
2. In `requirements.txt` swap the torch index/build to CUDA, e.g.
   `--extra-index-url https://download.pytorch.org/whl/cu121` with matching
   `torch`/`torchvision` cu121 wheels, then `docker compose build`.
3. `docker compose run --rm extract-gpu --source /data/raw/mygame.mp4`.

## Notes & caveats

- **Broadcast vs tactical camera**: these checkpoints are trained on broadcast
  footage. Fixed tactical cameras need different pitch-keypoint handling; expect
  lower homography validity on heavy pan/zoom/replays.
- **Ball is the bottleneck** (as expected): tiled slicing is slow and noisy on
  broadcast video — hence ball-off by default.
- **Reproducibility**: the image pins the Python stack and checks out
  `roboflow/sports` at `--build-arg SPORTS_REF` (default `main`; pass a commit
  SHA to freeze). If upstream changes its API you may need to bump pins.
- Everything lives under `./data`: input footage in `./data/raw/`, checkpoints in
  `./data/models/`, SigLIP weights in `./data/hf_cache/`, and results in
  `./data/gamestate/`. Assets download once. Output files are written as root
  (container default); `sudo chown` them if needed.

## Project layout

The logic lives in the `src` package (installable via `pyproject.toml`);
`main.py` is a thin launcher.

```
src/
  config.py      class ids, pitch geometry, coordinate-system constants
  geometry.py    homography build + image→pitch-metres conversion
  detection.py   player/jersey crop helpers for the team classifier
  teams.py       goalkeeper assignment + per-track majority vote
  ball.py        optional tiled ball detector (BallDetector)
  annotate.py    optional annotated-video overlay (VideoAnnotator)
  outputs.py     writing parquet/csv/jsonl/meta + version capture
  pipeline.py    run(args): the two-phase extraction orchestration (+ --prepare)
  cli.py         argument parser + entry point
  prerequisites/ raw game state -> event-ready game state (see below)
    stitch.py      track id stabilization (motion stitching)
    direction.py   team attacking-direction resolution
    rescale.py     source -> target pitch rescale
    ball.py        ball outlier rejection + Savitzky-Golay smoothing
    deadball.py    in-play / dead-ball heuristic proxy
    pipeline.py    run_prerequisites(df, cfg): compose the five transforms
    cli.py         argument parser + entry point (python -m src.prerequisites)
main.py                 thin launcher -> src.cli:main (Kaggle-friendly)
tests/                  unit tests for the pure logic (teams, geometry, prereqs)
```

Run it three equivalent ways:

```bash
python main.py    --source ... --out-dir ...   # launcher (no install)
python -m src.cli --source ... --out-dir ...   # module
football-ai       --source ... --out-dir ...   # console script (after pip install .)
```

Dev setup and tests (the pure-logic tests need only numpy + pytest, not the CV stack):

```bash
pip install -e ".[dev]"
pytest
```

## Hand-off downstream

`tracking.parquet` is the raw game state. A downstream adapter reshapes it into
Metrica-style per-frame tracking (home/away wide format, normalised coords using
the `pitch` dims in `meta.json`) to feed pitch control / EPV / OBSO. Door 2
(tracking-native) consumes positions directly and needs no ball. That adapter
consumes the **prepared** game state described next.

# Prerequisites (raw game state → event-ready game state)

Before any event detection or valuation, the raw Layer 1 state needs cleaning.
`src/prerequisites/` provides five **composable, non-destructive** transforms —
the stage between extraction and event/valuation work. Each one only **adds
columns** (or emits metadata) — originals are never overwritten — and each is
independently importable, unit-tested, and CLI-runnable. It depends only on
pandas / numpy / scipy (not the CV stack).

```bash
# whole pipeline (reads meta.json for fps / pitch dims / stride — nothing hardcoded)
python -m src.prerequisites run_prerequisites --in data/gamestate --out data/gamestate

# any single transform standalone (same flags)
python -m src.prerequisites smooth_ball --in data/gamestate --out /tmp/out
```

Or fold it into extraction so Layer 1 emits event-ready data in one command
(raw outputs are still written too — the prepared files sit alongside them):

```bash
docker compose run --rm extract --source /data/raw/mygame.mp4 --prepare
```

`--prepare` uses default thresholds; run the standalone command above to tune
them without re-running the (slow, GPU) video extraction.

Pipeline order: `stitch_ids → resolve_direction → smooth_ball → synth_dead_ball
→ rescale_coords` (rescale is independent; it runs last so synthetic ball rows
are rescaled too).

## Outputs (in the `--out` dir)

| file | contents |
|------|----------|
| `tracking_prepared.parquet` | every original row + column, plus all added columns |
| `frames_prepared.jsonl` | per-frame nested view (adds `in_play` / `in_play_conf`) |
| `prep_meta.json` | all params used + resolved per-team directions + target pitch + stitching summary |

### Added columns

`stable_id`, `attack_dir`, `pitch_x_t_m`, `pitch_y_t_m`, `ball_outlier`,
`ball_interp`, `synthetic`, `ball_x_s_m`, `ball_y_s_m`, `ball_x_ts_m`,
`ball_y_ts_m`, `ball_vx_ms`, `ball_vy_ms`, `ball_speed_ms`, `ball_accel_ms2`,
`in_play`, `in_play_conf`.

Coordinate frames: `pitch_x_m`/`pitch_y_m` and `ball_x_s_m`/`ball_y_s_m` are in
the **source** pitch frame (120×70); `pitch_x_t_m`/`pitch_y_t_m` and the smoothed
ball `ball_x_ts_m`/`ball_y_ts_m` are in the **target** frame (105×68). `prep_meta`
records this under `rescale_coords.coordinate_frames`.

## The five transforms (assumptions & key parameters)

1. **Track id stabilization** (`stitch_ids`) — ByteTrack fragments a player into
   several ids. With no appearance features, fragments are stitched by **motion
   only**: link the end of track A to the start of track B when the gap ≤
   `--stitch-max-gap-frames` (25), roles and (already voted) teams match, and A's
   velocity-extrapolated position is within `--stitch-max-dist-m` (5.0) of B's
   start. Adds `stable_id`; ball/referee/unlinked keep their id. *Cross-clip /
   multi-half global re-ID is a documented TODO, not implemented.*

2. **Team normalization + attacking direction** (`normalize_teams`) — resolves
   each team's `attack_dir` (+1 → attacks toward `x_max`, −1 → toward `x=0`) per
   period from **GK median pitch_x**. Fallbacks for sparse GK (`--min-gk-frames`,
   5): mirror the other team; then deepest-defender centroid; else leave null and
   warn (never guessed). `normalize_to_attack(x, attack_dir, length)` returns
   attacking-normalized coordinates for one team without rotating the shared frame.

3. **Coordinate rescale** (`rescale_coords`) — linear map from the source pitch
   in `meta.json` (120×70) to a target convention (`--target-pitch 105x68`
   default, `120x80` supported, or explicit `--target-length-m/--target-width-m`),
   origin preserved. Adds `pitch_x_t_m` / `pitch_y_t_m`, and — when the smoothed
   ball is present — `ball_x_ts_m` / `ball_y_ts_m` (smoothed ball in the target
   frame, so downstream code never mixes frames); recorded in `prep_meta`.

4. **Ball smoothing + outlier rejection** (`smooth_ball`) — **robustly** rejects
   every physically impossible point via an *iterative* speed gate
   (`--ball-max-speed-ms`, 36) as `ball_outlier` (never deleted): a point whose
   implied speed to its accepted neighbours exceeds the cap is removed and the
   gate re-run until no consecutive-frame step in the retained track exceeds it —
   catching two-frame excursions and step-changes, not only isolated spikes.
   Short gaps (`--ball-max-interp-gap`, 5 frames) are then interpolated with
   synthetic `ball_interp` rows, and *only then* Savitzky-Golay smooths each
   segment (`--ball-savgol-window` 7, `--ball-savgol-order` 2) into
   `ball_x_s_m`/`ball_y_s_m` (rejecting outliers first because S-G is a
   least-squares fit and is not robust to them), and a final rate limiter clamps
   any residual S-G overshoot of a near-cap move back to the cap; velocity/speed/
   accel are recomputed **from the smoothed track**, which is asserted to contain
   zero consecutive-frame steps above the cap. `pitch_stride` (from `meta.json`)
   is honoured so homography-stride steps don't register as spikes.

5. **Dead-ball / in-play flag** (`synth_dead_ball`) — a **tunable heuristic
   proxy, not a ground-truth stoppage signal**. Per-frame `in_play` + confidence
   from: ball out of bounds beyond `--oob-margin-m` (2.0) of the pitch extents, or
   sustained near-zero speed (`--still-speed-ms` 0.5) near a boundary
   (`--near-boundary-m` 3.0) for `--still-frames` (12). **Ball absence is treated
   as occlusion, never as a dead ball** — the last decision is carried forward
   with decaying confidence.

Run the tests (pandas + scipy + pytest; the smoke test needs the sample
`data/gamestate/tracking.parquet` and is skipped otherwise):

```bash
pytest tests/test_prereq_*.py
```
