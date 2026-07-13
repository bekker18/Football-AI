# Layer 1 — CV extraction (video → game state), fully Dockerised

Turns a soccer video into **structured per-frame game state**: tracked players /
goalkeepers / referees (+ optional ball), each with a team label and a position
in **pitch metres**. Built on `roboflow/sports` (YOLOv8 detection, ByteTrack,
SigLIP+UMAP+KMeans team assignment, homography). You only need Docker.

The upstream repo only *renders annotated video*; this project adds a custom
extractor that writes the actual data (parquet / csv / jsonl) plus an optional
annotated video for eyeballing.

> **Architecture & roadmap:** [`docs/strategy.md`](docs/strategy.md) records the
> committed stance — *ball-free / tracking-native is the default for full-match
> coverage; ball-based eventing is reserved for high-value windows* — the two
> "doors", the two-pass controller, and the build order (benchmark → ball-free
> eventing → two-pass gate).

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
  possession/    Layer 2: per-frame ball possessor (see below)
    zone.py        the four states + vectorized ball-to-player distances
    segments.py    collapse same-possessor runs into touches
    sweep.py       calibration mode: sweep R_pz, report coverage/clean/duel
    review.py      possession-review video overlay (the only cv2 dependency)
    pipeline.py    detect_possession(df, cfg): frames -> segments -> summary
    cli.py         argument parser + entry point (python -m src.possession)
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

# Possession zone (Layer 2 — who is on the ball?)

`src/possession/` is the **first Layer 2 component**: it assigns a **per-frame
ball possessor** from the prepared game state. It is the primitive the event
taxonomy is built on — a pass, a turnover, a duel and a dribble are all
statements about *how the possessor changes between frames*, so the possessor
stream has to exist, and be trustworthy, before any of that can be defined.
**No event logic lives here** — possessor assignment only.

```bash
python -m src.possession detect_possession --in data/gamestate --out data/gamestate
# -> possession_frames.parquet + possession_segments.parquet + possession_meta.json
```

Non-destructive: it only reads the prerequisite stage's outputs and writes its
own `possession_*` files alongside them. Depends only on pandas / numpy.

## The four states

Each frame lands in exactly one state, by counting how many **candidates**
(players + goalkeepers; **referees are excluded**) are within `R_pz` of the ball:

| state | meaning | possessor |
|-------|---------|-----------|
| `no_ball` | no usable smoothed ball position — **occlusion, not a stoppage** | never |
| `loose` | ball present but **nobody** within `R_pz` — pass in flight / loose ball | never |
| `possession` | **exactly one** candidate within `R_pz` | that player |
| `contested` | **two or more** within `R_pz` | the **nearest**, and the frame is flagged |

Two invariants the tests pin: a possessor is **never fabricated** on a `loose` or
`no_ball` frame, and a duel is **never resolved by guessing** beyond "nearest" —
`contested` is a flag for a later duel-resolution step, not a verdict. Possessor
identity is `stable_id` (the stitched track id), never the raw `object_id`.

`no_ball` is deliberately *not* a stoppage: ball absence is occlusion, exactly as
the prerequisites' `in_play` flag treats it. Dead-ball reasoning stays there.

## Coordinate-frame contract

This layer reads the **target** frame (105×68 m) **only**:

- players / goalkeepers → `pitch_x_t_m` / `pitch_y_t_m`
- ball (smoothed **and** rescaled) → `ball_x_ts_m` / `ball_y_ts_m`

The source-frame columns (`pitch_x_m` / `ball_x_s_m`) must **never** be mixed in:
the source→target rescale is *anisotropic* (x×0.875, y×0.971 for 120×70 → 105×68),
so a distance measured across the two frames is silently wrong.

## Outputs (in the `--out` dir)

| file | contents |
|------|----------|
| `possession_frames.parquet` | one row per frame: `frame, time_s, state, possessor_id, possessor_team, dist_m, n_in_zone` |
| `possession_segments.parquet` | possession **segments** (touches): maximal runs of the same possessor — `possessor_id, team, start_frame, end_frame, n_frames, start_time_s, end_time_s, n_contested`. **This is what the next layer reads.** |
| `possession_meta.json` | config + summary: coverage, clean %, duel %, team split, segment stats |

`dist_m` is the distance to the *nearest* candidate and is reported on `loose`
frames too (it says *how* loose); `possessor_id` is populated **only** on
`possession` / `contested` frames. A segment is broken by a `loose` or `no_ball`
frame — bridging a hold across a gap is a *hold heuristic*, which belongs to the
event layer, not to the primitive.

## The radius, and why 3.0 m is an upper bound

**`--r_pz` (default 3.0 m)** is the one real knob. Measured on the sample clip
`2e57b9_0`, the nearest player-to-ball distance is median **1.86 m** / p75 **3.93 m**,
and the radius trades coverage against duels:

| `R_pz` | coverage % | clean % | duel % |
|--------|-----------|---------|--------|
| 2.0 m | 51.5 | 99.7 | 0.3 |
| **3.0 m** | **62.7** | **98.1** | **1.9** |
| 4.0 m | 75.3 | 95.3 | 4.7 |
| 5.0 m | 82.6 | 87.2 | 12.8 |

Duels stay <2% up to 3.0 m and then accelerate, so **3.0 m** is the default.
Coverage is measured against **ball-frames**, not all frames: the ceiling is ball
*presence* (90.5% on this clip), not 100%.

> ⚠️ **Caveat — 3.0 m is an upper bound.** The calibration clip is a single
> open-play attacking phase with **no congested box and no set-pieces**. A corner
> or a goalmouth scramble packs many more players inside any given radius, so the
> duel rate at 3.0 m there will be nothing like ~2%. **Re-run the sweep on new
> footage and lower the radius before freezing it.**

## Sweep mode (re-validate the radius before freezing it)

```bash
python -m src.possession sweep_radii --in data/gamestate --out /tmp/out \
    --r-min 1.0 --r-max 5.0 --r-step 0.5     # -> possession_sweep.csv (+ table on stdout)
```

Reruns the detector at each radius and reports **coverage %, clean %, duel %,
number of segments and median hold length (frames)** per radius. Read it as:
coverage rises with the radius (good) while clean attribution falls and duels
accelerate (bad) — pick the largest radius whose duel rate is still acceptable on
*your* footage.

## Review mode (watch it — the numbers can't tell you it's the *right* player)

Coverage / clean / duel tell you *how much* was attributed, never whether it was
attributed to the **right player**. Only watching it can:

```bash
python -m src.possession review_possession --in data/gamestate --out data/gamestate \
    --video data/raw/2e57b9_0.mp4          # -> possession_review.mp4
# a slice, while you're iterating:
python -m src.possession review_possession --in data/gamestate --out /tmp/out \
    --video data/raw/2e57b9_0.mp4 --start-frame 600 --end-frame 750
```

Per frame it draws:

- the **possessor ringed in white**, labelled with its `stable_id`;
- every *other* candidate inside `R_pz` **ringed in orange** — so a `contested`
  frame visibly *shows* the duel instead of just asserting one;
- the ball, with a line to the possessor labelled with the measured `dist_m`;
- a colour-coded **state banner** (green possession / orange contested / grey
  loose / red no_ball);
- a **minimap** in the target 105×68 frame — the only place `R_pz` can be drawn
  honestly as a **circle**, because metres are linear there. In image space the
  same zone is a perspective-warped ellipse we have no homography to compute
  post-hoc, so we draw the distance line rather than fake a circle;
- a **timeline strip** along the bottom, one column per frame coloured by state,
  with a cursor — the clip's whole possession structure at a glance.

This is the only part of the layer that needs `cv2` (imported lazily, so
`import src.possession` stays dependency-light). Drawing on Layer 1's
`annotated.mp4` also works, but the overlays stack — prefer the raw clip.

```bash
pytest tests/test_possession*.py
```

# Benchmark (Layer 1 quality vs. ground truth)

`src/eval/` measures **how good the extraction actually is**, so eventing and
valuation rest on numbers rather than eyeballed video. It compares predicted
detections against ground truth **in pitch-metre space** and reports detection
(precision/recall/F1 at a distance gate), localization (mean/RMSE metre error),
identity (IDF1 — penalises id switches), and role/team accuracy. Depends only on
pandas / numpy / scipy.

```bash
# our tracking vs. a SoccerNet Game State Reconstruction sequence
python -m src.eval --pred data/gamestate --gt path/to/gsr_sequence --pitch 105x68

# A/B two of our own runs against each other
python -m src.eval --pred run_b/ --gt run_a/ --gt-format tracking
```

The metric engine is format-agnostic (consumes a canonical `frame, track_id, x,
y, role, team` table); `src/eval/adapters.py` converts our output and SoccerNet-
GSR labels onto it. Because our attacking direction is arbitrary, the evaluator
searches pitch orientations (identity + 180°) and reports which it used. **Note:**
the GSR coordinate assumptions (centimetres, pitch-centre origin) are documented
in `adapters.py` and should be confirmed against a real GSR sample.

# Ball-free eventing (high-value windows)

`src/events/` is the first **Door 2 / tracking-native** surface (see
[`docs/strategy.md`](docs/strategy.md)): a cheap pass over the whole match, using
player positions only (no ball), that flags the frame ranges worth spending the
expensive ball detector on. Per frame it measures, in each team's
attacking-normalized frame, how heavily they've committed into the opponent's
final third / box, scores it, and assembles contiguous high-value frames into
padded, gap-merged **windows**. This window stream is exactly what the two-pass
controller gates the ball detector on.

```bash
python -m src.events --in data/gamestate --out data/gamestate
# -> high_value_windows.json (+ value_signals.parquet)
```

Consumes the *prepared* tracking (`tracking_prepared.parquet` — needs
`attack_dir` + target-frame coords, so run `src.prerequisites` first). The
emitted `meta.coverage_frac` is the key number: the share of the match a ball
pass would touch instead of 100%. Depends only on pandas / numpy.

# Two-pass controller (gate the ball detector onto flagged windows)

`src/twopass/` is the mechanism that makes "ball only where it matters"
automatic. It reads the high-value windows, **gates** them under a frame budget
(highest-value first; an oversized top window is truncated around its core, never
dropped), and — in the full pass — re-decodes only those frames to run the ball
detector there, emitting a *sparse* ball table.

```bash
# gate only — no video / CV stack needed; shows how little of the match the ball runs on
python -m src.twopass --in data/gamestate --plan-only --budget-frac 0.10

# full pass 2 (needs the CV stack + checkpoints): detect the ball on the planned frames
python -m src.twopass --in data/gamestate --source /data/raw/game.mp4 \
    --model-dir data/models --budget-frac 0.10 --out data/gamestate
```

The gate/plan half (`plan.py`) is pure pandas/numpy and unit-tested; only the
Pass 2 executor (`controller.py`) imports the Layer 1 CV stack. Output
`ball_windows.parquet` feeds back into the prerequisites' ball smoothing for the
covered windows. See [`docs/strategy.md`](docs/strategy.md) for the full rationale.
