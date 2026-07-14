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

### …then Layer 2 (once you have game state)

```bash
docker compose build l2               # once, ~1 min

docker compose run --rm prep          # tracking.parquet   -> tracking_prepared.parquet
docker compose run --rm possession    # tracking_prepared  -> possession_frames/segments
docker compose run --rm actions       # possession_*       -> spadl_actions* + ball_aerial
docker compose run --rm l2            # the test suite
```

They must run **in that order** — each reads the previous one's output — and each
is non-destructive: it only ever writes its own files. Each defaults to
`--in data/gamestate --out data/gamestate`; append flags after the service name to
override:

```bash
docker compose run --rm possession --r_pz 2.5
docker compose run --rm actions --out /tmp/out --min-gap-frames 3 --no-aerial
docker compose run --rm actions review_actions --video data/raw/2e57b9_0.mp4
```

## The two images

One `Dockerfile`, two targets — and **`l1` is built `FROM l2`**, because that is
the actual relationship rather than a packaging trick:

| target | size | what it is | needs |
|---|---|---|---|
| **`l2`** | ~1.1 GB | game state → SPADL actions (`prep` · `possession` · `actions`) | pandas / numpy / scipy. **No GPU, no checkpoints, no video.** |
| **`l1`** *(default)* | ~4.0 GB | video → game state (`extract` · `download`) | …all of the above, **plus** torch / YOLO / SigLIP / `roboflow/sports` |

Layer 1 has to look at pixels. Layer 2 never does — it only reads the tables Layer
1 already wrote — so it runs the whole event pipeline in **seconds**, which is what
lets you re-tune a threshold and re-run eventing without going near the slow half.
The `src` package is bind-mounted into the `l2` image rather than copied, so a code
edit is live with **no rebuild**.

Two requirements files, mirroring the two targets, with **no pin written twice**:

- **`requirements.txt`** — everything except pixels (numpy / pandas / pyarrow /
  scipy / opencv-headless, plus the test deps). This *is* Layer 2.
- **`requirements-cv.txt`** — `-r requirements.txt` **+** torch / YOLO / SigLIP.
  This is Layer 1.

Layer 1 writes parquet with the same pandas/pyarrow that Layer 2 reads it back
with — a drifted pair is the kind of bug that surfaces as a corrupt column three
stages downstream rather than as a build failure, so it is made structurally
impossible rather than merely watched for. `pip` resolves both files in one pass in
the `l1` stage, so a conflict fails the *build* rather than an import in production.

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
2. In `requirements-cv.txt` swap the torch index/build to CUDA, e.g.
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
  cv/            Layer 1: video -> raw game state (the only half that sees pixels)
    config.py      class ids, pitch geometry, coordinate-system constants
    geometry.py    homography build + image→pitch-metres conversion
    detection.py   player/jersey crop helpers for the team classifier
    teams.py       goalkeeper assignment + per-track majority vote
    ball.py        optional tiled ball detector (BallDetector)
    annotate.py    optional annotated-video overlay (VideoAnnotator)
    outputs.py     writing parquet/csv/jsonl/meta + version capture
    pipeline.py    run(args): the two-phase extraction orchestration (+ --prepare)
    cli.py         argument parser + entry point (python -m src.cv)
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
  actions/       Layer 2: possession transitions -> SPADL actions (see below)
    source.py      the swappable possession-source interface (NOT the zone detector)
    geometry.py    ball/player tracks, gap path features, goal geometry
                   (emitted start/end come from PLAYERS -- the homography only
                    knows z=0, so a mid-flight ball's coords are fiction)
    aerial.py      airborne-ball flag from the img_y arc (heuristic, NOT height)
    transitions.py the transition rules: touches -> typed events
    spadl.py       the SPADL vocabulary + table contract (mirrors socceraction)
    emit.py        transitions -> SPADL rows + the provenance sidecar
    review.py      actions-review video overlay (the only cv2 dependency)
    pipeline.py    detect_actions(source, tracking, cfg): the one call
    cli.py         argument parser + entry point (python -m src.actions)
main.py                 thin launcher -> src.cv.cli:main (Kaggle-friendly)
tests/                  unit tests for the pure logic, one folder per module
  cv/                     teams, geometry
  prerequisites/          stitch, direction, rescale, ball, deadball, pipeline
  possession/             zone/segments, review overlay, smoke
  actions/                transitions, aerial, SPADL contract, review, smoke
  events/  twopass/  eval/
```

Run it three equivalent ways:

```bash
python main.py   --source ... --out-dir ...   # launcher (no install)
python -m src.cv --source ... --out-dir ...   # module
football-ai      --source ... --out-dir ...   # console script (after pip install .)
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
pytest tests/prerequisites
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
pytest tests/possession
```

# Event layer (Layer 2 — possession transitions → SPADL actions)

`src/actions/` turns the possessor stream into a **stream of on-ball events**,
serialized to **SPADL** so `socceraction` can compute xT / VAEP on it with no
adapter in between.

> Not to be confused with `src/events/`, which is the *ball-free high-value
> window* emitter used to schedule the Layer 1 ball detector (see below). This is
> the event **taxonomy**.

```bash
python -m src.actions detect_actions --in data/gamestate --out data/gamestate
# -> spadl_actions.parquet + spadl_actions_provenance.parquet
#  + ball_aerial.parquet + actions_meta.json
```

Non-destructive: it reads the possession and prerequisite stages' outputs and
writes its own files alongside them. Depends only on pandas / numpy —
`socceraction` is a **test** dependency (it pins our SPADL vocabulary and
validates the output), never a runtime one.

## The core idea: events are transitions, not segments

A pass is not something that happens *to* a player; it is the statement that the
ball **left A and arrived at B**. So the layer walks the *gaps between possession
segments* and names each one. The segments themselves are not events.

Two passes over the stream:

**1. Coalesce segments into touches.** Consecutive segments with the *same*
possessor separated by a short gap are **one touch**. At `R_pz = 3 m` the ball
routinely drifts a hair outside the zone and back — and a player driving with the
ball knocks it past the zone and runs onto it. Merging those first is what makes
a spurious pass *un-emittable* rather than something to recognise and discard
later.

**2. Walk the gaps between touches:**

| gap between touch A and touch B | emitted |
|---|---|
| **same team, different player** | `pass` / **success**, ending at B's reception point. `cross` if it *originates* from a wide, advanced area. |
| **cross-team** | a **turnover**, both sides emitted per SPADL convention: the loser's failed action **and** the winner's defensive one (see below). |
| **same player** | nothing — it was already coalesced into one touch in pass 1. |
| **within a touch** | `dribble` / **success**, *iff* the ball actually moved. |

**Carries are per touch, not per gap**: the carry is the ball's journey from
where a player received it to where they released it. That is what SPADL means
by `dribble`, and it is what keeps the chain **spatially continuous** — a pass
ends where the next player's carry starts, and that carry ends where their pass
starts. xT reads exactly those start→end deltas, so a chain full of teleports
would produce numbers that look fine and mean nothing.

A touch where the ball **sits still is the ball parked near a player, not a
dribble**, and emits nothing. The evidence for a carry is *movement*, never
duration.

**Receptions are not actions.** A successful pass already implies its reception;
SPADL models it as one row with `result=success` and the reception in
`end_x`/`end_y`.

### Turnovers emit both sides

The discriminator is whether the ball was **in flight** across the gap — i.e.
whether it actually travelled (`--flight-min-travel-m`, default 3.0 m). A tackle
takes the ball off a player who *has* it, so the ball barely moves; an
interception cuts out a ball that was already on its way somewhere.

| | losing team | winning team |
|---|---|---|
| ball **in flight** | `pass` / **fail** | `interception` / **success** |
| ball **settled** | `bad_touch` / **fail** | `tackle` / **success** |

## The gap guards (and why they are the whole ballgame)

`loose` is **~37% of the ball-frames** on the test clip. Most of it is real
in-flight passing, but some is just the ball drifting past the zone radius — so a
naive "every possessor change is an event" walk **invents passes**. Three guards
stop that. All are configurable, because they are footage-dependent.

| flag | default | what it stops |
|---|---|---|
| `--bridge-max-gap-frames` | `12` | Same-possessor segments this close become **one touch**. Absorbs the 1-frame zone blip *and* the knock-and-chase. |
| `--bridge-max-ball-dist-m` | `10.0` | …but only if the ball stayed *with* the carrier. A ball that genuinely left and came back is not one touch. |
| `--min-gap-frames` / `--min-ball-travel-m` | `2` / `1.5` | A possessor **change** is credible only if the ball went somewhere, **or** was loose long enough for a real transfer, **or** changed hands with no loose phase at all (a 0-frame gap = contact = a tackle, always credible). Anything else is the zone flickering between two players standing together. |
| `--min-path-coherence` | `0.5` | `straight-line / polyline` distance of the ball through the gap. 1.0 is a laser-straight delivery; low is a ball wobbling or ricocheting. Below this the action is still emitted but flagged **low-confidence** — this is what separates *a pass* from *an aimless deflection*. **Not applied to gaps the ball flew across** — see [Ball height](#ball-height-the-z0-problem-and-the-two-things-we-do-about-it). |
| `--max-gap-frames` | `100` | Beyond 4 s, too much of the transfer went unobserved to name a single event for it. |
| `--min-carry-m` / `--max-carry-m` | `2.0` / `60.0` | Ball movement within a touch needed to call it a dribble — and above which it is a tracking gap, not Maradona. |

Refusals are **counted and reported**, never swallowed: `actions_meta.json` breaks
them down by reason (`spurious`, `static_hold`, `same_player_long_gap`,
`gap_too_long`, `no_geometry`). A gap we declined to name is a gap where something
happened that we could not see, and burying that would make the chain look more
complete than it is.

## Occlusion: emitted, never trusted

`no_ball` frames (~71 on the test clip) are **occlusion, never a stoppage**. A
transition spanning them is still emitted — *a transition unseen is not a
transition that did not happen* — but it is tagged `occluded`, its confidence is
reduced, and its endpoints fall back to the **possessor's own position**. That
fallback's error is *bounded by the possession radius* (a possessor is within
`R_pz` of the ball by definition), rather than being a guess.

**Hook for a ball-free source:** if the ball is missing for a whole gap, the
geometry degrades gracefully to player positions and the layer still emits a
coherent (uniformly low-confidence) chain. A PathCRF-style possession model could
bridge or override those gaps — see `GapPath.from_tracks` in `geometry.py`.

## Ball height: the z=0 problem, and the two things we do about it

**The homography maps the image to the GROUND plane (z=0).** Every pitch
coordinate in this project is therefore the answer to *"where would this pixel be
if the thing in it were lying on the grass"*. For a ball on the grass that is
right. For a ball **in the air** it is wrong in a specific, systematic way: the
back-projected point is **stretched away from the camera** along the viewing ray,
by an amount that grows with the ball's height. There is **no height channel
anywhere in the pipeline** to correct it with.

So an aerial pass used to be recorded as a *flat, distorted ground track* — a ball
that appears to accelerate away, curve, and decelerate back, none of which
happened. And those fake coordinates were setting the `start_x/y` / `end_x/y` of
the very actions xT and VAEP are computed from.

You can see it happen on the sample clip. The ball is hoofed at ~frame 102 and
lands at ~frame 166. Frames **104–122 and 150–165 are all `ball_outlier=True`** in
`tracking_prepared.parquet` — the prerequisites' speed gate threw them out as
*physically impossible*. And it was right to, given what it could see: their
implied ground speed **is** impossible, **because the ball was in the air**. The
outlier flag is a *symptom* of the height problem — which is exactly why it cannot
be used to diagnose it.

### 1. Emitted geometry is anchored on PLAYERS, never on the mid-flight ball

| action | `start_x/y` | `end_x/y` |
|---|---|---|
| `pass` / `cross` | the **passer's** position on the last frame he controlled the ball | the **receiver's** position on the first frame he controlled it (the reception point) |
| `interception` / `tackle` | the **winner's** position where he won it | same |
| `pass`/`fail`, `bad_touch` | the **loser's** position where he lost it | the winner's |
| `dribble` | the **carrier's** own position at the touch's first frame | …and at its last |

A player standing over the ball **is on the ground** — the one plane the homography
is actually valid for. His error is bounded by the possession radius (a possessor
is within `R_pz` of the ball by definition); the airborne ball's error is bounded
by nothing.

**SPADL actions are `start → end`, not trajectories**, so this alone keeps every
distorted mid-flight coordinate out of the event stream and out of xT/VAEP. It also
makes the chain **exactly** continuous rather than approximately: a pass ends at the
receiver's position on frame *f*, and the receiver's carry starts at his position on
frame *f* — the same point, not a point 3 m away. (That is also what socceraction
means by a `dribble`: the connector between the action before and the action after.)

The ball path still decides **what the action was** — did the ball travel at all
(tackle vs. interception), how straight, was it in the air. It just never again says
*where* it happened. The provenance table reports both, separately:
`action_travel_m` (player → player, the geometry we stand behind) and
`ball_travel_m` (the evidence we judged the gap on).

> ⚠️ One escape hatch remains: if a player has **no position at all** on his own
> endpoint frame, the ball's is used instead. That is flagged `endpoint_from_ball`
> and docked confidence — it is the only route by which a ball coordinate can still
> reach an emitted action.

### 2. `airborne` — a per-ball-frame flag (`src/actions/aerial.py`)

We cannot recover the height. What we *can* do is **notice**, so everything
downstream can refuse to trust the ground coordinates.

The camera looks **down** at the pitch, so a ball going **up** moves **up the
image** and `img_y` **decreases**. A ball that rises and falls therefore traces a
**local MINIMUM in `img_y`** — an upward-opening parabola. That vertical arc
survives even though the horizontal geometry is ruined, because it lives in **image
space, upstream of the homography**. Per loose run (ball attributed to nobody):

1. **Clean `img_y` robustly** — running median + MAD, in image space. One bad
   detection (frame 136: `img_y=449` amid `~325`) must not be able to drag a
   least-squares vertex.
2. **Fit a quadratic.** Airborne when the curvature opens upward, the **vertex is
   observed inside the run**, the fit is good (R²), and the arc is deep enough.
3. **Corroborate** with elevated-but-smooth apparent ground speed and with the bbox
   height varying coherently (the ball is further away near the apex, so its box is
   smaller). The speed floor also **gates** — it is the main defence against a
   camera pan being read as an arc.
4. **Partial arcs** (ball already up on entry, or only the descent visible) emit
   `airborne=true` with **LOW, capped confidence** rather than forcing a parabola
   onto half an arc.

> **`ball_outlier` is deliberately NOT used as the spike filter.** It would be
> *circular*: it is a ground-speed gate, so it rejects airborne balls **because**
> they are airborne. Cleaning `img_y` with it would throw away two thirds of the
> sample clip's arc and keep only its middle.

**This is a heuristic, single-camera detector — not height recovery.** It answers
"was the ball probably off the ground here?" with a confidence, and nothing more.
Camera pan/tilt also moves `img_y`, which is why airborne additionally requires the
ball to be **loose**, the run to be **bounded**, and the speed/bbox evidence to
agree. Said out loud in `actions_meta.json` too.

**Output** — `ball_aerial.parquet`: `frame`, `airborne` (bool), `aerial_conf`
(0–1), one row per ball frame. A **sidecar**, joinable on `frame`, so the
prerequisite and possession stages' outputs stay byte-for-byte what they were.

**Consumption** — SPADL has no aerial action type, so an aerial pass is
**subtyped, not retyped**: it stays a `pass` (or a `cross` if it meets the existing
cross geometry) and its flight is recorded as `aerial` / `aerial_conf` in the
**provenance table**. `actions_meta.json` counts the aerial passes/crosses and
records every threshold used.

### …and the coherence guard is relaxed for them

`min_path_coherence` is a **straightness** test, and straightness is a property of
the ball's **ground** path. An airborne ball has no trustworthy ground path — a
cleanly struck 40 m diagonal comes out bent and scores like an aimless deflection.
Judging it by a test written for rolling balls would penalise it for exactly the
distortion we have just identified. So gaps flagged `airborne` are tested against
`--aerial-min-path-coherence` (**default `0.0` = bypassed**) instead.

### The aerial knobs

`--no-aerial` · `--aerial-min-run-frames 8` · `--aerial-max-run-frames 125` ·
`--aerial-min-curvature 0.02` · `--aerial-min-r2 0.80` ·
`--aerial-min-amplitude-px 8.0` · `--aerial-min-speed-ms 12.0` ·
`--aerial-bbox-min-corr 0.30` · `--aerial-min-path-coherence 0.0`

## The possession source is swappable — that is the point

The event layer **never reads the zone detector's internals**. It consumes a
stream of `(frame, time_s, possessor_id, team, state)` and nothing else:

```python
class PathCRFPossessionSource(PossessionSource):
    def stream(self):                      # the whole interface
        for f, pid, team in self.model.decode():
            yield PossessionFrame(f, f / self.fps, pid, team, STATE_POSSESSION)
```

Segments are **derived from the stream** (`segments_from_stream`), not delegated
to `possession_segments.parquet` — so a new source does not have to produce a
segments table and *cannot disagree with one*. There is a single definition of a
segment, and it lives in `source.py`. (The smoke test asserts our derived segments
match the upstream stage's on the real clip, which is what makes swapping the
source a *safe* change rather than a hopeful one.)

`ZonePossessionSource` is the milestone-1 source; `cli.py` names it on exactly one
line.

## The SPADL mapping

Output conforms to `socceraction.spadl.schema.SPADLSchema` (v1.5.3) — verified in
CI by `tests/actions/test_actions_spadl.py`, which asserts our mirrored enums are
**identical** to `socceraction.spadl.config`. (Mirroring the vocabulary is what
keeps the stage free of a `socceraction` runtime dependency; the test is what
stops the mirror rotting silently, since a reordered enum would turn every
`type_id` we emit into a *different, valid-looking* action.)

- Coordinates are the **target 105×68 frame**, which **is** SPADL's default pitch
  — so this layer applies **no rescale at all**. They are clipped into
  `[0,105]×[0,68]`, because the homography legitimately puts players a metre or
  two off the pitch and `SPADLSchema` rejects that.
- Coordinates are **not** normalized left-to-right. Call
  `spadl.play_left_to_right(actions, home_team_id)` with the `home_team_id`
  reported in `actions_meta.json` (the team whose `attack_dir` is `+1`).
- **`SPADLSchema` is `strict`** — an extra column is a hard failure. So the
  confidence / occlusion flags live in a **separate provenance table keyed by
  `action_id`** (`spadl_actions_provenance.parquet`): `confidence`, `occluded`,
  `low_confidence`, `duel_candidate`, ball-path features, and the frames each
  action came from. `left join` on `action_id` recovers everything.

### Known limitation: bodypart is not observable

There is **no pose data**, so the bodypart cannot be inferred. SPADL has no
`unknown` bodypart, and socceraction's own converters default to `foot`; we do the
same (`--bodypart` to override) and say so loudly in `actions_meta.json`. **Do not
read the bodypart columns as measured.**

## Explicitly NOT emitted (milestone 1)

Only `pass`, `cross`, `dribble`, `interception`, `tackle`, `bad_touch` — and the
pipeline **raises** if anything else appears (`EMITTED_ACTIONTYPES` in
`spadl.py`), so a future stage has to widen the scope consciously rather than by
accident.

| not built | extension point |
|---|---|
| **shots** | `transitions.py`, marked `EXTENSION POINT (shots)`. The goal geometry it needs is already computed on every transition (`cfg.goal_xy`, `dist_to_goal_start_m`/`_end_m` in the provenance table). What is missing is a ball-leaves-play signal, not the shape of the code. |
| **set pieces** | `transitions.py`, marked `EXTENSION POINT (set pieces)`. The prerequisites already emit an `in_play` flag; a gap spanning an out-of-play run restarts the game. Milestone 1 treats every gap as open play. |
| **duel resolution** | Turnovers out of a *fleeting contested touch* are emitted as seen and flagged `duel_candidate` (8 of 33 actions on the test clip — the possessor flickering between two players in the zone, rather than the ball truly changing hands). A duel resolver should collapse each pair into one won/lost duel. Counted in `actions_meta.json`, so the noise it would remove is **measurable rather than invisible**. |
| **ballistic height reconstruction** | `aerial.py`, marked `EXTENSION POINT (ballistic reconstruction)`. With the flight frames identified, you could fit `z(t) = z0 + vz·t − g·t²/2` anchored on the two endpoints (which are on the ground, and which we now take from the passer and receiver) and back-project each mid-flight image point onto *that* parabola instead of onto z=0. **Deliberately not built, and not needed:** SPADL actions are start→end, so player-anchoring already keeps the distortion out of xT/VAEP, and there is no consumer in this project for a corrected mid-flight ball position. It would also need camera intrinsics/extrinsics — a homography alone cannot invert a ray to a height. What it *would* unlock: aerial-duel detection, header/volley bodypart inference, shot trajectories over the bar. |
| **feeding `airborne` back upstream** | `cli.py`, marked `EXTENSION POINT`. Two upstream consumers want it. `smooth_ball` currently deletes flight frames as impossible-speed *outliers* (all of frames 104–122 and 150–165 of the sample clip's aerial pass) and S-G-smooths across the hole; knowing they are airborne it could hold them out as **untrusted** rather than **impossible**. `synth_dead_ball` reads "near a boundary and slow" as a stoppage — a ball in flight over the touchline is neither. **Not wired in:** the prerequisites run *before* the possession stream that says which frames are loose, so this would invert the stage order or force a second prerequisites pass over every clip. That is a pipeline change, not a feature. The flag is computed, persisted (`ball_aerial.parquet`) and joinable on `frame` — whoever takes it on starts from data. |

## Review mode (watch it — the counts can't tell you they're the *right* events)

"33 actions, 10 refused" tells you *how many* events were named, never whether
they were the **right** ones. Only watching does.

```bash
python -m src.actions review_actions --in data/gamestate --out data/gamestate \
    --video data/raw/2e57b9_0.mp4
# -> actions_review.mp4

# a slice, while you're iterating:
python -m src.actions review_actions --start-frame 600 --end-frame 750
```

Per frame it draws:

- the **possessor ringed white** and labelled with its `stable_id`;
- the **actor of the active action** ringed in that action's colour and labelled
  with its type;
- a banner with the action (`type` / `result` / actor / confidence / `OCCLUDED` /
  `DUEL?`), its subclassification (progressive-lateral-back, short/long, ball
  distance, path coherence), **and the segment and touch it came from** — so the
  derivation is on screen, not just the conclusion;
- a **minimap** in the target 105×68 frame carrying the **action geometry**: the
  active action as a **start→end arrow**, with the previous few fading behind it,
  so a passing move reads as a chain. Interceptions and tackles are won *on the
  spot*, so they draw as a ring rather than an arrow;
- **three stacked timeline strips — `SEGMENTS` → `TOUCHES` → `ACTIONS`** — which
  are the whole layer in one picture. Where `TOUCHES` has **fewer boundaries** than
  `SEGMENTS`, a blip or a knock-and-chase was coalesced away. Where `ACTIONS` is
  **dark**, the layer **refused to name** that gap.

The arrows are on the minimap and not on the video for a reason: an action is a
statement about two points in *pitch metres*, and drawing it into the image would
need a pitch→image homography, which we do not have post-hoc. The video carries
only what image space can honestly support — who has the ball, and who is acting.

This is the only part of the layer that needs `cv2` (imported lazily, so
`import src.actions` stays dependency-light).

## On the test clip (2e57b9_0)

23 possession segments → 18 touches → **33 SPADL actions** (11 passes, 10
dribbles, 4 interceptions, 4 tackles, 4 bad touches), 10 gaps refused. The chain
is time-ordered and spatially continuous, and loads through `SPADLSchema` with
zero schema errors.

**Aerial:** 4 of the 19 loose runs come out airborne (**1 full arc**, 3 partial),
flagging **113 ball frames**; 6 actions cross an airborne gap. The full arc is the
grounded case the detector was built on and is pinned in the smoke test — the
~1.1 s aerial pass at **frames 123–150**: `img_y` falls 352 → ~325 and climbs back
to ~350 (curvature `+0.135`, **R² = 0.992**, apex observed at frame **136.3**),
apparent ground speed sustained at ~29 m/s, ball loose throughout, and the one bad
detection at frame 136 robustly rejected. The pass across it takes its endpoints
from the **passer at frame 101 and the receiver at frame 167** — and the receiver's
dribble starts at *exactly* the point that pass ended.

**xT cannot be *fitted* on this clip** — and that is expected, not a bug: xT's
value surface is `P(score | cell)`, estimated from the **shots** in the stream,
and milestone 1 emits none by design, so the grid comes out all-zero.
socceraction's `fit()` still ingests the stream and builds a non-degenerate
move-transition matrix, and `rate()` values **exactly** the 17 successful
passes/crosses/dribbles and nothing else — which is the real proof that
socceraction understood our action types as the things we meant them to be.

```bash
pip install -e ".[dev,spadl]"     # socceraction, for the schema + xT tests
pytest tests/actions
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
