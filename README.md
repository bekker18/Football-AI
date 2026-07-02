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
mkdir -p data output          # bind-mount targets
docker compose build
docker compose run --rm download         # ~one-time: 3 checkpoints + sample clip → ./data
docker compose run --rm extract          # runs on the sample clip → ./output
```

Run on your own footage (drop the file in `./data` first):

```bash
docker compose run --rm extract \
  --source /data/mygame.mp4 \
  --out-dir /output \
  --save-video /output/annotated.mp4      # optional sanity-check overlay
```

Fast smoke test (cap frames, coarser fit sampling):

```bash
docker compose run --rm extract --source /data/mygame.mp4 --max-frames 300 --stride-fit 30
```

Enable ball detection (off by default — slow on CPU, and Layer 2 / Door 2 is
ball-free-friendly):

```bash
docker compose run --rm extract --source /data/mygame.mp4 --ball
```

## Outputs (in `./output`)

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
3. `docker compose run --rm extract-gpu --source /data/mygame.mp4`.

## Notes & caveats

- **Broadcast vs tactical camera**: these checkpoints are trained on broadcast
  footage. Fixed tactical cameras need different pitch-keypoint handling; expect
  lower homography validity on heavy pan/zoom/replays.
- **Ball is the bottleneck** (as expected): tiled slicing is slow and noisy on
  broadcast video — hence ball-off by default.
- **Reproducibility**: the image pins the Python stack and checks out
  `roboflow/sports` at `--build-arg SPORTS_REF` (default `main`; pass a commit
  SHA to freeze). If upstream changes its API you may need to bump pins.
- Assets cache under `./data` (checkpoints in `./data/models/`, SigLIP weights in
  `./data/hf_cache/`), so re-runs don't re-download. Output files are written as
  root (container default); `sudo chown` them if needed.

## Hand-off to Layer 2

`tracking.parquet` is the game state. The Layer 2 adapter reshapes it into
Metrica-style per-frame tracking (home/away wide format, normalised coords using
the `pitch` dims in `meta.json`) to feed pitch control / EPV / OBSO. Per the
plan, Door 2 (tracking-native) consumes positions directly and needs no ball.
