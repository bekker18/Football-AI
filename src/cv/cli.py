"""Command-line interface for the football-ai extractor."""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract soccer game state from video.")
    ap.add_argument("--source", required=True, help="input video path")
    ap.add_argument(
        "--out-dir", default="/data/gamestate", help="directory for outputs"
    )
    ap.add_argument(
        "--model-dir", default="/data/models", help="dir holding the .pt checkpoints"
    )
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--imgsz", type=int, default=1280, help="player-model inference size")
    ap.add_argument("--ball-imgsz", type=int, default=640, help="ball-model tile size")
    ap.add_argument(
        "--stride-fit",
        type=int,
        default=60,
        help="frame stride for collecting crops to fit the team classifier",
    )
    ap.add_argument(
        "--max-frames", type=int, default=0, help="0 = whole video (else cap)"
    )
    ap.add_argument(
        "--ball",
        action="store_true",
        help="enable ball detection (slow on CPU; off by default)",
    )
    ap.add_argument(
        "--ball-legacy-tracker",
        action="store_true",
        help="collapse ball candidates with sports' BallTracker instead of running "
        "in-play ball selection. For A/B only — the legacy rule locks onto static "
        "spare balls on the touchline.",
    )
    ap.add_argument(
        "--ball-window",
        type=int,
        default=0,
        help="frames of context used to score ball candidates. 0 = auto (~2 s at "
        "the clip's own fps, which is what the scoring is tuned for; a fixed frame "
        "count would mean 2 s at 25 fps but 0.85 s at 60 fps).",
    )
    ap.add_argument(
        "--ball-window-mode",
        default="centered",
        choices=("centered", "trailing"),
        help="centered = offline (looks both ways); trailing = causal, for the "
        "real-time path",
    )
    ap.add_argument(
        "--ball-min-score",
        type=float,
        default=0.20,
        help="minimum candidate-track score to be the in-play ball; below it NO "
        "ball is emitted for that window (null is a legitimate answer)",
    )
    ap.add_argument(
        "--save-video",
        default=None,
        help="optional path to also write an annotated mp4 (headless)",
    )
    ap.add_argument(
        "--prepare",
        action="store_true",
        help="after extraction, also run the prerequisites (stitch ids, attacking "
        "direction, ball smoothing, dead-ball flag, rescale) and write "
        "tracking_prepared.parquet / frames_prepared.jsonl / prep_meta.json "
        "alongside the raw outputs. Uses default thresholds; run "
        "`python -m src.prerequisites run_prerequisites` to tune them.",
    )
    ap.add_argument(
        "--full-crop",
        action="store_true",
        help="classify teams on the full player box instead of the jersey band",
    )
    ap.add_argument(
        "--no-team-vote",
        action="store_true",
        help="disable the per-track majority vote that stabilises team labels",
    )
    ap.add_argument(
        "--team-stride",
        type=int,
        default=10,
        help="predict team colour every N frames (labels carried forward per "
        "track in between; majority vote fills the rest). 1 = every frame.",
    )
    ap.add_argument(
        "--pitch-stride",
        type=int,
        default=1,
        help="recompute the homography every N frames, reusing it in between "
        "(raise to skip redundant pitch detection; higher = staler on pans).",
    )
    return ap


def _hf_env(model_dir: str) -> None:
    """Point Hugging Face at the cache `download_assets.sh` filled, before import.

    ``TeamClassifier`` fetches SigLIP (~813 MB) from the Hub the moment it is
    constructed — i.e. deep inside phase 1, minutes into a run. Pre-fetched, it is
    already sitting in ``<model-dir>/hf_cache`` and that construction is offline.

    Xet is Hugging Face's chunked transfer backend for large files. It is fine
    normally, but on some sandboxes (Kaggle notably) it stalls at 0 B/s while the
    plain HTTPS CDN — which is what the small ``config.json`` came down over —
    works, so a run gets 813 MB into nothing and hangs. Opting out costs a little
    speed and buys a download that actually finishes.

    Both are ``setdefault``: the Docker images set ``HF_HOME`` themselves, and
    anyone who wants Xet back can export ``HF_HUB_DISABLE_XET=0``.
    """
    os.environ.setdefault("HF_HOME", os.path.join(model_dir, "hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _hf_env(args.model_dir)
    # Import here so `--help` and arg errors don't pay the heavy CV-stack import.
    from .pipeline import run

    run(args)


if __name__ == "__main__":
    main()
