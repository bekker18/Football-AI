"""Command-line interface for the possession-zone detector.

Mirrors the prerequisites/events argparse style: an input dir in, an output dir
out, thresholds overridable.

    # assign a possessor per frame (the Layer 2 primitive)
    python -m src.possession detect_possession --in data/gamestate --out data/gamestate
    python -m src.possession detect_possession --in data/gamestate --out /tmp/out --r_pz 2.5

    # calibration: re-validate the radius on new (crowded) footage before freezing it
    python -m src.possession sweep_radii --in data/gamestate --out /tmp/out \
        --r-min 1.0 --r-max 5.0 --r-step 0.5

    # eyeball check: render the possessor onto the clip
    python -m src.possession review_possession --in data/gamestate --out data/gamestate \
        --video data/raw/2e57b9_0.mp4

Reads the prepared tracking (``tracking_prepared.parquet`` + ``prep_meta.json``);
writes ``possession_frames.parquet``, ``possession_segments.parquet`` and
``possession_meta.json`` (or ``possession_sweep.csv`` in sweep mode, or
``possession_review.mp4`` in review mode). The prerequisite stage's own outputs
are never touched. Only ``review_possession`` needs ``cv2``.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .config import config_from_prep_meta
from .pipeline import detect_possession
from .sweep import format_sweep, radius_grid, sweep_radii


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--in", dest="in_dir", default="data/gamestate",
                    help="dir with tracking_prepared.parquet + prep_meta.json")
    ap.add_argument("--out", dest="out_dir", default="data/gamestate",
                    help="output dir for the possession_* artifacts")
    ap.add_argument("--r_pz", "--r-pz", dest="r_pz_m", type=float, default=None,
                    help="possession-zone radius in metres (default 3.0; an "
                         "upper bound calibrated on open play -- lower it on "
                         "crowded footage)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m src.possession",
        description="Possession zone: assign a per-frame ball possessor.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect_possession", help="assign a possessor per frame")
    _add_common(detect)

    sweep = sub.add_parser("sweep_radii", help="calibrate: sweep the zone radius")
    _add_common(sweep)
    sweep.add_argument("--r-min", type=float, default=1.0,
                       help="smallest radius in the sweep (default 1.0)")
    sweep.add_argument("--r-max", type=float, default=5.0,
                       help="largest radius in the sweep (default 5.0)")
    sweep.add_argument("--r-step", type=float, default=0.5,
                       help="radius increment (default 0.5)")

    review = sub.add_parser("review_possession",
                            help="render the possession-review video (eyeball check)")
    _add_common(review)
    review.add_argument("--video", default=None,
                        help="video to draw on (default: data/raw/<meta source>)")
    review.add_argument("--video-out", default=None,
                        help="output path (default: <out>/possession_review.mp4)")
    review.add_argument("--start-frame", type=int, default=0,
                        help="first frame to render (default 0)")
    review.add_argument("--end-frame", type=int, default=None,
                        help="last frame to render (default: all)")
    return ap


def _load_prepared(in_dir: str):
    parquet = os.path.join(in_dir, "tracking_prepared.parquet")
    if not os.path.exists(parquet):
        raise SystemExit(
            f"no tracking_prepared.parquet in {in_dir!r}; run "
            f"`python -m src.prerequisites run_prerequisites` first."
        )
    df = pd.read_parquet(parquet)
    meta_path = os.path.join(in_dir, "prep_meta.json")
    prep_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            prep_meta = json.load(f)
    else:
        print(f"[warn] no prep_meta.json in {in_dir!r}; using default fps/pitch")
    return df, prep_meta


def _run_detect(df, cfg, out_dir: str) -> None:
    frames, segments, meta = detect_possession(df, cfg)

    os.makedirs(out_dir, exist_ok=True)
    frames_path = os.path.join(out_dir, "possession_frames.parquet")
    segs_path = os.path.join(out_dir, "possession_segments.parquet")
    meta_path = os.path.join(out_dir, "possession_meta.json")

    frames.to_parquet(frames_path, index=False)
    segments.to_parquet(segs_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    split = "  ".join(
        f"team {t}: {p:.1f}%" for t, p in sorted(meta["team_possession_pct"].items())
    )
    print(
        f"[possession] R_pz={cfg.r_pz_m:g} m  "
        f"coverage {meta['coverage_pct']:.1f}% of {meta['n_ball_frames']} ball-frames "
        f"(ceiling {meta['ball_presence_pct']:.1f}%)"
    )
    print(
        f"[possession] clean {meta['clean_pct']:.1f}%  duel {meta['duel_pct']:.1f}%  "
        f"{meta['n_segments']} segments  median hold "
        f"{meta['median_hold_frames']:.0f} frames"
    )
    if split:
        print(f"[possession] team split: {split}")
    print(f"[possession] wrote:\n  {frames_path}\n  {segs_path}\n  {meta_path}")


def _run_sweep(df, cfg, args) -> None:
    radii = radius_grid(args.r_min, args.r_max, args.r_step)
    sweep = sweep_radii(df, cfg, radii)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "possession_sweep.csv")
    sweep.to_csv(csv_path, index=False)

    print(format_sweep(sweep))
    print(
        "\n[possession] coverage rises with the radius while clean attribution "
        "falls and duels accelerate.\n"
        "[possession] pick the largest radius whose duel rate is still "
        "acceptable on THIS footage."
    )
    print(f"[possession] wrote:\n  {csv_path}")


def _default_video(prep_meta: dict) -> str:
    """Where Layer 1's source clip normally lives (from the recorded meta)."""
    source = (prep_meta.get("source_meta", {}) or {}).get("source") or ""
    return os.path.join("data", "raw", os.path.basename(source))


def _run_review(df, cfg, prep_meta, args) -> None:
    from .review import render_review

    video_in = args.video or _default_video(prep_meta)
    video_out = args.video_out or os.path.join(args.out_dir, "possession_review.mp4")

    frames, _, meta = detect_possession(df, cfg)
    info = render_review(
        df, frames, cfg, video_in, video_out,
        start_frame=args.start_frame, end_frame=args.end_frame,
    )
    print(
        f"[possession] reviewed {info['n_frames_written']} frames at "
        f"R_pz={cfg.r_pz_m:g} m  (clean {meta['clean_pct']:.1f}%  "
        f"duel {meta['duel_pct']:.1f}%)"
    )
    print(
        "[possession] white ring = possessor; orange ring = rival inside the "
        "zone (a duel); minimap circle = R_pz to scale."
    )
    print(f"[possession] wrote:\n  {info['video_out']}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    df, prep_meta = _load_prepared(args.in_dir)
    overrides = {"r_pz_m": args.r_pz_m} if args.r_pz_m is not None else {}
    cfg = config_from_prep_meta(prep_meta, **overrides)

    if args.command == "sweep_radii":
        _run_sweep(df, cfg, args)
    elif args.command == "review_possession":
        _run_review(df, cfg, prep_meta, args)
    else:
        _run_detect(df, cfg, args.out_dir)


if __name__ == "__main__":
    main()
