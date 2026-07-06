"""Command-line interface for the prerequisites stage.

Mirrors Layer 1's argparse style. One command runs the whole pipeline; each
transform is also available standalone for composition/debugging:

    python -m src.prerequisites run_prerequisites --in data/gamestate --out data/gamestate
    python -m src.prerequisites stitch_ids        --in data/gamestate --out /tmp/out
    python -m src.prerequisites smooth_ball        --in data/gamestate --out /tmp/out

All thresholds are overridable; unspecified ones fall back to meta.json context
(fps, pitch dims, stride) and the documented defaults in ``config.py``.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict

from .ball import smooth_ball
from .config import TARGET_PITCH_PRESETS, config_from_meta
from .deadball import synth_dead_ball
from .direction import resolve_direction
from .io import load_gamestate, write_prepared
from .pipeline import run_prerequisites
from .rescale import rescale_coords
from .stitch import stitch_ids


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--in", dest="in_dir", default="data/gamestate",
                    help="input dir holding tracking.parquet + meta.json")
    ap.add_argument("--out", dest="out_dir", default="data/gamestate",
                    help="output dir for the *_prepared.* artifacts")
    ap.add_argument("--period-col", default=None,
                    help="optional column naming the period/half (default: single period)")
    # stitch
    ap.add_argument("--stitch-max-gap-frames", type=int, default=None,
                    help="max frame gap to stitch two track fragments (default 25)")
    ap.add_argument("--stitch-max-dist-m", type=float, default=None,
                    help="extrapolated-vs-actual position tolerance in m (default 5.0)")
    ap.add_argument("--stitch-vel-window", type=int, default=None,
                    help="frames used to estimate a track's end velocity (default 5)")
    # direction
    ap.add_argument("--min-gk-frames", type=int, default=None,
                    help="min GK valid frames before a team's GK counts as solid (default 5)")
    # rescale
    ap.add_argument("--target-pitch", choices=sorted(TARGET_PITCH_PRESETS),
                    default=None, help="named target pitch convention (default 105x68)")
    ap.add_argument("--target-length-m", type=float, default=None,
                    help="explicit target pitch length (overrides --target-pitch)")
    ap.add_argument("--target-width-m", type=float, default=None,
                    help="explicit target pitch width (overrides --target-pitch)")
    # ball
    ap.add_argument("--ball-max-speed-ms", type=float, default=None,
                    help="ball speed gate for impossible steps (default 36.0)")
    ap.add_argument("--ball-max-interp-gap", type=int, default=None,
                    help="interpolate ball gaps no longer than this many frames (default 5)")
    ap.add_argument("--ball-savgol-window", type=int, default=None,
                    help="Savitzky-Golay window, odd (default 7)")
    ap.add_argument("--ball-savgol-order", type=int, default=None,
                    help="Savitzky-Golay polynomial order (default 2)")
    # dead ball
    ap.add_argument("--oob-margin-m", type=float, default=None,
                    help="metres beyond a line before the ball reads out-of-bounds (default 2.0)")
    ap.add_argument("--still-speed-ms", type=float, default=None,
                    help="ball speed below which it counts as stationary (default 0.5)")
    ap.add_argument("--still-frames", type=int, default=None,
                    help="stationary frames near a line before dead-ball (default 12)")
    ap.add_argument("--near-boundary-m", type=float, default=None,
                    help="distance to a line counted as 'near' (default 3.0)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m src.prerequisites",
        description="Prerequisites: raw game state -> event-ready game state.",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run_prerequisites", "stitch_ids", "normalize_teams",
                 "rescale_coords", "smooth_ball", "synth_dead_ball"):
        p = sub.add_parser(name, help=name.replace("_", " "))
        _add_common(p)
    return ap


def _cfg_overrides(args) -> dict:
    """Collect the CLI threshold overrides (only the non-None ones)."""
    keys = (
        "period_col", "stitch_max_gap_frames", "stitch_max_dist_m", "stitch_vel_window",
        "min_gk_frames", "ball_max_speed_ms", "ball_max_interp_gap",
        "ball_savgol_window", "ball_savgol_order", "oob_margin_m", "still_speed_ms",
        "still_frames", "near_boundary_m",
    )
    over = {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}
    # target pitch: explicit dims win, else a named preset
    if args.target_length_m is not None:
        over["target_length_m"] = args.target_length_m
    if args.target_width_m is not None:
        over["target_width_m"] = args.target_width_m
    if args.target_pitch and (args.target_length_m is None and args.target_width_m is None):
        L, W = TARGET_PITCH_PRESETS[args.target_pitch]
        over["target_length_m"], over["target_width_m"] = L, W
    return over


# each standalone command maps to a single transform
_SINGLE: Dict[str, Callable] = {
    "stitch_ids": stitch_ids,
    "normalize_teams": resolve_direction,
    "rescale_coords": rescale_coords,
    "smooth_ball": smooth_ball,
    "synth_dead_ball": synth_dead_ball,
}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    df, meta = load_gamestate(args.in_dir)
    cfg = config_from_meta(meta, **_cfg_overrides(args))

    if args.command == "run_prerequisites":
        out_df, prep_meta = run_prerequisites(df, cfg)
    else:
        transform = _SINGLE[args.command]
        out_df, step_meta = transform(df, cfg)
        prep_meta = {"config": cfg.as_meta(), "steps": {args.command: step_meta}}

    prep_meta["source_meta"] = {
        k: meta.get(k) for k in ("source", "fps", "pitch", "perf") if k in meta
    }
    paths = write_prepared(args.out_dir, out_df, prep_meta)

    print(f"[prereq] command={args.command}  rows={len(out_df)}")
    print(f"[prereq] wrote:\n  {paths['parquet']}\n  {paths['jsonl']}\n  {paths['meta']}")


if __name__ == "__main__":
    main()
