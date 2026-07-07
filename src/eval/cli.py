"""Command-line interface for the Layer 1 benchmark.

    python -m src.eval --pred data/gamestate --gt path/to/gsr_sequence \
        --pitch 105x68 --match-dist-m 2.0

``--pred`` is our tracking output (dir or parquet/csv). ``--gt`` is a SoccerNet-
GSR sequence (dir or Labels-GameState.json) unless ``--gt-format tracking`` is
given (then it is another of our tracking tables, e.g. for A/B runs). Writes a
JSON report and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import os

from .adapters import load_soccernet_gsr, load_tracking
from .config import EvalConfig, VALID_TRANSFORMS
from .pipeline import evaluate, summarize

_PITCH_PRESETS = {"105x68": (105.0, 68.0), "120x70": (120.0, 70.0), "120x80": (120.0, 80.0)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m src.eval",
        description="Benchmark Layer 1 extraction against ground truth.",
    )
    ap.add_argument("--pred", required=True,
                    help="our tracking output: dir or parquet/csv")
    ap.add_argument("--gt", required=True,
                    help="ground truth: SoccerNet-GSR sequence dir/json, "
                         "or a tracking table with --gt-format tracking")
    ap.add_argument("--gt-format", choices=("soccernet", "tracking"),
                    default="soccernet")
    ap.add_argument("--pitch", choices=sorted(_PITCH_PRESETS), default="105x68",
                    help="shared pitch frame for both inputs (default 105x68)")
    ap.add_argument("--match-dist-m", type=float, default=2.0,
                    help="pitch-distance gate for a true positive (default 2.0)")
    ap.add_argument("--flip", nargs="+", choices=VALID_TRANSFORMS,
                    default=["none", "rot180"],
                    help="orientation transforms to search (default none rot180)")
    ap.add_argument("--roles", nargs="+", default=["player", "goalkeeper"],
                    help="roles to evaluate (default player goalkeeper); "
                         "pass 'all' to keep every role")
    ap.add_argument("--gsr-units", choices=("cm", "m", "mm"), default="cm",
                    help="pitch-coordinate units in the GSR labels (default cm)")
    ap.add_argument("--gsr-corner-origin", action="store_true",
                    help="GSR coords already corner-origin (skip centre shift)")
    ap.add_argument("--out", default=None,
                    help="write the JSON report here (default: <pred>/eval_report.json "
                         "when --pred is a dir, else ./eval_report.json)")
    return ap


def _resolve_out(args) -> str:
    if args.out:
        return args.out
    if os.path.isdir(args.pred):
        return os.path.join(args.pred, "eval_report.json")
    return "eval_report.json"


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    L, W = _PITCH_PRESETS[args.pitch]
    roles = None if args.roles == ["all"] else tuple(args.roles)
    cfg = EvalConfig(
        pitch_length_m=L, pitch_width_m=W,
        match_dist_m=args.match_dist_m, flip_candidates=tuple(args.flip), roles=roles,
    )

    pred = load_tracking(args.pred, cfg)
    if args.gt_format == "tracking":
        gt = load_tracking(args.gt, cfg)
    else:
        gt = load_soccernet_gsr(
            args.gt, cfg, pitch_units=args.gsr_units,
            center_origin=not args.gsr_corner_origin,
        )

    report = evaluate(gt, pred, cfg)
    report["config"] = cfg.as_meta()
    report["inputs"] = dict(pred=args.pred, gt=args.gt, gt_format=args.gt_format)

    out = _resolve_out(args)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(summarize(report))
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()
