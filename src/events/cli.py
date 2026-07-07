"""Command-line interface for the ball-free window emitter.

    python -m src.events --in data/gamestate --out data/gamestate

Reads the prepared tracking (``tracking_prepared.parquet`` + ``prep_meta.json``)
and writes ``high_value_windows.json`` (+ ``value_signals.parquet``). Thresholds
fall back to prep_meta context and the documented defaults in ``config.py``.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .config import config_from_prep_meta
from .pipeline import detect_high_value_windows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m src.events",
        description="Ball-free eventing: emit high-value windows from tracking.",
    )
    ap.add_argument("--in", dest="in_dir", default="data/gamestate",
                    help="dir with tracking_prepared.parquet + prep_meta.json")
    ap.add_argument("--out", dest="out_dir", default="data/gamestate",
                    help="output dir for high_value_windows.json / value_signals.parquet")
    ap.add_argument("--value-threshold", type=float, default=None,
                    help="frame value at/above which it is high-value (default 4.0)")
    ap.add_argument("--window-merge-gap-frames", type=int, default=None,
                    help="bridge high-value runs this many frames apart (default 25)")
    ap.add_argument("--window-min-frames", type=int, default=None,
                    help="discard windows shorter than this (default 12)")
    ap.add_argument("--window-pad-frames", type=int, default=None,
                    help="pad each kept window by this on both sides (default 25)")
    ap.add_argument("--no-signals", action="store_true",
                    help="do not write the per-frame value_signals.parquet")
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


def _overrides(args) -> dict:
    keys = ("value_threshold", "window_merge_gap_frames", "window_min_frames",
            "window_pad_frames")
    return {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    df, prep_meta = _load_prepared(args.in_dir)
    cfg = config_from_prep_meta(prep_meta, **_overrides(args))

    windows, signals, meta = detect_high_value_windows(df, cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    win_path = os.path.join(args.out_dir, "high_value_windows.json")
    with open(win_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(meta=meta, windows=windows.to_dict(orient="records")),
            f, indent=2,
        )

    outputs = [win_path]
    if not args.no_signals:
        sig_path = os.path.join(args.out_dir, "value_signals.parquet")
        signals.to_parquet(sig_path, index=False)
        outputs.append(sig_path)

    print(
        f"[events] {meta['n_windows']} windows over {meta['n_frames']} frames; "
        f"ball-pass coverage {meta['coverage_frac'] * 100:.1f}% of the match"
    )
    print("[events] wrote:\n  " + "\n  ".join(outputs))


if __name__ == "__main__":
    main()
