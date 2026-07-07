"""Command-line interface for the two-pass controller.

    # gate only (no CV stack needed): see how little of the match the ball runs on
    python -m src.twopass --in data/gamestate --plan-only

    # full pass 2: re-decode the planned frames and detect the ball there
    python -m src.twopass --in data/gamestate --source /data/raw/game.mp4 \
        --model-dir data/models --budget-frac 0.10 --out data/gamestate

Reads ``high_value_windows.json`` (from ``src.events``), plans the ball-frame
budget, and — unless ``--plan-only`` — runs the ball detector on those frames,
writing a sparse ``ball_windows.parquet`` plus the ``twopass_plan.json`` manifest.
"""

from __future__ import annotations

import argparse
import json
import os

from .config import TwoPassConfig
from .plan import plan_ball_frames


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m src.twopass",
        description="Gate the ball detector onto high-value windows.",
    )
    ap.add_argument("--in", dest="in_dir", default="data/gamestate",
                    help="dir with high_value_windows.json (from src.events)")
    ap.add_argument("--out", dest="out_dir", default="data/gamestate",
                    help="output dir for ball_windows.parquet / twopass_plan.json")
    ap.add_argument("--source", default=None,
                    help="video to re-decode for pass 2 (omit with --plan-only)")
    ap.add_argument("--model-dir", default="data/models",
                    help="dir with football-pitch/ball-detection.pt")
    ap.add_argument("--budget-frac", type=float, default=0.10,
                    help="ball runs on at most this share of frames (default 0.10; "
                         "use -1 for no cap)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="optional hard cap on selected windows")
    ap.add_argument("--min-peak-value", type=float, default=0.0,
                    help="ignore windows below this peak value")
    ap.add_argument("--ball-imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    ap.add_argument("--plan-only", action="store_true",
                    help="write the plan and exit; no video / CV stack needed")
    return ap


def _load_windows(in_dir: str):
    path = os.path.join(in_dir, "high_value_windows.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"no high_value_windows.json in {in_dir!r}; run `python -m src.events` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    import pandas as pd
    windows = pd.DataFrame(doc.get("windows", []))
    total_frames = int(doc.get("meta", {}).get("n_frames", 0))
    return windows, total_frames


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    windows, total_frames = _load_windows(args.in_dir)

    cfg = TwoPassConfig(
        budget_frac=(None if args.budget_frac is not None and args.budget_frac < 0
                     else args.budget_frac),
        max_windows=args.max_windows,
        min_peak_value=args.min_peak_value,
        ball_imgsz=args.ball_imgsz,
        device=args.device,
    )

    frames, plan = plan_ball_frames(windows, total_frames, cfg)
    plan["config"] = cfg.as_meta()

    os.makedirs(args.out_dir, exist_ok=True)
    plan_path = os.path.join(args.out_dir, "twopass_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(
        f"[twopass] {plan['n_windows_selected']}/{plan['n_windows_total']} windows "
        f"selected -> ball on {plan['n_frames']} frames "
        f"({plan['coverage_frac'] * 100:.1f}% of the match)"
    )
    print(f"[twopass] wrote {plan_path}")

    if args.plan_only:
        return
    if not args.source:
        raise SystemExit("pass 2 needs --source VIDEO (or use --plan-only)")

    from .controller import run_ball_on_frames

    pitch_pt = os.path.join(args.model_dir, "football-pitch-detection.pt")
    ball_pt = os.path.join(args.model_dir, "football-ball-detection.pt")
    for p in (pitch_pt, ball_pt):
        if not os.path.exists(p):
            raise SystemExit(f"missing checkpoint: {p}")

    ball_df = run_ball_on_frames(
        args.source, frames, pitch_pt=pitch_pt, ball_pt=ball_pt, cfg=cfg,
    )
    ball_path = os.path.join(args.out_dir, "ball_windows.parquet")
    ball_df.to_parquet(ball_path, index=False)
    print(
        f"[twopass] detected ball in {len(ball_df)} frames -> {ball_path}\n"
        f"[twopass] feed it to the prerequisites' ball smoothing for those windows."
    )


if __name__ == "__main__":
    main()
