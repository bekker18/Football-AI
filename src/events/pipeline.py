"""Compose the ball-free high-value window emitter.

``detect_high_value_windows`` = per-frame signals -> assembled windows, plus a
coverage summary. The coverage fraction (share of frames inside a window) is the
number that justifies the two-pass design: it is the slice of the match the
expensive ball detector would run on instead of all of it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import EventConfig
from .signals import attacking_signals
from .windows import windows_from_signals


def detect_high_value_windows(
    df: pd.DataFrame, cfg: EventConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the emitter on a prepared tracking table.

    Returns ``(windows, signals, meta)``: the scored windows, the per-frame value
    stream they were built from, and a summary (config + coverage).
    """
    signals = attacking_signals(df, cfg)
    windows = windows_from_signals(signals, cfg)

    n_frames = int(len(signals))
    covered = int(
        sum(int(r.end_frame) - int(r.start_frame) + 1 for r in windows.itertuples())
    )
    meta = dict(
        config=cfg.as_meta(),
        n_frames=n_frames,
        n_windows=int(len(windows)),
        frames_covered=covered,
        coverage_frac=round(covered / n_frames, 4) if n_frames else 0.0,
        value_score=dict(
            mean=round(float(signals["value_score"].mean()), 4) if n_frames else 0.0,
            max=round(float(signals["value_score"].max()), 4) if n_frames else 0.0,
            threshold=cfg.value_threshold,
        ),
    )
    return windows, signals, meta
