"""Transform 3 — coordinate rescale (source pitch -> target convention).

Layer 1 emits positions on the source pitch declared in ``meta.json`` (here
120x70 m, from ``SoccerPitchConfiguration``). Most valuation models expect a
standard pitch — 105x68 by default, 120x80 also supported. This is a linear
rescale that preserves the top-left origin:

    x' = x * L_target / L_source ,   y' = y * W_target / W_source

Applied identically to players, goalkeepers and ball. New columns
``pitch_x_t_m`` / ``pitch_y_t_m`` are added; ``pitch_x_m`` / ``pitch_y_m`` are
never overwritten.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import COL_PITCH_X_T, COL_PITCH_Y_T, PrepConfig


def rescale_coords(
    df: pd.DataFrame,
    cfg: PrepConfig,
    src_x: str = "pitch_x_m",
    src_y: str = "pitch_y_m",
) -> Tuple[pd.DataFrame, dict]:
    """Add ``pitch_x_t_m`` / ``pitch_y_t_m`` on the target pitch convention.

    NaN source coords propagate to NaN targets. Returns ``(df, meta)`` recording
    the source/target dimensions and the applied scale factors.
    """
    df = df.copy()
    sx = cfg.target_length_m / cfg.source_length_m
    sy = cfg.target_width_m / cfg.source_width_m

    df[COL_PITCH_X_T] = df[src_x] * sx
    df[COL_PITCH_Y_T] = df[src_y] * sy

    meta = dict(
        source_pitch_m=[cfg.source_length_m, cfg.source_width_m],
        target_pitch_m=[cfg.target_length_m, cfg.target_width_m],
        scale=[round(sx, 6), round(sy, 6)],
        origin="top-left (preserved)",
        columns=[COL_PITCH_X_T, COL_PITCH_Y_T],
    )
    return df, meta
