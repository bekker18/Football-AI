"""Transform 3 — coordinate rescale (source pitch -> target convention).

Layer 1 emits positions on the source pitch declared in ``meta.json`` (here
120x70 m, from ``SoccerPitchConfiguration``). Most valuation models expect a
standard pitch — 105x68 by default, 120x80 also supported. This is a linear
rescale that preserves the top-left origin:

    x' = x * L_target / L_source ,   y' = y * W_target / W_source

Applied identically to players, goalkeepers and ball. New columns
``pitch_x_t_m`` / ``pitch_y_t_m`` are added; ``pitch_x_m`` / ``pitch_y_m`` are
never overwritten.

If the smoothed ball columns (``ball_x_s_m`` / ``ball_y_s_m``, in the SOURCE
frame) are present, the same rescale is also applied to them, emitting
``ball_x_ts_m`` / ``ball_y_ts_m`` in the TARGET frame. Without this bridge any
downstream code mixing the smoothed ball with rescaled player coordinates would
be silently off by the scale factor. ``prep_meta`` records which frame each ball
column lives in.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import (
    COL_BALL_XS,
    COL_BALL_XTS,
    COL_BALL_YS,
    COL_BALL_YTS,
    COL_PITCH_X_T,
    COL_PITCH_Y_T,
    PrepConfig,
)


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

    columns = [COL_PITCH_X_T, COL_PITCH_Y_T]
    # bridge the smoothed ball into the target frame too, when it exists
    if COL_BALL_XS in df.columns and COL_BALL_YS in df.columns:
        df[COL_BALL_XTS] = df[COL_BALL_XS] * sx
        df[COL_BALL_YTS] = df[COL_BALL_YS] * sy
        columns += [COL_BALL_XTS, COL_BALL_YTS]

    src_frame = f"source {cfg.source_length_m:g}x{cfg.source_width_m:g}"
    tgt_frame = f"target {cfg.target_length_m:g}x{cfg.target_width_m:g}"
    ball_frames = {
        "pitch_x_m/pitch_y_m": src_frame,
        COL_PITCH_X_T + "/" + COL_PITCH_Y_T: tgt_frame,
        COL_BALL_XS + "/" + COL_BALL_YS: f"{src_frame} (smoothed ball)",
        COL_BALL_XTS + "/" + COL_BALL_YTS: f"{tgt_frame} (smoothed ball)",
    }

    meta = dict(
        source_pitch_m=[cfg.source_length_m, cfg.source_width_m],
        target_pitch_m=[cfg.target_length_m, cfg.target_width_m],
        scale=[round(sx, 6), round(sy, 6)],
        origin="top-left (preserved)",
        columns=columns,
        coordinate_frames=ball_frames,
    )
    return df, meta
