"""Transform 5 — dead-ball / in-play flag (heuristic proxy).

Emits a per-frame ``in_play`` boolean plus an ``in_play_conf`` confidence. This
is a *tunable proxy*, NOT a ground-truth stoppage signal, and it deliberately
does **not** equate ball absence with a dead ball (absence here is occlusion).

A frame reads as NOT in play when the smoothed ball is either
  * out of bounds beyond a margin (``oob_margin_m``) of the pitch extents, or
  * effectively stationary (``< still_speed_ms``) near a boundary
    (``near_boundary_m``) for a sustained run (``>= still_frames``).
While the ball is absent the last decision is carried forward with a decaying
confidence, so occlusion never fabricates a stoppage.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import (
    BALL_OBJECT_ID,
    COL_BALL_SPEED,
    COL_BALL_XS,
    COL_BALL_YS,
    COL_IN_PLAY,
    COL_IN_PLAY_CONF,
    PrepConfig,
)


def _clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def synth_dead_ball(df: pd.DataFrame, cfg: PrepConfig) -> Tuple[pd.DataFrame, dict]:
    """Add per-row ``in_play`` (bool) + ``in_play_conf`` (0..1), broadcast per frame.

    Uses the smoothed ball position/speed when available (falls back to raw
    ``pitch_x_m`` if :func:`smooth_ball` has not run). Returns ``(df, meta)``.
    """
    df = df.copy()
    L, W = cfg.source_length_m, cfg.source_width_m

    x_col = COL_BALL_XS if COL_BALL_XS in df.columns else "pitch_x_m"
    y_col = COL_BALL_YS if COL_BALL_YS in df.columns else "pitch_y_m"

    ball = df[df["object_id"] == BALL_OBJECT_ID].dropna(subset=[x_col, y_col])
    ball = ball.sort_values("frame").drop_duplicates("frame", keep="last")
    bx = dict(zip(ball["frame"].astype(int), ball[x_col].astype(float)))
    by = dict(zip(ball["frame"].astype(int), ball[y_col].astype(float)))
    if COL_BALL_SPEED in ball.columns:
        bsp = dict(zip(ball["frame"].astype(int), ball[COL_BALL_SPEED].astype(float)))
    else:
        bsp = {}

    frames = sorted(int(f) for f in df["frame"].unique())

    # first pass: per-present-frame raw signals
    still_run = 0
    decided_in_play = {}
    decided_conf = {}
    prev_in_play = True  # assume the clip opens in play
    prev_conf = cfg.absent_conf_floor
    n_oob = n_still = n_absent = 0

    for f in frames:
        if f in bx:
            x, y = bx[f], by[f]
            speed = bsp.get(f, np.nan)
            dist_beyond = max(0.0, -x, x - L, -y, y - W)
            min_edge = min(x, L - x, y, W - y)
            near = min_edge <= cfg.near_boundary_m
            slow = (not np.isnan(speed)) and speed < cfg.still_speed_ms

            if near and slow:
                still_run += 1
            else:
                still_run = 0

            oob = dist_beyond > cfg.oob_margin_m
            still_dead = still_run >= cfg.still_frames

            if oob:
                in_play = False
                conf = _clamp01(0.5 + 0.5 * (dist_beyond - cfg.oob_margin_m) / max(cfg.oob_margin_m, 1e-6))
                n_oob += 1
            elif still_dead:
                in_play = False
                conf = _clamp01(0.5 + 0.5 * (still_run - cfg.still_frames) / max(cfg.still_frames, 1))
                n_still += 1
            else:
                in_play = True
                conf = _clamp01(0.5 + 0.5 * min_edge / max(cfg.near_boundary_m, 1e-6))

            prev_in_play, prev_conf = in_play, conf
        else:
            # ball absent -> occlusion, NOT a stoppage: carry the last decision,
            # decay confidence toward the floor.
            in_play = prev_in_play
            conf = max(cfg.absent_conf_floor, prev_conf * cfg.absent_conf_decay)
            prev_conf = conf
            still_run = 0
            n_absent += 1

        decided_in_play[f] = bool(in_play)
        decided_conf[f] = round(float(conf), 4)

    df[COL_IN_PLAY] = df["frame"].astype(int).map(decided_in_play).astype("boolean")
    df[COL_IN_PLAY_CONF] = df["frame"].astype(int).map(decided_conf).astype(float)

    n_in_play = sum(decided_in_play.values())
    meta = dict(
        params=dict(
            oob_margin_m=cfg.oob_margin_m,
            still_speed_ms=cfg.still_speed_ms,
            still_frames=cfg.still_frames,
            near_boundary_m=cfg.near_boundary_m,
            absent_conf_decay=cfg.absent_conf_decay,
            absent_conf_floor=cfg.absent_conf_floor,
        ),
        n_frames=len(frames),
        n_in_play=int(n_in_play),
        n_dead=int(len(frames) - n_in_play),
        n_frames_ball_oob=n_oob,
        n_frames_ball_still=n_still,
        n_frames_ball_absent=n_absent,
        source=x_col,
        note=(
            "HEURISTIC PROXY, not a ground-truth stoppage signal. Tune "
            "oob_margin_m / still_speed_ms / still_frames / near_boundary_m per "
            "footage. Ball absence is treated as occlusion, never as a dead ball."
        ),
    )
    return df, meta
