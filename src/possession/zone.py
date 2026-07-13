"""Per-frame possession-zone classification — the Layer 2 primitive.

For every frame we measure the ball-to-player distance for each candidate
(players + goalkeepers; referees excluded) and classify the frame into exactly
one of four states:

===============  ===========================================================
``no_ball``      no usable smoothed ball position -> **occlusion, not a
                 stoppage**. Never a possessor. (Dead-ball reasoning lives in
                 the prerequisites' ``in_play`` flag, not here.)
``loose``        ball present but NO candidate within ``r_pz_m`` -- a pass in
                 flight or a genuinely loose ball. Never a possessor.
``possession``   exactly ONE candidate within ``r_pz_m`` -> that player is the
                 possessor.
``contested``    TWO OR MORE candidates within ``r_pz_m`` -> the **nearest** is
                 named possessor and the frame is flagged. We do not try to
                 resolve the duel any further; a later duel-resolution step can
                 use the flag.
===============  ===========================================================

Two invariants this module guarantees, and that the tests pin:

1. A possessor is **never fabricated**: ``possessor_id`` is null on every
   ``loose`` and ``no_ball`` frame.
2. Distances are computed **entirely within the target (105x68) frame** -- see
   the coordinate contract in ``config.py``.

Identity is ``stable_id`` (the prerequisites' stitched track id), not the raw
ByteTrack ``object_id``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BALL_OBJECT_ID,
    COL_BALL_XTS,
    COL_BALL_YTS,
    COL_PITCH_X_T,
    COL_PITCH_Y_T,
    COL_STABLE_ID,
    PEOPLE_ROLES,
    STATE_CONTESTED,
    STATE_LOOSE,
    STATE_NO_BALL,
    STATE_POSSESSION,
    PossessionConfig,
)

# internal working columns (leading underscore => never emitted)
_BX, _BY, _DIST, _IN_ZONE = "_ball_x", "_ball_y", "_dist", "_in_zone"

FRAME_COLUMNS = [
    "frame", "time_s", "state", "possessor_id", "possessor_team",
    "dist_m", "n_in_zone",
]


def _require_columns(df: pd.DataFrame) -> None:
    """Fail loudly if the table isn't the prepared (Layer 2-ready) one."""
    missing = [
        c for c in (COL_STABLE_ID, COL_PITCH_X_T, COL_PITCH_Y_T,
                    COL_BALL_XTS, COL_BALL_YTS)
        if c not in df.columns
    ]
    if missing:
        raise KeyError(
            f"possession layer needs prepared column(s) {missing}; run "
            f"`python -m src.prerequisites run_prerequisites` first."
        )


def ball_positions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per frame that has a **usable** smoothed+rescaled ball position.

    Frames whose ball row is missing entirely, or present with a null
    ``ball_x_ts_m`` (rejected as an outlier and not interpolated), are simply
    absent from the result -- those become ``no_ball``.

    The ball columns are renamed to private names because the prepared table
    carries ``ball_x_ts_m`` on *every* row (null off the ball row), so a naive
    join back onto the candidates would collide.
    """
    ball = df[df["object_id"] == BALL_OBJECT_ID]
    ball = ball.dropna(subset=[COL_BALL_XTS, COL_BALL_YTS])
    out = (
        ball[["frame", COL_BALL_XTS, COL_BALL_YTS]]
        .rename(columns={COL_BALL_XTS: _BX, COL_BALL_YTS: _BY})
        .groupby("frame", as_index=False)  # defensive: one ball row per frame
        .first()
    )
    return out


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Possession candidates: players + goalkeepers with a usable position.

    Referees are excluded (they are not possession candidates), as are rows with
    a null target-frame coordinate or a null ``stable_id`` -- we can't measure or
    name those.
    """
    cand = df[df["role"].isin(PEOPLE_ROLES)]
    cols = ["frame", COL_STABLE_ID, "team", COL_PITCH_X_T, COL_PITCH_Y_T]
    return cand.dropna(subset=[COL_STABLE_ID, COL_PITCH_X_T, COL_PITCH_Y_T])[cols]


def ball_to_player_distances(
    df: pd.DataFrame, cfg: PossessionConfig
) -> pd.DataFrame:
    """Every (frame, candidate) pair that has a ball, with its distance.

    One vectorized ``np.hypot`` over the whole merged table -- no Python loop
    over players. Frames with no ball, or with no candidates, simply produce no
    rows here and are handled by the caller.

    Public because the review overlay needs the same per-candidate distances (to
    draw *everyone* inside the zone, not just the possessor) and must not
    re-derive them independently.
    """
    cand = candidates(df)
    ball = ball_positions(df)
    if cand.empty or ball.empty:
        # nothing to measure (no candidates anywhere, or no ball anywhere): an
        # empty pair table, correctly typed, so the caller's merges still work
        empty = cand.iloc[0:0].copy()
        for col in (_BX, _BY, _DIST):
            empty[col] = np.array([], dtype=float)
        empty[_IN_ZONE] = np.array([], dtype=bool)
        return empty

    merged = cand.merge(ball, on="frame", how="inner")

    dx = merged[COL_PITCH_X_T].to_numpy(dtype=float) - merged[_BX].to_numpy(dtype=float)
    dy = merged[COL_PITCH_Y_T].to_numpy(dtype=float) - merged[_BY].to_numpy(dtype=float)
    dist = np.hypot(dx, dy)

    merged[_DIST] = dist
    merged[_IN_ZONE] = dist <= cfg.r_pz_m
    return merged


def possession_frames(df: pd.DataFrame, cfg: PossessionConfig) -> pd.DataFrame:
    """Classify every frame and name a possessor where one is warranted.

    Returns one row per frame of the input table, with columns::

        frame, time_s, state, possessor_id, possessor_team, dist_m, n_in_zone

    ``dist_m`` is the distance to the **nearest candidate** whenever a ball and
    at least one candidate exist -- it is reported on ``loose`` frames too (it is
    the diagnostic that says *how* loose), and is null only when there is nothing
    to measure. ``possessor_id`` / ``possessor_team``, by contrast, are populated
    **only** on ``possession`` / ``contested`` frames.
    """
    _require_columns(df)

    all_frames = np.sort(df["frame"].unique())
    out = pd.DataFrame({"frame": all_frames})

    pairs = ball_to_player_distances(df, cfg)
    has_ball = set(ball_positions(df)["frame"].tolist())

    if len(pairs):
        # nearest candidate per frame: stable sort then take the first row
        nearest = (
            pairs.sort_values(["frame", _DIST], kind="stable")
            .groupby("frame", as_index=False)
            .first()
        )
        n_in_zone = (
            pairs.groupby("frame")[_IN_ZONE].sum().rename("n_in_zone").reset_index()
        )
        out = out.merge(
            nearest[["frame", COL_STABLE_ID, "team", _DIST]], on="frame", how="left"
        ).merge(n_in_zone, on="frame", how="left")
    else:
        out[COL_STABLE_ID] = np.nan
        out["team"] = np.nan
        out[_DIST] = np.nan
        out["n_in_zone"] = np.nan

    out["n_in_zone"] = out["n_in_zone"].fillna(0).astype(int)

    # --- state machine, vectorized -------------------------------------- #
    ball_present = out["frame"].isin(has_ball).to_numpy()
    n_in = out["n_in_zone"].to_numpy()

    state = np.where(
        ~ball_present, STATE_NO_BALL,
        np.where(n_in == 0, STATE_LOOSE,
                 np.where(n_in == 1, STATE_POSSESSION, STATE_CONTESTED)),
    )
    out["state"] = state

    # a possessor exists only where somebody is actually inside the zone; this
    # is what makes "never fabricate a possessor on a loose/no_ball frame" true
    # by construction rather than by convention.
    attributed = ball_present & (n_in >= 1)
    out["possessor_id"] = out[COL_STABLE_ID].where(attributed)
    out["possessor_team"] = out["team"].where(attributed)
    out["dist_m"] = out[_DIST].where(ball_present)
    out["n_in_zone"] = np.where(ball_present, n_in, 0)

    if "time_s" in df.columns:
        time_by_frame = df.groupby("frame")["time_s"].first()
        out["time_s"] = out["frame"].map(time_by_frame)
    else:
        out["time_s"] = out["frame"] / cfg.fps

    return out[FRAME_COLUMNS].reset_index(drop=True)
