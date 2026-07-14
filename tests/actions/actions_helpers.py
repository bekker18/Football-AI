"""Synthetic fixture builders for the event-layer tests.

Building a possession stream + a prepared tracking table by hand for every test
is noisy and gets copy-pasted wrong. These helpers let a test say what it means:

    frames = stream(
        hold(player=1, team=0, frames=range(0, 10)),
        loose(range(10, 16)),
        hold(player=2, team=0, frames=range(16, 26)),
    )

...and then place the ball and the players where the scenario needs them.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.actions.config import (
    STATE_CONTESTED,
    STATE_LOOSE,
    STATE_NO_BALL,
    STATE_POSSESSION,
    ActionConfig,
)

FPS = 25.0

Point = Tuple[float, float]


def hold(player: int, team: int, frames: Iterable[int],
         state: str = STATE_POSSESSION) -> List[dict]:
    """Frames where ``player`` is the possessor."""
    return [
        dict(frame=int(f), time_s=f / FPS, state=state,
             possessor_id=player, possessor_team=team)
        for f in frames
    ]


def contested(player: int, team: int, frames: Iterable[int]) -> List[dict]:
    """Frames where ``player`` is the nearest of several candidates in the zone."""
    return hold(player, team, frames, state=STATE_CONTESTED)


def loose(frames: Iterable[int]) -> List[dict]:
    """Frames where the ball is visible but nobody is within the zone."""
    return [
        dict(frame=int(f), time_s=f / FPS, state=STATE_LOOSE,
             possessor_id=None, possessor_team=None)
        for f in frames
    ]


def no_ball(frames: Iterable[int]) -> List[dict]:
    """Frames where the ball is occluded -- NOT a stoppage."""
    return [
        dict(frame=int(f), time_s=f / FPS, state=STATE_NO_BALL,
             possessor_id=None, possessor_team=None)
        for f in frames
    ]


def stream(*parts: List[dict]) -> pd.DataFrame:
    """Concatenate frame-runs into a possession_frames-shaped table."""
    rows = [r for part in parts for r in part]
    df = pd.DataFrame(rows).sort_values("frame", kind="stable").reset_index(drop=True)
    df["possessor_id"] = df["possessor_id"].astype("Int64")
    df["possessor_team"] = df["possessor_team"].astype("float64")
    return df


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def ball_line(frames: Iterable[int], start: Point, end: Point) -> Dict[int, Point]:
    """Ball moving in a straight line from ``start`` to ``end`` over ``frames``."""
    fs = list(frames)
    if len(fs) == 1:
        return {fs[0]: start}
    return {
        f: _lerp(start, end, i / (len(fs) - 1)) for i, f in enumerate(fs)
    }


def ball_still(frames: Iterable[int], at: Point) -> Dict[int, Point]:
    """Ball sitting at one spot."""
    return {int(f): at for f in frames}


# --------------------------------------------------------------------------- #
# image space -- where the aerial detector lives
# --------------------------------------------------------------------------- #
def aerial_img_y(
    frames: Iterable[int], y_ground: float = 352.0, apex_rise_px: float = 27.0
) -> Dict[int, float]:
    """The ball's IMAGE-y as it rises and falls: a clean up-then-down arc.

    The camera looks *down* at the pitch, so a ball going UP moves UP the image and
    ``img_y`` **decreases**. A flight is therefore a **local MINIMUM** in img_y --
    an upward-opening parabola -- and that is what the detector fits. Getting this
    sign backwards is the easiest possible way to build a detector that finds
    nothing, so the fixture states it explicitly: img_y starts and ends at
    ``y_ground`` and dips ``apex_rise_px`` at the apex.

    Modelled on the sample clip's aerial pass (frames 123-150 of 2e57b9_0): img_y
    falls 352 -> ~325 and comes back to ~350 over ~1.1 s.
    """
    fs = [int(f) for f in frames]
    if len(fs) < 2:
        return {f: y_ground for f in fs}
    mid = 0.5 * (fs[0] + fs[-1])
    half = 0.5 * (fs[-1] - fs[0])
    return {
        f: float(y_ground - apex_rise_px * (1.0 - ((f - mid) / half) ** 2))
        for f in fs
    }


def flat_img_y(
    frames: Iterable[int], y: float = 400.0, jitter: float = 0.0
) -> Dict[int, float]:
    """A ball on the grass: img_y goes nowhere. Deterministic jitter, never random."""
    return {
        int(f): float(y + jitter * ((int(f) % 3) - 1)) for f in frames
    }


def _bbox_height(img_y: float) -> float:
    """A crude but honest projection model: further up the image => smaller ball.

    A ball at the top of its arc is further from the camera, so it images smaller.
    ``img_y`` and the bbox height therefore rise and fall **together**, which is the
    independent corroboration the detector looks for (a camera tilt moves img_y
    without changing the ball's apparent size).
    """
    return float(max(4.0, 9.0 + 0.05 * (img_y - 330.0)))


def _ball_speeds(ball: Dict[int, Optional[Point]]) -> Dict[int, float]:
    """Apparent GROUND speed per frame, m/s, from consecutive ball positions.

    This is the same (distorted) quantity ``prerequisites.smooth_ball`` writes into
    ``ball_speed_ms`` -- the aerial detector's speed floor reads it, and only ever
    asks "fast?", never "how fast?".
    """
    known = {f: pt for f, pt in ball.items() if pt is not None}
    fs = sorted(known)
    out: Dict[int, float] = {}
    for i, f in enumerate(fs):
        lo = known[fs[max(0, i - 1)]]
        hi = known[fs[min(len(fs) - 1, i + 1)]]
        span = fs[min(len(fs) - 1, i + 1)] - fs[max(0, i - 1)]
        if span <= 0:
            out[f] = 0.0
            continue
        d = float(np.hypot(hi[0] - lo[0], hi[1] - lo[1]))
        out[f] = d / (span / FPS)
    return out


def tracking(
    ball: Dict[int, Optional[Point]],
    players: Dict[int, Dict[int, Point]],
    teams: Dict[int, int],
    img: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """A prepared-tracking-shaped table: ball rows + player rows, target frame.

    ``ball`` maps frame -> position, or frame -> ``None`` for an occluded frame
    (the ball row exists but its smoothed coordinate is null, exactly as the
    prerequisites emit it). ``players`` maps stable_id -> {frame: position}.

    ``img`` maps frame -> ``img_y``, the ball's position in **image** space. Supply
    it for the tests that care whether the ball was off the ground -- that question
    can only be answered upstream of the homography, so it is the one thing the
    pitch-metre columns above cannot express. ``bbox_y1``/``bbox_y2`` and
    ``ball_speed_ms`` are derived, so a fixture states the arc once and the
    corroborating signals follow from it rather than being hand-tuned to agree.
    Omit it entirely and the layer must still run -- just never flagging anything.
    """
    speeds = _ball_speeds(ball)
    rows: List[dict] = []
    for f, pt in sorted(ball.items()):
        iy = None if img is None else img.get(int(f))
        h = np.nan if iy is None else _bbox_height(iy)
        rows.append(dict(
            frame=int(f), time_s=f / FPS, object_id=0, role="ball", team=np.nan,
            stable_id=pd.NA,
            pitch_x_t_m=np.nan, pitch_y_t_m=np.nan,
            ball_x_ts_m=np.nan if pt is None else pt[0],
            ball_y_ts_m=np.nan if pt is None else pt[1],
            img_y=np.nan if iy is None else float(iy),
            bbox_y1=np.nan if iy is None else float(iy) - h,
            bbox_y2=np.nan if iy is None else float(iy),
            ball_speed_ms=speeds.get(int(f), np.nan),
            ball_outlier=False,
        ))
    for pid, track in players.items():
        for f, pt in sorted(track.items()):
            rows.append(dict(
                frame=int(f), time_s=f / FPS, object_id=100 + pid, role="player",
                team=float(teams[pid]), stable_id=pid,
                pitch_x_t_m=pt[0], pitch_y_t_m=pt[1],
                ball_x_ts_m=np.nan, ball_y_ts_m=np.nan,
                img_y=np.nan, bbox_y1=np.nan, bbox_y2=np.nan,
                ball_speed_ms=np.nan, ball_outlier=False,
            ))
    df = pd.DataFrame(rows).sort_values(["frame", "object_id"], kind="stable")
    df["stable_id"] = df["stable_id"].astype("Int64")
    return df.reset_index(drop=True)


def near(ball: Dict[int, Optional[Point]], frames: Iterable[int],
         offset: Point = (0.5, 0.5)) -> Dict[int, Point]:
    """A player track that shadows the ball -- what a possessor looks like.

    Used when a test cares about the ball's path, not the player's, but the
    possessor still has to *be* somewhere (and be inside the possession zone,
    which is what makes them the possessor in the first place).
    """
    out: Dict[int, Point] = {}
    last: Point = (0.0, 0.0)
    for f in frames:
        pt = ball.get(int(f))
        if pt is not None:
            last = pt
        out[int(f)] = (last[0] + offset[0], last[1] + offset[1])
    return out


def config(**overrides) -> ActionConfig:
    """An ActionConfig with the two teams pointing in opposite directions.

    team 0 attacks toward +x, team 1 toward -x -- the same convention the
    prerequisites resolve from goalkeeper positions.
    """
    cfg = ActionConfig(fps=FPS, attack_dir={0: 1, 1: -1}, game_id="test")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
