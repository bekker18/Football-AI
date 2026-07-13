"""Collapse the per-frame possessor stream into possession SEGMENTS (touches).

A segment is a **maximal run of consecutive frames with the same possessor**.
This is the table the next layer (the event taxonomy) will read: a pass, a
turnover and a dribble are all statements *about the boundary between two
segments*, so the segments have to exist before any of that can be defined.

Deliberately literal, so it stays inspectable:

- A ``loose`` or ``no_ball`` frame has no possessor, so it **breaks** the run.
  We do NOT bridge a possessor across a gap -- "the player kept the ball while
  the ball was briefly occluded / momentarily >R_pz away" is a *hold* heuristic,
  and that belongs to the event layer, not to the primitive.
- A jump in frame number also breaks the run (segments are contiguous in time).
- ``contested`` frames DO carry a possessor (the nearest player), so they extend
  that player's segment; each segment reports ``n_contested`` so a later
  duel-resolution step can see how much of the touch was actually a duel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STATE_CONTESTED

SEGMENT_COLUMNS = [
    "possessor_id", "team", "start_frame", "end_frame", "n_frames",
    "start_time_s", "end_time_s", "n_contested",
]


def possession_segments(frames: pd.DataFrame) -> pd.DataFrame:
    """Collapse maximal same-possessor runs in a per-frame table into touches.

    Takes the output of :func:`~src.possession.zone.possession_frames` and
    returns one row per segment::

        possessor_id, team, start_frame, end_frame, n_frames,
        start_time_s, end_time_s, n_contested

    ``n_frames`` is the frame count of the touch (== ``end_frame - start_frame +
    1``, since a segment is contiguous by construction).
    """
    held = frames[frames["possessor_id"].notna()].sort_values("frame", kind="stable")
    if held.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    pid = held["possessor_id"].to_numpy()
    frame_no = held["frame"].to_numpy()

    # a new segment starts where the possessor changes OR the frames aren't
    # adjacent (a break in the run -- loose ball, occlusion, or a frame gap)
    new_segment = np.ones(len(held), dtype=bool)
    new_segment[1:] = (pid[1:] != pid[:-1]) | (frame_no[1:] != frame_no[:-1] + 1)
    seg_id = np.cumsum(new_segment) - 1

    held = held.assign(
        _seg=seg_id,
        _contested=(held["state"] == STATE_CONTESTED).astype(int),
    )

    out = (
        held.groupby("_seg", as_index=False)
        .agg(
            possessor_id=("possessor_id", "first"),
            team=("possessor_team", "first"),
            start_frame=("frame", "min"),
            end_frame=("frame", "max"),
            n_frames=("frame", "size"),
            start_time_s=("time_s", "min"),
            end_time_s=("time_s", "max"),
            n_contested=("_contested", "sum"),
        )
        .drop(columns="_seg")
    )
    return out[SEGMENT_COLUMNS].reset_index(drop=True)
