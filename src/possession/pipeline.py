"""Compose the possession-zone detector: frames -> segments -> summary.

``detect_possession`` is the one call the CLI (and the next layer) needs.

The summary is the honest scorecard for the stage, so it reports its
denominators explicitly rather than leaving them to be guessed:

- **coverage** = attributed frames / **ball** frames. The ceiling is ball
  *presence*, not the frame count: on clip 2e57b9_0 only 90.5% of frames have a
  usable smoothed ball at all, so coverage is measured against those. (The share
  of *all* frames is also reported, as ``coverage_all_frames_pct``.)
- **clean** and **duel** = possession / contested as a share of the **attributed**
  frames (they sum to 100%). "Clean" means exactly one candidate was in the zone.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import (
    STATE_CONTESTED,
    STATE_LOOSE,
    STATE_NO_BALL,
    STATE_POSSESSION,
    STATES,
    PossessionConfig,
)
from .segments import possession_segments
from .zone import possession_frames


def summarize(
    frames: pd.DataFrame, segments: pd.DataFrame, cfg: PossessionConfig
) -> dict:
    """Coverage / clean / duel / team-split summary for the stage meta."""
    counts = frames["state"].value_counts()
    n = {s: int(counts.get(s, 0)) for s in STATES}

    n_frames = int(len(frames))
    n_ball = n[STATE_LOOSE] + n[STATE_POSSESSION] + n[STATE_CONTESTED]
    n_attr = n[STATE_POSSESSION] + n[STATE_CONTESTED]

    def pct(num: int, den: int) -> float:
        return round(100.0 * num / den, 2) if den else 0.0

    # team possession split: share of ATTRIBUTED frames each team held
    attributed = frames[frames["possessor_id"].notna()]
    split = attributed["possessor_team"].value_counts(dropna=True)
    team_split = {
        str(int(t)): pct(int(c), n_attr) for t, c in split.items() if pd.notna(t)
    }

    hold = segments["n_frames"]
    return dict(
        config=cfg.as_meta(),
        r_pz_m=cfg.r_pz_m,
        n_frames=n_frames,
        n_ball_frames=n_ball,
        n_attributed_frames=n_attr,
        states={s: n[s] for s in STATES},
        # coverage: of the frames where the ball is actually visible, how many
        # did we manage to attribute to a player?
        coverage_pct=pct(n_attr, n_ball),
        coverage_all_frames_pct=pct(n_attr, n_frames),
        ball_presence_pct=pct(n_ball, n_frames),  # the ceiling coverage can reach
        # of the attributed frames, how many were unambiguous vs. a duel?
        clean_pct=pct(n[STATE_POSSESSION], n_attr),
        duel_pct=pct(n[STATE_CONTESTED], n_attr),
        team_possession_pct=team_split,
        n_segments=int(len(segments)),
        median_hold_frames=float(hold.median()) if len(hold) else 0.0,
        mean_hold_frames=round(float(hold.mean()), 2) if len(hold) else 0.0,
        note=(
            "coverage denominator = frames with a usable smoothed ball "
            "(no_ball frames are occlusion, not stoppages); clean/duel "
            "denominator = attributed frames. r_pz_m=3.0 was calibrated on an "
            "open-play clip with no congested box -- treat it as an upper bound "
            "and re-run `sweep_radii` on crowded footage."
        ),
    )


def detect_possession(
    df: pd.DataFrame, cfg: PossessionConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the detector on a prepared tracking table.

    Returns ``(frames, segments, meta)``: the per-frame possession table, the
    collapsed possession segments (touches), and the summary.
    """
    frames = possession_frames(df, cfg)
    segments = possession_segments(frames)
    meta = summarize(frames, segments, cfg)
    return frames, segments, meta
