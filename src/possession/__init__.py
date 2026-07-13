"""Possession zone — the first Layer 2 component.

Assigns a **per-frame ball possessor** from the cleaned game state produced by
the prerequisite stage. This is the primitive the event taxonomy is built on: a
pass, a turnover, a duel and a dribble are all statements about *how the
possessor changes between frames*, so the possessor stream has to exist, and be
trustworthy, before any of that can be defined.

Deliberately conservative. It answers exactly one question -- "who, if anyone,
is on the ball in this frame?" -- and refuses to guess beyond it:

- no usable ball -> ``no_ball`` (occlusion, **not** a stoppage), no possessor
- nobody within R_pz -> ``loose`` (pass in flight / loose ball), no possessor
- exactly one within R_pz -> ``possession``
- two or more within R_pz -> ``contested``: the nearest is named, and the frame
  is flagged for a later duel-resolution step rather than resolved by guessing

    from src.possession import PossessionConfig, detect_possession

Consumes the *prepared* tracking table (needs ``stable_id`` + the target-frame
coordinates from ``src.prerequisites``). Depends only on pandas / numpy -- not
the Layer 1 CV stack. **No event logic lives here.**
"""

from __future__ import annotations

from .config import PossessionConfig, config_from_prep_meta
from .pipeline import detect_possession, summarize
from .segments import possession_segments
from .sweep import format_sweep, radius_grid, sweep_radii
from .zone import (
    ball_positions,
    ball_to_player_distances,
    candidates,
    possession_frames,
)

# NOTE: `review` (the video overlay) is deliberately NOT imported here -- it is
# the only part that needs cv2, and importing this package must stay cheap and
# dependency-light. Import it directly: `from src.possession.review import ...`.

__all__ = [
    "PossessionConfig",
    "config_from_prep_meta",
    "possession_frames",
    "ball_positions",
    "ball_to_player_distances",
    "candidates",
    "possession_segments",
    "detect_possession",
    "summarize",
    "sweep_radii",
    "radius_grid",
    "format_sweep",
]
