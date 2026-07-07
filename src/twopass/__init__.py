"""Two-pass controller — gate the expensive ball detector onto flagged windows.

The mechanism that makes "reserve ball-based eventing for high-value clips" (see
docs/strategy.md) automatic:

1. Pass 1 (cheap, ball-free, whole match): Layer 1 in player-only mode +
   ``src.events`` -> a stream of high-value windows.
2. **Gate** (:func:`plan_ball_frames`): pick which windows to spend the ball
   budget on and turn them into the exact set of frames to re-decode. Pure logic,
   fully unit-tested, no CV stack.
3. Pass 2 (:func:`run_ball_on_frames`, needs the CV stack): re-decode only those
   frames, run the ball detector + homography there, emit a **sparse** ball
   table covering the windows.

Net effect: the ball cost lands on a few percent of the match instead of all of
it. The gate/plan half depends only on pandas / numpy; only the Pass 2 executor
imports the Layer 1 CV stack.
"""

from __future__ import annotations

from .config import TwoPassConfig
from .plan import frames_to_ranges, plan_ball_frames

__all__ = [
    "TwoPassConfig",
    "plan_ball_frames",
    "frames_to_ranges",
]
