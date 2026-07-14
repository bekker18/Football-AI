"""Layer 2 event layer: possession transitions -> SPADL actions.

The possession-zone detector (``src.possession``) says *who is on the ball on
each frame*. This layer says *what happened*, by walking the **transitions**
between possession segments and naming each one: a pass, a cross, a carry, a
turnover.

Milestone 1 emits only the events that fall directly out of those transitions --
``pass``, ``cross``, ``dribble``, ``interception``, ``tackle``, ``bad_touch``.
Shots, set pieces and duel resolution are deliberately out of scope (see the
``EXTENSION POINT`` markers in :mod:`src.actions.transitions`).

Every action's ``start`` / ``end`` comes from the **players** -- the passer's
position at release, the receiver's at reception -- and never from the ball. The
homography only knows the ground plane (z=0), so an airborne ball's coordinates
are stretched away from the camera and would otherwise put fiction straight into
xT and VAEP. :mod:`src.actions.aerial` flags which frames those are.

Output is SPADL, so ``socceraction`` can compute xT / VAEP on it without an
adapter. Note this package is **not** ``src.events`` -- that one is the ball-free
high-value *window* emitter used to schedule the Layer 1 ball detector, and is
unrelated.

Kept dependency-light on purpose: pandas + numpy only. ``socceraction`` is a test
dependency (it pins our SPADL vocabulary and validates the output), never a
runtime one.
"""

from .aerial import AerialRun, AerialTrack, detect_airborne
from .config import ActionConfig, config_from_prep_meta
from .pipeline import detect_actions, run_stages
from .source import (
    PossessionFrame,
    PossessionSource,
    ZonePossessionSource,
    segments_from_stream,
)

__all__ = [
    "ActionConfig",
    "config_from_prep_meta",
    "detect_actions",
    "run_stages",
    "AerialRun",
    "AerialTrack",
    "detect_airborne",
    "PossessionFrame",
    "PossessionSource",
    "ZonePossessionSource",
    "segments_from_stream",
]
