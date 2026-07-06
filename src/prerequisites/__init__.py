"""Prerequisites: raw game state -> event-ready game state.

The stage between Layer 1 (CV extraction) and event detection / valuation.
Composable, non-destructive transforms that clean and enrich the Layer 1 output.
Each transform adds columns (or emits metadata) and never silently overwrites
originals; each is importable and usable on its own, or together via
:func:`run_prerequisites`.

    from src.prerequisites import (
        PrepConfig, config_from_meta, load_gamestate, write_prepared,
        stitch_ids, resolve_direction, rescale_coords, smooth_ball,
        synth_dead_ball, run_prerequisites, normalize_to_attack,
    )

Only depends on pandas / numpy / scipy — not the Layer 1 CV stack.
"""

from __future__ import annotations

from .ball import savgol_smooth, smooth_ball
from .config import PrepConfig, config_from_meta
from .deadball import synth_dead_ball
from .direction import attacking_frame, normalize_to_attack, resolve_direction
from .io import frames_from_df, load_gamestate, write_prepared
from .pipeline import run_prerequisites
from .rescale import rescale_coords
from .stitch import global_reid_todo, stitch_ids

__all__ = [
    "PrepConfig",
    "config_from_meta",
    "load_gamestate",
    "write_prepared",
    "frames_from_df",
    "stitch_ids",
    "global_reid_todo",
    "resolve_direction",
    "normalize_to_attack",
    "attacking_frame",
    "rescale_coords",
    "smooth_ball",
    "savgol_smooth",
    "synth_dead_ball",
    "run_prerequisites",
]
