"""Ball-free eventing — tracking-native surfaces derived from positions alone.

The first, load-bearing piece is the **high-value window emitter**: a cheap pass
over the whole match (player/GK positions only, no ball) that flags the frame
ranges worth spending the expensive ball detector on. That stream of windows is
exactly what the two-pass controller (see docs/strategy.md) gates on, so it is
built first — the rest of the ball-free surfaces (pressing, formations, pitch
control) hang off the same per-frame signals later.

    from src.events import EventConfig, detect_high_value_windows

Consumes the *prepared* tracking table (needs ``attack_dir`` + target-frame
coordinates from ``src.prerequisites``). Depends only on pandas / numpy — not the
Layer 1 CV stack.
"""

from __future__ import annotations

from .config import EventConfig, config_from_prep_meta
from .pipeline import detect_high_value_windows
from .signals import attacking_signals
from .windows import windows_from_signals

__all__ = [
    "EventConfig",
    "config_from_prep_meta",
    "attacking_signals",
    "windows_from_signals",
    "detect_high_value_windows",
]
