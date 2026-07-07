"""Configuration for ball-free eventing.

Everything the window emitter needs, with documented defaults. Pitch context
(fps, target dimensions) comes from the prerequisites' ``prep_meta.json`` via
:func:`config_from_prep_meta`; thresholds are tunable without re-running Layer 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# Prepared-table columns this layer reads (from src.prerequisites output).
COL_ATTACK_DIR = "attack_dir"
COL_PITCH_X_T = "pitch_x_t_m"
COL_PITCH_Y_T = "pitch_y_t_m"
PEOPLE_ROLES = ("player", "goalkeeper")


@dataclass
class EventConfig:
    """Parameters for the ball-free high-value window emitter.

    A frame is "high value" when an attacking team has committed enough players
    deep into the opponent's final third; contiguous high-value frames become
    windows (with gap-merging and padding). Distances are metres in the target
    pitch frame; times/gaps are frames.
    """

    # --- pitch context (from prep_meta.json) ---
    fps: float = 25.0
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    period_col: Optional[str] = None

    # --- per-frame attacking signal ---
    final_third_frac: float = 2.0 / 3.0  # attacking-normalized x fraction of "deep"
    box_frac: float = 0.83  # ~18-yard-box edge as a fraction of pitch length
    min_players_for_signal: int = 6  # need this many valid team players to trust a frame

    # --- value score -> high-value gate ---
    # value = w_third * (# attackers in final third) + w_box * (# in box)
    #         + w_depth * (deepest attacker's normalized x)
    w_third: float = 1.0
    w_box: float = 2.0
    w_depth: float = 1.0
    value_threshold: float = 4.0  # frame is high-value at/above this

    # --- window assembly ---
    window_merge_gap_frames: int = 25  # bridge high-value runs <= this apart (1 s)
    window_min_frames: int = 12  # discard windows shorter than this
    window_pad_frames: int = 25  # pad each kept window by this on both sides

    overrides: dict = field(default_factory=dict)

    def final_third_x(self) -> float:
        """Attacking-normalized x (metres) at the start of the final third."""
        return self.final_third_frac * self.pitch_length_m

    def box_x(self) -> float:
        """Attacking-normalized x (metres) at the edge of the penalty box."""
        return self.box_frac * self.pitch_length_m

    def as_meta(self) -> dict:
        return asdict(self)


def config_from_prep_meta(prep_meta: dict, **overrides) -> EventConfig:
    """Build an :class:`EventConfig` from a prerequisites ``prep_meta.json``.

    Reads fps and the *target* pitch dims (the frame the prepared coordinates
    live in), then applies any non-None keyword overrides.
    """
    cfg = EventConfig()

    src = prep_meta.get("source_meta", {}) or {}
    cfg.fps = float(src.get("fps", cfg.fps) or cfg.fps)

    steps = prep_meta.get("steps", {}) or {}
    rescale = steps.get("rescale_coords", {}) or {}
    target = rescale.get("target_pitch") or rescale.get("target") or {}
    cfg.pitch_length_m = float(target.get("length_m", cfg.pitch_length_m))
    cfg.pitch_width_m = float(target.get("width_m", cfg.pitch_width_m))

    applied = {}
    for key, value in overrides.items():
        if value is None or not hasattr(cfg, key):
            continue
        setattr(cfg, key, value)
        applied[key] = value
    cfg.overrides = applied
    return cfg
