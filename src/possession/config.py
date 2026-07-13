"""Configuration for the possession-zone detector.

Every tunable lives here with a documented default. Pitch/fps context comes from
the prerequisites' ``prep_meta.json`` via :func:`config_from_prep_meta`; the only
real knob is :attr:`PossessionConfig.r_pz_m`, the possession-zone radius.

Coordinate-frame contract
-------------------------
This layer reads the **TARGET** frame (105x68 m by default) only:

- players / goalkeepers -> ``pitch_x_t_m`` / ``pitch_y_t_m``
- ball (smoothed AND rescaled) -> ``ball_x_ts_m`` / ``ball_y_ts_m``

The source-frame columns (``pitch_x_m`` / ``ball_x_s_m``) must never be mixed in:
the source->target rescale is *anisotropic* (x*0.875, y*0.971 for 120x70 ->
105x68), so a distance computed across the two frames is silently wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# Reserved object_id for the ball in Layer 1 output (see src/config.py). Defined
# locally so importing this package never pulls in the CV stack.
BALL_OBJECT_ID = 0

# Roles that may hold the ball. Referees are never possession candidates.
PEOPLE_ROLES = ("player", "goalkeeper")

# --- prepared-table columns this layer READS (from src.prerequisites) --- #
COL_STABLE_ID = "stable_id"  # possessor identity (NOT the raw object_id)
COL_PITCH_X_T = "pitch_x_t_m"  # player x, target frame
COL_PITCH_Y_T = "pitch_y_t_m"  # player y, target frame
COL_BALL_XTS = "ball_x_ts_m"  # ball x, smoothed + target frame
COL_BALL_YTS = "ball_y_ts_m"  # ball y, smoothed + target frame

# --- the four per-frame possession states --- #
STATE_NO_BALL = "no_ball"  # no usable smoothed ball position (occlusion)
STATE_LOOSE = "loose"  # ball present, nobody within R_pz
STATE_POSSESSION = "possession"  # exactly one candidate within R_pz
STATE_CONTESTED = "contested"  # 2+ candidates within R_pz; nearest wins, flagged
STATES = (STATE_NO_BALL, STATE_LOOSE, STATE_POSSESSION, STATE_CONTESTED)

# States that carry a possessor. no_ball/loose NEVER do.
ATTRIBUTED_STATES = (STATE_POSSESSION, STATE_CONTESTED)


@dataclass
class PossessionConfig:
    """Parameters for the possession-zone detector.

    A frame is attributed to a player when that player is the only candidate
    within :attr:`r_pz_m` metres of the ball (``possession``), or the nearest of
    several (``contested``). Distances are metres in the **target** pitch frame.
    """

    # --- context (from prep_meta.json) ---
    fps: float = 25.0
    pitch_length_m: float = 105.0  # target frame; documents where r_pz_m lives
    pitch_width_m: float = 68.0
    period_col: Optional[str] = None

    # --- the possession zone ---
    # Calibrated on clip 2e57b9_0: nearest player-to-ball distance is median
    # 1.86 m / p75 3.93 m; duel rate stays <2% up to 3.0 m then accelerates
    # (4.7% at 4 m, 12.8% at 5 m) while clean attribution holds >98% to 3.0 m.
    #
    # CAVEAT: that clip is a single open-play attacking phase with no congested
    # box and no set-pieces, so 3.0 m is an UPPER BOUND. Crowded footage
    # (corners, goalmouth scrambles) will duel far more at this radius -- re-run
    # the sweep mode (`python -m src.possession sweep_radii`) on new footage and
    # lower this before freezing it.
    r_pz_m: float = 3.0

    # bookkeeping: fields overridden on the CLI (for the manifest)
    overrides: dict = field(default_factory=dict)

    def as_meta(self) -> dict:
        """Serializable snapshot of every parameter (for possession_meta.json)."""
        return asdict(self)


def config_from_prep_meta(prep_meta: dict, **overrides) -> PossessionConfig:
    """Build a :class:`PossessionConfig` from a prerequisites ``prep_meta.json``.

    Reads fps and the *target* pitch dims (the frame the prepared coordinates
    live in), then applies any non-None keyword overrides.
    """
    cfg = PossessionConfig()

    src = prep_meta.get("source_meta", {}) or {}
    cfg.fps = float(src.get("fps", cfg.fps) or cfg.fps)

    steps = prep_meta.get("steps", {}) or {}
    rescale = steps.get("rescale_coords", {}) or {}
    target = rescale.get("target_pitch_m") or rescale.get("target_pitch") or None
    if isinstance(target, (list, tuple)) and len(target) == 2:
        cfg.pitch_length_m, cfg.pitch_width_m = float(target[0]), float(target[1])
    elif isinstance(target, dict):
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
