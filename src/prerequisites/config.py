"""Configuration for the prerequisites transforms.

Every tunable threshold lives here with a documented default. The CLI populates
a :class:`PrepConfig` from ``meta.json`` (fps, source pitch dims, pitch stride)
plus any command-line overrides, and each transform reads only from this object
— nothing is hardcoded at the call site.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# Reserved object_id for the ball in Layer 1 output (see src/config.py). Defined
# locally so importing this package never pulls in the CV stack.
BALL_OBJECT_ID = 0

# Roles that represent trackable people (as opposed to ball / referee-only ops).
PEOPLE_ROLES = ("player", "goalkeeper")

# --- names of the columns each transform ADDS (never overwrites originals) --- #
COL_STABLE_ID = "stable_id"
COL_ATTACK_DIR = "attack_dir"
COL_PITCH_X_T = "pitch_x_t_m"
COL_PITCH_Y_T = "pitch_y_t_m"
COL_BALL_OUTLIER = "ball_outlier"
COL_BALL_INTERP = "ball_interp"
COL_SYNTHETIC = "synthetic"
COL_BALL_XS = "ball_x_s_m"
COL_BALL_YS = "ball_y_s_m"
COL_BALL_VX = "ball_vx_ms"
COL_BALL_VY = "ball_vy_ms"
COL_BALL_SPEED = "ball_speed_ms"
COL_BALL_ACCEL = "ball_accel_ms2"
COL_IN_PLAY = "in_play"
COL_IN_PLAY_CONF = "in_play_conf"

# Named target-pitch conventions selectable on the CLI (length_m, width_m).
TARGET_PITCH_PRESETS = {
    "105x68": (105.0, 68.0),
    "120x80": (120.0, 80.0),
}


@dataclass
class PrepConfig:
    """All parameters used by the prerequisite transforms.

    Grouped by transform. Distances are metres, speeds metres/second, gaps in
    frames. ``fps``, ``source_length_m``, ``source_width_m`` and ``pitch_stride``
    are normally filled from ``meta.json`` via :func:`config_from_meta`.
    """

    # --- context (from meta.json) ---
    fps: float = 25.0
    source_length_m: float = 120.0
    source_width_m: float = 70.0
    pitch_stride: int = 1
    period_col: Optional[str] = None  # None => whole clip is one period

    # --- 1. track id stabilization ---
    stitch_max_gap_frames: int = 25  # link only across gaps <= this (1 s @ 25fps)
    stitch_max_dist_m: float = 5.0  # extrapolated-vs-actual position tolerance
    stitch_vel_window: int = 5  # frames used to estimate a track's end velocity

    # --- 2. team normalization + attacking direction ---
    min_gk_frames: int = 5  # a team's GK is "sparse" below this many valid frames

    # --- 3. coordinate rescale ---
    target_length_m: float = 105.0
    target_width_m: float = 68.0

    # --- 4. ball smoothing + outlier rejection ---
    ball_max_speed_ms: float = 36.0  # speed gate for impossible ball steps
    ball_max_interp_gap: int = 5  # interpolate missing runs no longer than this
    ball_savgol_window: int = 7  # Savitzky-Golay window (odd), @ 25 fps
    ball_savgol_order: int = 2  # Savitzky-Golay polynomial order

    # --- 5. dead-ball / in-play flag ---
    oob_margin_m: float = 2.0  # ball must be > this beyond a line to read as OOB
    still_speed_ms: float = 0.5  # below this counts as "stationary"
    still_frames: int = 12  # stationary run this long near a line => dead
    near_boundary_m: float = 3.0  # "near a line" tolerance
    absent_conf_decay: float = 0.9  # per-frame confidence decay while ball absent
    absent_conf_floor: float = 0.2  # confidence never decays below this

    # bookkeeping: fields that were overridden on the CLI (for the manifest)
    overrides: dict = field(default_factory=dict)

    def as_meta(self) -> dict:
        """Serializable snapshot of every parameter (for prep_meta.json)."""
        d = asdict(self)
        return d


def config_from_meta(meta: dict, **overrides) -> PrepConfig:
    """Build a :class:`PrepConfig` from a Layer 1 ``meta.json`` dict.

    Reads fps and the *source* pitch dimensions from ``meta`` (never hardcoded),
    then applies any keyword ``overrides`` (typically parsed CLI args). Unknown
    or ``None`` overrides are ignored so callers can pass a sparse dict.
    """
    cfg = PrepConfig()
    cfg.fps = float(meta.get("fps", cfg.fps) or cfg.fps)
    pitch = meta.get("pitch", {}) or {}
    cfg.source_length_m = float(pitch.get("length_m", cfg.source_length_m))
    cfg.source_width_m = float(pitch.get("width_m", cfg.source_width_m))
    perf = meta.get("perf", {}) or {}
    cfg.pitch_stride = int(perf.get("pitch_stride", cfg.pitch_stride) or cfg.pitch_stride)

    applied = {}
    for key, value in overrides.items():
        if value is None or not hasattr(cfg, key):
            continue
        setattr(cfg, key, value)
        applied[key] = value
    cfg.overrides = applied
    return cfg
