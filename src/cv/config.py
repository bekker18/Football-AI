"""Shared constants and pitch geometry.

Pitch coordinate system: x in [0, 120] m (length), y in [0, 70] m (width),
origin at the top-left pitch corner. Matches SoccerPitchConfiguration (cm/100).
"""

from __future__ import annotations

import numpy as np
from sports.configs.soccer import SoccerPitchConfiguration

# Class ids in the player-detection model.
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Reserved object_id for the ball. ByteTrack ids for people are >= 1 and
# untracked people fall back to -1, so 0 collides with neither. Keeps the
# object_id column a clean integer (no nulls) and lets you select the ball's
# whole trajectory with a single `object_id == 0`.
BALL_OBJECT_ID = 0

# Palette index used for a player whose team hasn't been predicted yet (see
# annotate.VideoAnnotator). Kept here so the sentinel and the colour agree.
UNKNOWN_COLOR_IDX = 3

PITCH = SoccerPitchConfiguration()
PITCH_VERTICES = np.array(PITCH.vertices)  # (N, 2) in cm
PITCH_LEN_M = PITCH.length / 100.0
PITCH_WID_M = PITCH.width / 100.0
