"""Possession review overlay: the pure geometry (no cv2, no video).

Only the drawing calls need cv2; the geometry that decides *where* things land —
and therefore whether the R_pz circle on the minimap is honest — is pure numpy
and is tested here.
"""

import numpy as np
import pytest

from src.possession import PossessionConfig
from src.possession.config import (
    STATE_CONTESTED,
    STATE_LOOSE,
    STATE_NO_BALL,
    STATE_POSSESSION,
)
from src.possession.review import (
    STATE_BGR,
    UNKNOWN_BGR,
    minimap_geometry,
    pitch_to_px,
    state_strip,
    team_color,
)


def test_minimap_scale_is_uniform_so_the_zone_stays_a_circle():
    """An anisotropic minimap would draw R_pz as an ellipse and lie about it."""
    cfg = PossessionConfig(pitch_length_m=105.0, pitch_width_m=68.0)
    scale, w, h, pad = minimap_geometry(cfg, width=460, pad=16)

    # one scale for both axes: a metre is the same number of pixels either way
    span_x = pitch_to_px(cfg.pitch_length_m, 0, scale, pad)[0] - pitch_to_px(0, 0, scale, pad)[0]
    span_y = pitch_to_px(0, cfg.pitch_width_m, scale, pad)[1] - pitch_to_px(0, 0, scale, pad)[1]
    assert span_x / cfg.pitch_length_m == pytest.approx(span_y / cfg.pitch_width_m, rel=1e-3)

    # the pitch fits the canvas with the padding it asked for
    assert span_x == pytest.approx(w - 2 * pad, abs=1)
    assert h == pytest.approx(span_y + 2 * pad, abs=1)


def test_pitch_origin_maps_to_the_padded_corner():
    cfg = PossessionConfig()
    scale, w, h, pad = minimap_geometry(cfg)
    assert pitch_to_px(0.0, 0.0, scale, pad) == (pad, pad)
    # the far corner lands inside the canvas
    fx, fy = pitch_to_px(cfg.pitch_length_m, cfg.pitch_width_m, scale, pad)
    assert fx <= w and fy <= h


def test_state_strip_colors_every_frame_by_state():
    states = np.array([STATE_NO_BALL, STATE_LOOSE, STATE_POSSESSION, STATE_CONTESTED])
    strip = state_strip(states, width=4, height=3)
    assert strip.shape == (3, 4, 3)
    for i, s in enumerate(states):
        assert tuple(strip[0, i]) == STATE_BGR[s]


def test_state_strip_resamples_to_the_requested_width():
    """A 90-minute match and a 30 s clip both have to fit the same strip."""
    states = np.array([STATE_POSSESSION] * 500 + [STATE_LOOSE] * 500)
    strip = state_strip(states, width=100, height=2)
    assert strip.shape == (2, 100, 3)
    assert tuple(strip[0, 0]) == STATE_BGR[STATE_POSSESSION]   # first half
    assert tuple(strip[0, -1]) == STATE_BGR[STATE_LOOSE]       # second half


def test_state_strip_handles_an_empty_clip():
    strip = state_strip(np.array([]), width=10, height=3)
    assert strip.shape == (3, 10, 3) and strip.sum() == 0


def test_the_four_states_are_visually_distinct():
    colors = [STATE_BGR[s] for s in
              (STATE_NO_BALL, STATE_LOOSE, STATE_POSSESSION, STATE_CONTESTED)]
    assert len(set(colors)) == 4


def test_team_color_tolerates_a_null_team():
    assert team_color(0.0) != team_color(1.0)
    assert team_color(float("nan")) == UNKNOWN_BGR
    assert team_color(None) == UNKNOWN_BGR


def test_review_module_imports_without_cv2():
    """Importing the overlay must not drag in the CV stack; only rendering does."""
    import src.possession.review as review
    assert hasattr(review, "render_review")
    # the package __init__ must stay cv2-free
    import src.possession as pkg
    assert "review" not in pkg.__all__
