"""Tests for the pure image->pitch conversion (numpy only, no CV stack)."""

import numpy as np

from src.geometry import to_pitch_m


class _FakeTransformer:
    """Stand-in homography: scales pixel coords by 10 to land in cm-space."""

    def transform_points(self, pts):
        return np.asarray(pts, dtype=np.float32) * 10.0


def test_to_pitch_m_without_homography():
    assert to_pitch_m(None, 100.0, 200.0) == (None, None, False)


def test_to_pitch_m_converts_cm_to_m():
    x, y, ok = to_pitch_m(_FakeTransformer(), 50.0, 70.0)
    assert ok is True
    # 50 px * 10 = 500 cm = 5.0 m ; 70 px * 10 = 700 cm = 7.0 m
    assert x == 5.0
    assert y == 7.0
