"""Ball smoothing: outlier flagging, short-gap interpolation, kinematics."""

import numpy as np
import pytest

from prereq_helpers import make_df, row
from src.prerequisites import PrepConfig, savgol_smooth, smooth_ball
from src.prerequisites.config import (
    BALL_OBJECT_ID,
    COL_BALL_INTERP,
    COL_BALL_OUTLIER,
    COL_BALL_SPEED,
    COL_BALL_XS,
    COL_SYNTHETIC,
)


def _ball_line(frames, xs):
    return make_df([row(f, BALL_OBJECT_ID, "ball", None, x, 30.0) for f, x in zip(frames, xs)])


def test_flags_isolated_spike_only():
    frames = list(range(10))
    xs = [float(f) for f in frames]
    xs[5] = 500.0  # single-frame impossible jump
    out, meta = smooth_ball(_ball_line(frames, xs), PrepConfig())
    ball = out[out["object_id"] == BALL_OBJECT_ID].sort_values("frame")
    flags = ball.set_index("frame")[COL_BALL_OUTLIER]
    assert bool(flags[5]) is True
    assert flags.drop(index=5).fillna(False).eq(False).all()
    assert meta["n_outliers"] == 1
    # spike frame keeps its raw value but gets a plausible smoothed replacement
    assert out.loc[(out["frame"] == 5) & (out["object_id"] == 0), "pitch_x_m"].iloc[0] == 500.0
    assert abs(ball.set_index("frame")[COL_BALL_XS][5] - 5.0) < 3.0


def test_interpolates_short_gap_with_synthetic_rows():
    out, meta = smooth_ball(_ball_line([0, 1, 2, 5, 6], [0, 1, 2, 5, 6]), PrepConfig())
    assert meta["n_synthetic_rows"] == 2  # frames 3 and 4 filled
    synth = out[out[COL_SYNTHETIC] == True]  # noqa: E712
    assert sorted(synth["frame"].tolist()) == [3, 4]
    assert synth[COL_BALL_INTERP].astype(bool).all()
    assert (synth["role"] == "ball").all()


def test_leaves_long_gap_missing():
    out, meta = smooth_ball(_ball_line([0, 1, 2, 20], [0, 1, 2, 20]), PrepConfig())
    assert meta["n_synthetic_rows"] == 0
    assert out[out[COL_SYNTHETIC] == True].empty  # noqa: E712


def test_speed_recomputed_from_smoothed_track():
    # steady 1 m/frame @ 25 fps => 25 m/s
    out, _ = smooth_ball(_ball_line(list(range(12)), list(range(12))), PrepConfig())
    ball = out[out["object_id"] == BALL_OBJECT_ID]
    mid = ball[(ball["frame"] > 2) & (ball["frame"] < 9)]
    assert np.allclose(mid[COL_BALL_SPEED].to_numpy(), 25.0, atol=1.0)


def test_savgol_smooth_reduces_noise_and_handles_short_input():
    rng = np.random.default_rng(0)
    clean = np.linspace(0, 10, 40)
    noisy = clean + rng.normal(0, 0.3, size=clean.size)
    sm = savgol_smooth(noisy, 7, 2)
    assert np.mean((sm - clean) ** 2) < np.mean((noisy - clean) ** 2)
    # too-short segment: falls back to a polynomial fit, never raises
    assert len(savgol_smooth(np.array([1.0, 2.0]), 7, 2)) == 2
