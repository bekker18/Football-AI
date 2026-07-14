"""Coordinate rescale: linear map source pitch -> target convention."""

import numpy as np

from prereq_helpers import make_df, row
from src.prerequisites import PrepConfig, rescale_coords


def test_rescale_scales_and_preserves_origin():
    df = make_df([
        row(0, 0, "ball", None, 120.0, 70.0),   # far corner -> target far corner
        row(0, 1, "player", 0, 0.0, 0.0),        # origin stays origin
        row(1, 1, "player", 0, 60.0, 35.0),      # centre stays centre (proportional)
    ])
    cfg = PrepConfig()  # source 120x70, target 105x68
    out, meta = rescale_coords(df, cfg)

    assert out.loc[0, "pitch_x_t_m"] == 105.0
    assert out.loc[0, "pitch_y_t_m"] == 68.0
    assert out.loc[1, "pitch_x_t_m"] == 0.0 and out.loc[1, "pitch_y_t_m"] == 0.0
    assert np.isclose(out.loc[2, "pitch_x_t_m"], 60.0 * 105.0 / 120.0)
    # originals untouched
    assert out.loc[0, "pitch_x_m"] == 120.0
    assert meta["target_pitch_m"] == [105.0, 68.0]


def test_rescale_120x80_and_nan_propagation():
    df = make_df([row(0, 0, "ball", None, None, None, valid=False)])
    cfg = PrepConfig(target_length_m=120.0, target_width_m=80.0)
    out, meta = rescale_coords(df, cfg)
    assert np.isnan(out.loc[0, "pitch_x_t_m"])  # NaN source -> NaN target
    assert meta["scale"] == [1.0, round(80.0 / 70.0, 6)]
