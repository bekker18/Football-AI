"""Team attacking-direction resolution + attacking-normalized coordinates."""

import numpy as np

from prereq_helpers import make_df, row
from src.prerequisites import PrepConfig, normalize_to_attack, resolve_direction
from src.prerequisites.config import COL_ATTACK_DIR


def _with_gks(gk0_x, gk1_x, n=6):
    rows = []
    for i in range(n):
        rows.append(row(i, 10, "goalkeeper", 0, gk0_x, 35.0))
        rows.append(row(i, 20, "goalkeeper", 1, gk1_x, 35.0))
    return make_df(rows)


def test_direction_from_gk_medians():
    df = _with_gks(gk0_x=12.7, gk1_x=84.0)  # team0 GK deep left, team1 deep right
    out, meta = resolve_direction(df, PrepConfig())
    d0 = out.loc[out["team"] == 0, COL_ATTACK_DIR].unique().tolist()
    d1 = out.loc[out["team"] == 1, COL_ATTACK_DIR].unique().tolist()
    assert d0 == [1.0]    # GK low x -> attacks +x
    assert d1 == [-1.0]   # GK high x -> attacks -x
    assert meta["periods"]["None"]["method"] == "gk_median"


def test_direction_sparse_gk_falls_back_to_opposite():
    # only team0 GK has enough frames; team1 gets the opposite direction
    rows = []
    for i in range(6):
        rows.append(row(i, 10, "goalkeeper", 0, 12.0, 35.0))
    rows.append(row(0, 20, "goalkeeper", 1, 80.0, 35.0))  # 1 frame => sparse
    out, meta = resolve_direction(make_df(rows), PrepConfig())
    d0 = out.loc[out["team"] == 0, COL_ATTACK_DIR].dropna().unique().tolist()
    d1 = out.loc[out["team"] == 1, COL_ATTACK_DIR].dropna().unique().tolist()
    assert d0 == [1.0] and d1 == [-1.0]
    assert meta["periods"]["None"]["method"] == "gk_single"


def test_direction_unresolved_when_no_evidence():
    df = make_df([row(0, 0, "ball", None, 60.0, 35.0)])
    out, meta = resolve_direction(df, PrepConfig())
    assert out[COL_ATTACK_DIR].isna().all()
    assert meta["periods"]["None"]["method"] == "unresolved"


def test_normalize_to_attack_flips_only_negative_dir():
    x = np.array([10.0, 10.0, 10.0])
    d = np.array([1.0, -1.0, np.nan])
    out = normalize_to_attack(x, d, pitch_length=120.0)
    assert out[0] == 10.0        # +1 keeps
    assert out[1] == 110.0       # -1 mirrors to L - x
    assert out[2] == 10.0        # NaN passes through
