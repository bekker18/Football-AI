"""Track id stabilization: motion-based stitching of fragmented tracks."""

from prereq_helpers import make_df, row
from src.prerequisites import PrepConfig, stitch_ids
from src.prerequisites.config import COL_STABLE_ID


def _fragmented():
    rows = []
    # track A (id 1): frames 0..4 drifting +x at 0.2 m/frame (=> 5 m/s), y=30
    for i in range(5):
        rows.append(row(i, 1, "player", 0, 10.0 + 0.2 * i, 30.0))
    # track B (id 2): resumes at frame 7 where A's velocity would land it
    for i, f in enumerate((7, 8, 9)):
        rows.append(row(f, 2, "player", 0, 11.4 + 0.2 * i, 30.0))
    # decoy C (id 3): other team, far away -> must NOT link to A/B
    for f in (7, 8, 9):
        rows.append(row(f, 3, "player", 1, 100.0, 10.0))
    return make_df(rows)


def test_stitch_links_extrapolated_fragment():
    df = _fragmented()
    out, meta = stitch_ids(df, PrepConfig())
    b_ids = out.loc[out["object_id"] == 2, COL_STABLE_ID].unique().tolist()
    a_ids = out.loc[out["object_id"] == 1, COL_STABLE_ID].unique().tolist()
    assert b_ids == a_ids == [1]      # B folded into A's stable id
    assert meta["n_links"] == 1
    assert out.loc[out["object_id"] == 3, COL_STABLE_ID].tolist() == [3, 3, 3]


def test_stitch_respects_gap_limit():
    df = _fragmented()
    # gap between A (ends f4) and B (starts f7) is 3; forbid gaps > 2
    out, meta = stitch_ids(df, PrepConfig(stitch_max_gap_frames=2))
    assert meta["n_links"] == 0
    assert out.loc[out["object_id"] == 2, COL_STABLE_ID].unique().tolist() == [2]


def test_stitch_respects_team_mismatch():
    df = _fragmented()
    # flip B to the other team -> teams no longer match -> no link
    df.loc[df["object_id"] == 2, "team"] = 1.0
    out, meta = stitch_ids(df, PrepConfig())
    assert meta["n_links"] == 0


def test_ball_and_untracked_keep_identity():
    df = make_df([
        row(0, 0, "ball", None, 60.0, 35.0),
        row(0, -1, "player", 0, 50.0, 20.0),
    ])
    out, _ = stitch_ids(df, PrepConfig())
    assert out.loc[out["object_id"] == 0, COL_STABLE_ID].tolist() == [0]
    assert out.loc[out["object_id"] == -1, COL_STABLE_ID].tolist() == [-1]
