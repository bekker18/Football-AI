"""Dead-ball / in-play proxy flag."""

from prereq_helpers import make_df, row
from src.prerequisites import PrepConfig, synth_dead_ball
from src.prerequisites.config import BALL_OBJECT_ID, COL_IN_PLAY, COL_IN_PLAY_CONF


def test_out_of_bounds_reads_as_not_in_play():
    rows = [row(f, BALL_OBJECT_ID, "ball", None, 60.0, 35.0) for f in range(5)]
    rows += [row(f, BALL_OBJECT_ID, "ball", None, 200.0, 35.0) for f in range(5, 10)]
    out, meta = synth_dead_ball(make_df(rows), PrepConfig())
    per_frame = out.drop_duplicates("frame").set_index("frame")[COL_IN_PLAY]
    assert bool(per_frame[0]) is True
    assert bool(per_frame[9]) is False
    assert meta["n_frames_ball_oob"] == 5


def test_ball_absence_is_not_a_dead_ball():
    rows = [row(f, BALL_OBJECT_ID, "ball", None, 60.0, 35.0) for f in range(3)]
    # frames 3..5: ball missing (occlusion), only a player present
    rows += [row(f, 1, "player", 0, 50.0, 20.0) for f in range(3, 6)]
    out, _ = synth_dead_ball(make_df(rows), PrepConfig())
    per_frame = out.drop_duplicates("frame").set_index("frame")
    assert per_frame[COL_IN_PLAY].astype(bool).all()  # carried forward as in-play
    # confidence decays across the occlusion
    assert per_frame[COL_IN_PLAY_CONF][5] < per_frame[COL_IN_PLAY_CONF][3]
