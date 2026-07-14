"""Possession zone: per-frame state classification + segment collapsing.

The synthetic fixtures pin the four states one frame at a time, so a regression
in the state machine names the state it broke.
"""

import numpy as np
import pandas as pd
import pytest

from src.possession import (
    PossessionConfig,
    detect_possession,
    possession_frames,
    possession_segments,
    radius_grid,
    sweep_radii,
)
from src.possession.config import (
    STATE_CONTESTED,
    STATE_LOOSE,
    STATE_NO_BALL,
    STATE_POSSESSION,
)

# the columns the possession layer reads out of the prepared table
PREP_COLS = ["frame", "time_s", "object_id", "stable_id", "role", "team",
             "pitch_x_t_m", "pitch_y_t_m", "ball_x_ts_m", "ball_y_ts_m"]

FPS = 25.0


def player_row(frame, stable_id, x, y, team=0.0, role="player"):
    """One candidate (player/goalkeeper/referee) at a target-frame position."""
    return dict(
        frame=frame, time_s=round(frame / FPS, 4), object_id=int(stable_id),
        stable_id=float(stable_id), role=role, team=team,
        pitch_x_t_m=float(x), pitch_y_t_m=float(y),
        ball_x_ts_m=np.nan, ball_y_ts_m=np.nan,  # ball cols are null off the ball row
    )


def ball_row(frame, x, y):
    """The ball row (object_id 0). ``x=None`` => no usable smoothed position."""
    return dict(
        frame=frame, time_s=round(frame / FPS, 4), object_id=0,
        stable_id=0.0, role="ball", team=np.nan,
        pitch_x_t_m=np.nan, pitch_y_t_m=np.nan,
        ball_x_ts_m=np.nan if x is None else float(x),
        ball_y_ts_m=np.nan if y is None else float(y),
    )


def make_prepared(rows):
    return pd.DataFrame(rows, columns=PREP_COLS)


# --------------------------------------------------------------------------- #
# the four states, one frame each
# --------------------------------------------------------------------------- #

def test_single_candidate_in_zone_is_possession():
    cfg = PossessionConfig(r_pz_m=3.0)
    df = make_prepared([
        ball_row(0, 50.0, 34.0),
        player_row(0, 7, 51.0, 34.0),      # 1.0 m away -> in the zone
        player_row(0, 8, 60.0, 34.0, team=1.0),  # 10 m away -> outside
    ])
    f = possession_frames(df, cfg).iloc[0]
    assert f["state"] == STATE_POSSESSION
    assert f["possessor_id"] == 7
    assert f["possessor_team"] == 0.0
    assert f["n_in_zone"] == 1
    assert f["dist_m"] == pytest.approx(1.0)


def test_ball_far_from_everyone_is_loose_with_no_possessor():
    cfg = PossessionConfig(r_pz_m=3.0)
    df = make_prepared([
        ball_row(0, 50.0, 34.0),
        player_row(0, 7, 60.0, 34.0),   # 10 m
        player_row(0, 8, 40.0, 34.0),   # 10 m
    ])
    f = possession_frames(df, cfg).iloc[0]
    assert f["state"] == STATE_LOOSE
    assert pd.isna(f["possessor_id"])          # never fabricate a possessor
    assert pd.isna(f["possessor_team"])
    assert f["n_in_zone"] == 0
    assert f["dist_m"] == pytest.approx(10.0)  # still reports HOW loose


def test_two_candidates_in_zone_is_contested_and_nearest_wins():
    cfg = PossessionConfig(r_pz_m=3.0)
    df = make_prepared([
        ball_row(0, 50.0, 34.0),
        player_row(0, 7, 52.0, 34.0, team=0.0),  # 2.0 m
        player_row(0, 8, 51.0, 34.0, team=1.0),  # 1.0 m -> nearest
    ])
    f = possession_frames(df, cfg).iloc[0]
    assert f["state"] == STATE_CONTESTED
    assert f["possessor_id"] == 8               # nearest, but flagged contested
    assert f["possessor_team"] == 1.0
    assert f["n_in_zone"] == 2
    assert f["dist_m"] == pytest.approx(1.0)


def test_missing_ball_position_is_no_ball_not_a_stoppage():
    cfg = PossessionConfig(r_pz_m=3.0)
    df = make_prepared([
        ball_row(0, None, None),           # occluded: no usable smoothed ball
        player_row(0, 7, 50.0, 34.0),      # sitting right where the ball was
    ])
    f = possession_frames(df, cfg).iloc[0]
    assert f["state"] == STATE_NO_BALL
    assert pd.isna(f["possessor_id"])
    assert pd.isna(f["dist_m"])
    assert f["n_in_zone"] == 0


def test_absent_ball_row_is_also_no_ball():
    """A frame with no ball row at all (not just a null one) must not crash."""
    cfg = PossessionConfig(r_pz_m=3.0)
    df = make_prepared([player_row(0, 7, 50.0, 34.0)])
    f = possession_frames(df, cfg).iloc[0]
    assert f["state"] == STATE_NO_BALL
    assert pd.isna(f["possessor_id"])


# --------------------------------------------------------------------------- #
# edges: nothing here may crash
# --------------------------------------------------------------------------- #

def test_referees_are_not_possession_candidates():
    cfg = PossessionConfig(r_pz_m=3.0)
    df = make_prepared([
        ball_row(0, 50.0, 34.0),
        player_row(0, 9, 50.5, 34.0, role="referee"),  # right on the ball
        player_row(0, 7, 60.0, 34.0),                  # the only real candidate
    ])
    f = possession_frames(df, cfg).iloc[0]
    assert f["state"] == STATE_LOOSE      # the ref does not hold the ball
    assert pd.isna(f["possessor_id"])


def test_goalkeepers_are_candidates():
    cfg = PossessionConfig(r_pz_m=3.0)
    df = make_prepared([
        ball_row(0, 5.0, 34.0),
        player_row(0, 1, 5.5, 34.0, role="goalkeeper", team=1.0),
    ])
    f = possession_frames(df, cfg).iloc[0]
    assert f["state"] == STATE_POSSESSION
    assert f["possessor_id"] == 1


def test_frame_with_ball_but_zero_candidates_is_loose():
    cfg = PossessionConfig(r_pz_m=3.0)
    df = make_prepared([ball_row(0, 50.0, 34.0)])
    f = possession_frames(df, cfg).iloc[0]
    assert f["state"] == STATE_LOOSE
    assert pd.isna(f["possessor_id"])
    assert pd.isna(f["dist_m"])       # nothing to measure against
    assert f["n_in_zone"] == 0


def test_null_player_coordinates_are_ignored_not_crashed():
    cfg = PossessionConfig(r_pz_m=3.0)
    df = make_prepared([
        ball_row(0, 50.0, 34.0),
        player_row(0, 7, np.nan, np.nan),   # no homography for this row
        player_row(0, 8, 51.0, 34.0),
    ])
    f = possession_frames(df, cfg).iloc[0]
    assert f["state"] == STATE_POSSESSION
    assert f["possessor_id"] == 8


def test_radius_is_configurable():
    """The same frame flips state as R_pz moves across the player's distance."""
    df = make_prepared([
        ball_row(0, 50.0, 34.0),
        player_row(0, 7, 54.0, 34.0),   # exactly 4.0 m away
    ])
    tight = possession_frames(df, PossessionConfig(r_pz_m=3.0)).iloc[0]
    wide = possession_frames(df, PossessionConfig(r_pz_m=5.0)).iloc[0]
    assert tight["state"] == STATE_LOOSE
    assert wide["state"] == STATE_POSSESSION and wide["possessor_id"] == 7


def test_empty_table_produces_empty_output():
    cfg = PossessionConfig()
    frames, segments, meta = detect_possession(make_prepared([]), cfg)
    assert len(frames) == 0 and len(segments) == 0
    assert meta["n_frames"] == 0 and meta["coverage_pct"] == 0.0


def test_missing_prepared_columns_raise_a_useful_error():
    df = pd.DataFrame({"frame": [0], "object_id": [0], "role": ["ball"]})
    with pytest.raises(KeyError, match="prerequisites"):
        possession_frames(df, PossessionConfig())


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #

def _run(frames_spec, r_pz=3.0):
    """Build a clip from [(frame, ball_xy | None, [(id, x, team), ...]), ...]."""
    rows = []
    for frame, ball_xy, players in frames_spec:
        bx, by = ball_xy if ball_xy else (None, None)
        rows.append(ball_row(frame, bx, by))
        for sid, x, team in players:
            rows.append(player_row(frame, sid, x, 34.0, team=team))
    return detect_possession(make_prepared(rows), PossessionConfig(r_pz_m=r_pz))


def test_segments_collapse_a_run_of_the_same_possessor():
    # player 7 holds the ball for frames 0-4
    spec = [(f, (50.0, 34.0), [(7, 50.5, 0.0), (8, 70.0, 1.0)]) for f in range(5)]
    _, segments, _ = _run(spec)
    assert len(segments) == 1
    s = segments.iloc[0]
    assert s["possessor_id"] == 7 and s["team"] == 0.0
    assert s["start_frame"] == 0 and s["end_frame"] == 4
    assert s["n_frames"] == 5
    assert s["n_contested"] == 0


def test_a_loose_frame_breaks_a_segment():
    # 7 holds (0,1), ball goes loose (2), 7 holds again (3,4) -> TWO touches,
    # because bridging a hold across a loose ball is event logic, not a primitive
    spec = []
    for f in range(5):
        ball_x = 50.0 if f != 2 else 80.0   # frame 2: ball far from everyone
        spec.append((f, (ball_x, 34.0), [(7, 50.5, 0.0)]))
    frames, segments, _ = _run(spec)
    assert frames.loc[frames["frame"] == 2, "state"].iloc[0] == STATE_LOOSE
    assert len(segments) == 2
    assert segments["n_frames"].tolist() == [2, 2]
    assert segments["possessor_id"].tolist() == [7, 7]


def test_possession_change_starts_a_new_segment():
    spec = []
    for f in range(6):
        holder = (7, 50.5, 0.0) if f < 3 else (8, 50.5, 1.0)
        spec.append((f, (50.0, 34.0), [holder]))
    _, segments, meta = _run(spec)
    assert segments["possessor_id"].tolist() == [7, 8]
    assert segments["start_frame"].tolist() == [0, 3]
    assert segments["end_frame"].tolist() == [2, 5]
    assert meta["team_possession_pct"] == {"0": 50.0, "1": 50.0}


def test_contested_frames_extend_a_segment_and_are_counted():
    # 7 is nearest throughout; 8 steps into the zone for frames 2-3
    spec = []
    for f in range(5):
        players = [(7, 50.5, 0.0)]
        if f in (2, 3):
            players.append((8, 51.5, 1.0))   # inside R_pz, but farther than 7
        spec.append((f, (50.0, 34.0), players))
    frames, segments, meta = _run(spec)
    assert (frames["state"] == STATE_CONTESTED).sum() == 2
    assert len(segments) == 1                 # one unbroken touch by 7
    s = segments.iloc[0]
    assert s["possessor_id"] == 7 and s["n_frames"] == 5 and s["n_contested"] == 2
    assert meta["duel_pct"] == 40.0 and meta["clean_pct"] == 60.0


def test_no_ball_frames_never_appear_in_any_segment():
    spec = []
    for f in range(5):
        ball = None if f == 2 else (50.0, 34.0)   # frame 2 occluded
        spec.append((f, ball, [(7, 50.5, 0.0)]))
    frames, segments, meta = _run(spec)
    assert frames.loc[frames["frame"] == 2, "state"].iloc[0] == STATE_NO_BALL
    covered = set()
    for s in segments.itertuples():
        covered |= set(range(int(s.start_frame), int(s.end_frame) + 1))
    assert 2 not in covered
    assert meta["n_ball_frames"] == 4          # the occluded frame is not a ball frame
    assert meta["coverage_pct"] == 100.0       # ...and so does not hurt coverage


def test_no_possessor_is_ever_assigned_on_loose_or_no_ball_frames():
    spec = [
        (0, (50.0, 34.0), [(7, 50.5, 0.0)]),   # possession
        (1, (80.0, 34.0), [(7, 50.5, 0.0)]),   # loose
        (2, None, [(7, 50.5, 0.0)]),           # no_ball
    ]
    frames, _, _ = _run(spec)
    unattributed = frames[frames["state"].isin([STATE_LOOSE, STATE_NO_BALL])]
    assert len(unattributed) == 2
    assert unattributed["possessor_id"].isna().all()
    assert unattributed["possessor_team"].isna().all()


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #

def test_radius_grid_is_inclusive():
    assert radius_grid(1.0, 3.0, 0.5) == [1.0, 1.5, 2.0, 2.5, 3.0]
    with pytest.raises(ValueError):
        radius_grid(1.0, 3.0, 0.0)


def test_sweep_coverage_is_monotonic_in_the_radius():
    """A bigger zone can only ever attribute more frames, never fewer."""
    spec = []
    for f in range(10):
        spec.append((f, (50.0, 34.0), [(7, 50.0 + f * 0.6, 0.0), (8, 70.0, 1.0)]))
    df = make_prepared([
        r for frame, ball, players in spec
        for r in ([ball_row(frame, *ball)]
                  + [player_row(frame, sid, x, 34.0, team=t) for sid, x, t in players])
    ])
    sweep = sweep_radii(df, PossessionConfig(), radius_grid(1.0, 5.0, 1.0))
    assert len(sweep) == 5
    cov = sweep["coverage_pct"].tolist()
    assert cov == sorted(cov)                  # monotonically non-decreasing
    assert set(sweep.columns) >= {
        "r_pz_m", "coverage_pct", "clean_pct", "duel_pct",
        "n_segments", "median_hold_frames",
    }
