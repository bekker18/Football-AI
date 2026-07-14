"""Tests for the pure team-voting logic (no CV stack required)."""

from src.cv.teams import majority_vote_teams


def test_majority_vote_locks_track_to_mode():
    rows = [
        {"object_id": 7, "role": "player", "team": 0},
        {"object_id": 7, "role": "player", "team": 1},  # stray flicker
        {"object_id": 7, "role": "player", "team": 0},
    ]
    n_tracks, n_flips = majority_vote_teams(rows)
    assert n_tracks == 1
    assert n_flips == 1
    assert [r["team"] for r in rows] == [0, 0, 0]


def test_majority_vote_fills_none_from_track():
    rows = [
        {"object_id": 3, "role": "player", "team": 1},
        {"object_id": 3, "role": "player", "team": None},  # not-yet-predicted frame
    ]
    n_tracks, n_flips = majority_vote_teams(rows)
    assert n_tracks == 1
    assert rows[1]["team"] == 1


def test_ball_and_untracked_are_ignored():
    rows = [
        {"object_id": 0, "role": "ball", "team": None},
        {"object_id": -1, "role": "player", "team": 0},  # untracked sentinel
    ]
    n_tracks, n_flips = majority_vote_teams(rows)
    assert n_tracks == 0
    assert n_flips == 0


def test_goalkeeper_and_player_share_a_track_id():
    # Same physical person mis-classified across frames still votes as one team.
    rows = [
        {"object_id": 5, "role": "player", "team": 1},
        {"object_id": 5, "role": "goalkeeper", "team": 1},
        {"object_id": 5, "role": "player", "team": 0},
    ]
    n_tracks, n_flips = majority_vote_teams(rows)
    assert n_tracks == 1
    assert [r["team"] for r in rows] == [1, 1, 1]
