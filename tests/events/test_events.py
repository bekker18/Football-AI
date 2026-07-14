"""Ball-free eventing: per-frame signals + high-value window assembly."""

import pandas as pd

from src.events import EventConfig, detect_high_value_windows
from src.events.signals import attacking_signals
from src.events.windows import windows_from_signals

# columns the events layer reads from the prepared table
PREP_COLS = ["frame", "time_s", "object_id", "role", "team",
             "attack_dir", "pitch_x_t_m", "pitch_y_t_m"]


def prep_rows(frame, team, xs, *, attack_dir=1.0, fps=25.0):
    """One team's players in a frame at the given attacking-normalized x's."""
    rows = []
    for i, x in enumerate(xs):
        rows.append(dict(
            frame=frame, time_s=round(frame / fps, 4), object_id=100 * int(team) + i,
            role="player", team=float(team), attack_dir=attack_dir,
            pitch_x_t_m=float(x), pitch_y_t_m=34.0,
        ))
    return rows


def make_prepared(rows):
    return pd.DataFrame(rows, columns=PREP_COLS)


def test_signal_high_when_team_commits_into_the_box():
    cfg = EventConfig(pitch_length_m=105.0)
    deep = make_prepared(prep_rows(0, team=0, xs=[90, 92, 95, 96, 98, 100]))  # in box
    shallow = make_prepared(prep_rows(1, team=0, xs=[20, 22, 24, 26, 28, 30]))  # own half

    s_deep = attacking_signals(deep, cfg)
    s_shallow = attacking_signals(shallow, cfg)
    assert s_deep["value_score"].iloc[0] >= cfg.value_threshold
    assert s_deep["n_box"].iloc[0] == 6
    assert s_shallow["value_score"].iloc[0] < cfg.value_threshold


def test_too_few_players_is_untrusted_zero():
    cfg = EventConfig(pitch_length_m=105.0, min_players_for_signal=6)
    # only 3 players deep -> below the trust threshold -> value forced to 0
    rows = make_prepared(prep_rows(0, team=0, xs=[95, 96, 97]))
    s = attacking_signals(rows, cfg)
    assert s["value_score"].iloc[0] == 0.0


def test_windows_form_from_contiguous_high_frames():
    cfg = EventConfig(pitch_length_m=105.0, window_merge_gap_frames=5,
                      window_min_frames=12, window_pad_frames=3)
    rows = []
    for f in range(50):
        xs = [95, 96, 97, 98, 99, 100] if 10 <= f <= 30 else [15, 16, 17, 18, 19, 20]
        rows.append(prep_rows(f, team=0, xs=xs))
    df = make_prepared([r for frame_rows in rows for r in frame_rows])

    windows, signals, meta = detect_high_value_windows(df, cfg)
    assert meta["n_windows"] == 1
    w = windows.iloc[0]
    assert w["core_start_frame"] == 10 and w["core_end_frame"] == 30
    assert w["start_frame"] == 7 and w["end_frame"] == 33  # padded by 3
    assert w["attacking_team"] == 0
    assert 0.0 < meta["coverage_frac"] < 1.0


def test_short_isolated_run_is_discarded():
    cfg = EventConfig(pitch_length_m=105.0, window_merge_gap_frames=5,
                      window_min_frames=12, window_pad_frames=3)
    rows = []
    for f in range(50):
        # one long run (10..30) + an isolated 3-frame blip (45..47) far enough
        # away (gap 15 > merge gap 5) to stay its own, too-short cluster
        high = (10 <= f <= 30) or (45 <= f <= 47)
        xs = [95, 96, 97, 98, 99, 100] if high else [15, 16, 17, 18, 19, 20]
        rows.append(prep_rows(f, team=0, xs=xs))
    df = make_prepared([r for frame_rows in rows for r in frame_rows])

    windows, _, meta = detect_high_value_windows(df, cfg)
    assert meta["n_windows"] == 1  # the blip is dropped
    assert windows.iloc[0]["core_end_frame"] == 30


def test_empty_windows_when_nothing_is_high():
    cfg = EventConfig(pitch_length_m=105.0)
    rows = []
    for f in range(20):
        rows.append(prep_rows(f, team=0, xs=[15, 16, 17, 18, 19, 20]))
    df = make_prepared([r for fr in rows for r in fr])
    windows, _, meta = detect_high_value_windows(df, cfg)
    assert meta["n_windows"] == 0 and len(windows) == 0
