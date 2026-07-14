"""The two-pass gate: window selection, budget, and frame-range collapsing."""

import numpy as np
import pandas as pd

from src.twopass import TwoPassConfig, frames_to_ranges, plan_ball_frames

WIN_COLS = ["window_id", "start_frame", "end_frame", "peak_value"]


def win(wid, start, end, peak):
    return dict(window_id=wid, start_frame=start, end_frame=end, peak_value=peak)


def table(rows):
    return pd.DataFrame(rows, columns=WIN_COLS)


def test_frames_to_ranges_collapses_contiguous():
    assert frames_to_ranges(np.array([1, 2, 3, 7, 8, 20])) == [(1, 3), (7, 8), (20, 20)]
    assert frames_to_ranges(np.array([])) == []


def test_no_budget_selects_every_window():
    w = table([win(0, 0, 9, 5.0), win(1, 50, 59, 8.0)])
    cfg = TwoPassConfig(budget_frac=None)
    frames, plan = plan_ball_frames(w, total_frames=100, cfg=cfg)
    assert plan["n_windows_selected"] == 2
    assert plan["n_frames"] == 20
    assert set(plan["selected_window_ids"]) == {0, 1}


def test_budget_prefers_highest_value_windows():
    # three 10-frame windows but budget only fits two (20 frames of 100 = 20%)
    w = table([win(0, 0, 9, 3.0), win(1, 20, 29, 9.0), win(2, 40, 49, 6.0)])
    cfg = TwoPassConfig(budget_frac=0.20)
    frames, plan = plan_ball_frames(w, total_frames=100, cfg=cfg)
    assert plan["budget_frames"] == 20
    assert plan["n_frames"] == 20
    # the two highest-value windows (1 and 2), not the low-value 0
    assert plan["selected_window_ids"] == [1, 2]
    assert plan["coverage_frac"] == 0.20


def test_overlapping_windows_union_frames_once():
    w = table([win(0, 0, 9, 5.0), win(1, 5, 14, 4.0)])  # overlap on 5..9
    cfg = TwoPassConfig(budget_frac=None)
    frames, plan = plan_ball_frames(w, total_frames=100, cfg=cfg)
    assert plan["n_frames"] == 15  # 0..14, overlap counted once
    assert frames.tolist() == list(range(15))


def test_windows_clamped_to_clip_bounds():
    w = table([win(0, -5, 4, 5.0), win(1, 95, 110, 4.0)])
    cfg = TwoPassConfig(budget_frac=None)
    frames, plan = plan_ball_frames(w, total_frames=100, cfg=cfg)
    # clamped to [0,4] (5 frames) and [95,99] (5 frames)
    assert plan["n_frames"] == 10
    assert frames.min() == 0 and frames.max() == 99


def test_min_peak_value_filters_windows():
    w = table([win(0, 0, 9, 2.0), win(1, 20, 29, 9.0)])
    cfg = TwoPassConfig(budget_frac=None, min_peak_value=5.0)
    frames, plan = plan_ball_frames(w, total_frames=100, cfg=cfg)
    assert plan["selected_window_ids"] == [1]


def test_oversized_top_window_is_truncated_not_dropped():
    # single window (100 frames) far exceeds a 10-frame budget; it must be
    # truncated to the budget around its centre, never skipped to zero coverage.
    w = table([win(0, 0, 99, 9.0)])
    cfg = TwoPassConfig(budget_frac=0.10)  # 10 frames of 100
    frames, plan = plan_ball_frames(w, total_frames=100, cfg=cfg)
    assert plan["n_frames"] == 10
    assert plan["selected_window_ids"] == [0]
    assert plan["truncated_window_ids"] == [0]
    # centred on the midpoint (~49): a contiguous 10-frame chunk around it
    assert frames.min() >= 40 and frames.max() <= 59
    assert list(frames) == list(range(frames.min(), frames.min() + 10))


def test_truncation_centers_on_core_when_present():
    rows = [dict(window_id=0, start_frame=0, end_frame=99, peak_value=9.0,
                 core_start_frame=80, core_end_frame=88)]
    w = pd.DataFrame(rows)
    cfg = TwoPassConfig(budget_frac=0.10)
    frames, plan = plan_ball_frames(w, total_frames=100, cfg=cfg)
    assert plan["n_frames"] == 10
    # centred on the core midpoint (84), not the window midpoint (49)
    assert 84 in frames and frames.min() >= 79


def test_empty_windows_plans_nothing():
    cfg = TwoPassConfig()
    frames, plan = plan_ball_frames(table([]), total_frames=100, cfg=cfg)
    assert len(frames) == 0 and plan["n_frames"] == 0
    assert plan["n_windows_total"] == 0
