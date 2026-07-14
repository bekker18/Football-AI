"""Per-frame matching + detection/localization metrics."""

import numpy as np
import pandas as pd

from src.eval import EvalConfig, detection_metrics, match_all, match_frame
from src.eval.metrics import localization_metrics


def det(frame, tid, x, y, role="player", team=0):
    return dict(frame=frame, track_id=tid, x=x, y=y, role=role, team=team)


def table(rows):
    return pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "role", "team"])


def test_match_frame_hungarian_respects_gate():
    gt = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred = np.array([[0.5, 0.0], [10.0, 5.0]])  # 2nd pair is 5 m apart
    gi, pj, d = match_frame(gt, pred, gate_m=2.0)
    assert list(gi) == [0] and list(pj) == [0]  # only the close pair survives
    assert d[0] == 0.5


def test_match_frame_empty_inputs():
    gi, pj, d = match_frame(np.empty((0, 2)), np.array([[1.0, 1.0]]), 2.0)
    assert len(gi) == len(pj) == len(d) == 0


def test_detection_metrics_perfect():
    gt = table([det(0, 1, 10, 34), det(0, 2, 20, 34)])
    pred = table([det(0, 5, 10, 34), det(0, 6, 20, 34)])
    m = match_all(gt, pred, EvalConfig())
    d = detection_metrics(len(m), len(gt), len(pred))
    assert d["precision"] == d["recall"] == d["f1"] == 1.0
    assert d["fp"] == d["fn"] == 0


def test_detection_metrics_counts_fp_and_fn():
    # 2 gt, 3 pred, one pred is a phantom far away -> 2 TP, 1 FP, 0 FN
    gt = table([det(0, 1, 10, 34), det(0, 2, 20, 34)])
    pred = table([det(0, 5, 10, 34), det(0, 6, 20, 34), det(0, 7, 90, 5)])
    m = match_all(gt, pred, EvalConfig())
    d = detection_metrics(len(m), len(gt), len(pred))
    assert d["tp"] == 2 and d["fp"] == 1 and d["fn"] == 0
    assert d["recall"] == 1.0
    assert round(d["precision"], 3) == round(2 / 3, 3)


def test_localization_error_is_metres():
    gt = table([det(0, 1, 10.0, 34.0)])
    pred = table([det(0, 5, 11.5, 34.0)])  # 1.5 m off, within gate 2.0
    m = match_all(gt, pred, EvalConfig())
    loc = localization_metrics(m)
    assert loc["n"] == 1
    assert abs(loc["mean_m"] - 1.5) < 1e-9
    assert abs(loc["rmse_m"] - 1.5) < 1e-9
