"""IDF1, attribute accuracy, flip search, and the as_detections adapter."""

import numpy as np
import pandas as pd

from src.eval import (
    EvalConfig,
    as_detections,
    attribute_accuracy,
    evaluate,
    idf1,
    match_all,
)


def det(frame, tid, x, y, role="player", team=0):
    return dict(frame=frame, track_id=tid, x=x, y=y, role=role, team=team)


def table(rows):
    return pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "role", "team"])


def _moving_track(tid, n=10, y=34.0, role="player", team=0):
    return [det(f, tid, float(f), y, role, team) for f in range(n)]


def test_idf1_perfect_identity():
    gt = table(_moving_track(1))
    pred = table(_moving_track(99))  # one pred track, different id, same path
    m = match_all(gt, pred, EvalConfig())
    r = idf1(gt, pred, m)
    assert r["idf1"] == 1.0 and r["idfp"] == 0 and r["idfn"] == 0


def test_idf1_penalizes_id_switch():
    # gt is one 10-frame track; prediction splits it into two 5-frame ids.
    gt = table(_moving_track(1, n=10))
    pred_rows = [det(f, 100, float(f), 34.0) for f in range(5)]
    pred_rows += [det(f, 200, float(f), 34.0) for f in range(5, 10)]
    pred = table(pred_rows)
    m = match_all(gt, pred, EvalConfig())
    # every frame still matches -> detection recall is perfect...
    assert len(m) == 10
    # ...but identity is halved by the switch: IDTP=5, IDFP=5, IDFN=5.
    r = idf1(gt, pred, m)
    assert r["idf1"] == 0.5
    assert r["idtp"] == 5 and r["idfp"] == 5 and r["idfn"] == 5


def test_team_accuracy_is_permutation_invariant():
    # predicted team is the *inverse* label of gt on every matched pair;
    # the swapped mapping should score a perfect 1.0.
    gt = table([det(0, 1, 10, 34, team=0), det(0, 2, 20, 34, team=1)])
    pred = table([det(0, 5, 10, 34, team=1), det(0, 6, 20, 34, team=0)])
    m = match_all(gt, pred, EvalConfig())
    a = attribute_accuracy(m, EvalConfig())
    assert a["team"]["accuracy"] == 1.0
    assert a["team"]["mapping"] == "swapped"


def test_flip_search_recovers_rotated_predictions():
    cfg = EvalConfig(pitch_length_m=105.0, pitch_width_m=68.0)
    gt = table([det(0, 1, 20.0, 10.0), det(0, 2, 40.0, 50.0)])
    # predictions rotated 180deg: x -> 105-x, y -> 68-y
    pred = table([det(0, 5, 85.0, 58.0), det(0, 6, 65.0, 18.0)])
    report = evaluate(gt, pred, cfg)
    assert report["orientation"] == "rot180"
    assert report["detection"]["f1"] == 1.0
    # under identity these would all miss the gate
    scores = {s["transform"]: s["f1"] for s in report["flip_search"]}
    assert scores["none"] == 0.0 and scores["rot180"] == 1.0


def test_as_detections_drops_unfixed_and_filters_roles():
    raw = pd.DataFrame(
        dict(
            frame=[0, 0, 0, 0],
            track_id=[1, 2, 3, 0],
            x=[10.0, np.nan, 30.0, 50.0],  # row 2 has no pitch fix
            y=[34.0, 34.0, 34.0, 34.0],
            role=["player", "player", "referee", "ball"],
            team=[0, 1, None, None],
        )
    )
    out = as_detections(raw, roles=("player", "goalkeeper"))
    # NaN-x row dropped; referee + ball filtered out -> only track_id 1 remains
    assert out["track_id"].tolist() == [1]
