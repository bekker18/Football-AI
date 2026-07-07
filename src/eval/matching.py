"""Per-frame bipartite matching between ground-truth and predicted detections.

Within each frame, ground-truth and predicted points are matched one-to-one by
minimum total pitch distance (Hungarian assignment), then any assigned pair
farther apart than the gate is rejected. Rejected/leftover points on each side
become false negatives / false positives.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .config import EvalConfig


def match_frame(
    gt_xy: np.ndarray, pred_xy: np.ndarray, gate_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match one frame's points.

    ``gt_xy`` / ``pred_xy`` are ``(n, 2)`` metre coordinates. Returns
    ``(gt_idx, pred_idx, dist)`` for accepted matches (distance <= gate), each a
    1-D array of equal length. Empty inputs yield empty arrays.
    """
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return np.array([], int), np.array([], int), np.array([], float)

    cost = cdist(gt_xy, pred_xy)  # euclidean, metres
    gi, pj = linear_sum_assignment(cost)
    d = cost[gi, pj]
    keep = d <= gate_m
    return gi[keep], pj[keep], d[keep]


def match_all(
    gt: pd.DataFrame, pred: pd.DataFrame, cfg: EvalConfig
) -> pd.DataFrame:
    """Match every frame present in either table.

    Returns one row per accepted match with the columns ``frame, gt_id, pred_id,
    dist, gt_role, pred_role, gt_team, pred_team`` — the substrate every metric
    is computed from. Frames with detections on only one side contribute no match
    rows (their points are pure FP or FN, counted downstream from the totals).
    """
    gate = cfg.match_dist_m
    frames = np.union1d(gt["frame"].to_numpy(), pred["frame"].to_numpy())
    gt_by = {f: g for f, g in gt.groupby("frame")}
    pred_by = {f: p for f, p in pred.groupby("frame")}

    recs = []
    for f in frames:
        g = gt_by.get(f)
        p = pred_by.get(f)
        if g is None or p is None:
            continue
        gi, pj, dist = match_frame(
            g[["x", "y"]].to_numpy(float), p[["x", "y"]].to_numpy(float), gate
        )
        if len(gi) == 0:
            continue
        gr = g.iloc[gi].reset_index(drop=True)
        pr = p.iloc[pj].reset_index(drop=True)
        recs.append(
            pd.DataFrame(
                dict(
                    frame=int(f),
                    gt_id=gr["track_id"].to_numpy(),
                    pred_id=pr["track_id"].to_numpy(),
                    dist=dist,
                    gt_role=gr["role"].to_numpy(),
                    pred_role=pr["role"].to_numpy(),
                    gt_team=gr["team"].to_numpy(),
                    pred_team=pr["team"].to_numpy(),
                )
            )
        )

    cols = ["frame", "gt_id", "pred_id", "dist", "gt_role", "pred_role",
            "gt_team", "pred_team"]
    if not recs:
        return pd.DataFrame(columns=cols)
    return pd.concat(recs, ignore_index=True)
