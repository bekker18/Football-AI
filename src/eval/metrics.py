"""Metrics computed from the match table + the raw detection tables.

- :func:`detection_metrics` — precision / recall / F1 from TP/FP/FN counts.
- :func:`localization_metrics` — mean / RMSE metre error over matched pairs.
- :func:`attribute_accuracy` — role and permutation-invariant team accuracy.
- :func:`idf1` — identity-level F1 (global trajectory assignment; punishes id
  switches, unlike the per-frame detection score).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import EvalConfig


def detection_metrics(n_tp: int, n_gt: int, n_pred: int) -> dict:
    """Precision/recall/F1 from a true-positive count and the two totals.

    ``n_tp`` accepted matches; ``n_gt``/``n_pred`` total ground-truth/predicted
    detections. FN = n_gt - n_tp, FP = n_pred - n_tp.
    """
    n_fn = n_gt - n_tp
    n_fp = n_pred - n_tp
    precision = n_tp / n_pred if n_pred else 0.0
    recall = n_tp / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return dict(
        tp=int(n_tp), fp=int(n_fp), fn=int(n_fn),
        precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4),
    )


def localization_metrics(matches: pd.DataFrame) -> dict:
    """Positional error over matched pairs, in metres."""
    if matches.empty:
        return dict(n=0, mean_m=None, rmse_m=None, p95_m=None)
    d = matches["dist"].to_numpy(float)
    return dict(
        n=int(len(d)),
        mean_m=round(float(d.mean()), 4),
        rmse_m=round(float(np.sqrt((d ** 2).mean())), 4),
        p95_m=round(float(np.percentile(d, 95)), 4),
    )


def _accuracy(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """Fraction equal over positions where both are non-null, plus that count."""
    ok = pd.notna(a) & pd.notna(b)
    n = int(ok.sum())
    if n == 0:
        return 0.0, 0
    return float((a[ok] == b[ok]).mean()), n


def attribute_accuracy(matches: pd.DataFrame, cfg: EvalConfig) -> dict:
    """Role accuracy and permutation-invariant team accuracy over matched pairs.

    Team labels are arbitrary cluster ids, so team accuracy is the better of the
    two label permutations (identity vs 0<->1 swap); the chosen mapping is
    reported. Role labels are compared directly (adapters are responsible for a
    shared role vocabulary).
    """
    out = {}
    if cfg.eval_role:
        acc, n = _accuracy(
            matches["gt_role"].to_numpy(object), matches["pred_role"].to_numpy(object)
        )
        out["role"] = dict(accuracy=round(acc, 4), n=n)

    if cfg.eval_team:
        gt = pd.to_numeric(matches["gt_team"], errors="coerce").to_numpy(float)
        pr = pd.to_numeric(matches["pred_team"], errors="coerce").to_numpy(float)
        ok = ~(np.isnan(gt) | np.isnan(pr))
        n = int(ok.sum())
        if n == 0:
            out["team"] = dict(accuracy=None, n=0, mapping=None)
        else:
            identity = float((gt[ok] == pr[ok]).mean())
            swapped = float((gt[ok] == (1 - pr[ok])).mean())
            if swapped > identity:
                out["team"] = dict(accuracy=round(swapped, 4), n=n, mapping="swapped")
            else:
                out["team"] = dict(accuracy=round(identity, 4), n=n, mapping="identity")
    return out


def idf1(gt: pd.DataFrame, pred: pd.DataFrame, matches: pd.DataFrame) -> dict:
    """Identity F1 via a global one-to-one trajectory assignment.

    For every (gt track, pred track) pair, ``m_ij`` counts the frames in which
    they are matched (present and within the gate — i.e. rows of ``matches``).
    A min-cost assignment then maps whole trajectories 1:1 so as to maximise the
    matched frames (IDTP); leftover GT/pred detections are IDFN / IDFP.

        IDP = IDTP / (IDTP + IDFP),  IDR = IDTP / (IDTP + IDFN)
        IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)
    """
    len_gt = gt.groupby("track_id").size()
    len_pred = pred.groupby("track_id").size()
    total_gt = int(len_gt.sum())
    total_pred = int(len_pred.sum())

    if total_gt == 0 or total_pred == 0 or matches.empty:
        idtp = 0
    else:
        m = (
            matches.groupby(["gt_id", "pred_id"]).size().reset_index(name="m")
        )
        gt_ids = list(len_gt.index)
        pred_ids = list(len_pred.index)
        gi = {t: k for k, t in enumerate(gt_ids)}
        pi = {t: k for k, t in enumerate(pred_ids)}
        M, N = len(gt_ids), len(pred_ids)

        mmat = np.zeros((M, N))
        for _, r in m.iterrows():
            mmat[gi[int(r["gt_id"])], pi[int(r["pred_id"])]] = r["m"]

        lg = len_gt.to_numpy(float)  # aligned with gt_ids
        lp = len_pred.to_numpy(float)  # aligned with pred_ids

        # Square cost matrix with dummy nodes: gt i may match pred j (cost
        # lg_i+lp_j-2*m_ij) or its own FN-dummy (cost lg_i); pred j may match its
        # FP-dummy (cost lp_j); dummy-dummy is free.
        INF = 1e9
        size = M + N
        cost = np.full((size, size), INF)
        cost[:M, :N] = lg[:, None] + lp[None, :] - 2.0 * mmat
        for i in range(M):
            cost[i, N + i] = lg[i]
        for j in range(N):
            cost[M + j, j] = lp[j]
        cost[M:, N:] = 0.0

        ri, cj = linear_sum_assignment(cost)
        total_cost = cost[ri, cj].sum()
        idtp = int(round((total_gt + total_pred - total_cost) / 2.0))

    idfn = total_gt - idtp
    idfp = total_pred - idtp
    idp = idtp / (idtp + idfp) if (idtp + idfp) else 0.0
    idr = idtp / (idtp + idfn) if (idtp + idfn) else 0.0
    idf1_v = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    return dict(
        idtp=int(idtp), idfp=int(idfp), idfn=int(idfn),
        idp=round(idp, 4), idr=round(idr, 4), idf1=round(idf1_v, 4),
    )
