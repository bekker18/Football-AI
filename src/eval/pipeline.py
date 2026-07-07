"""Top-level evaluation: run the flip search, compute every metric, report.

``evaluate`` takes two canonical detection tables (ground truth, prediction) in
the same pitch frame and returns a nested report. Because our attacking
direction / pitch orientation is arbitrary, it evaluates the prediction under
each configured orientation transform and keeps the one with the best detection
F1 — reporting which was chosen so a flip is never silent.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import EvalConfig
from .detections import apply_transform, as_detections
from .matching import match_all
from .metrics import attribute_accuracy, detection_metrics, idf1, localization_metrics


def _evaluate_oriented(gt: pd.DataFrame, pred: pd.DataFrame, cfg: EvalConfig) -> dict:
    """All metrics for one fixed orientation of ``pred``."""
    matches = match_all(gt, pred, cfg)
    det = detection_metrics(n_tp=len(matches), n_gt=len(gt), n_pred=len(pred))
    return dict(
        detection=det,
        localization=localization_metrics(matches),
        identity=idf1(gt, pred, matches),
        attributes=attribute_accuracy(matches, cfg),
        _matches=matches,
    )


def evaluate(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    cfg: EvalConfig,
    *,
    already_canonical: bool = True,
) -> dict:
    """Evaluate ``pred`` against ``gt`` (both canonical detection tables).

    Set ``already_canonical=False`` to run :func:`as_detections` (with role
    filtering) on inputs that merely *look* canonical. Returns a report dict:
    ``{orientation, n_gt, n_pred, n_frames, detection, localization, identity,
    attributes, flip_search}``.
    """
    if not already_canonical:
        gt = as_detections(gt, roles=cfg.roles)
        pred = as_detections(pred, roles=cfg.roles)

    best = None  # (res, f1, transform)
    search = []
    for t in cfg.flip_candidates:
        res = _evaluate_oriented(gt, apply_transform(pred, t, cfg), cfg)
        f1 = res["detection"]["f1"]
        search.append(dict(transform=t, f1=f1))
        if best is None or f1 > best[1]:
            best = (res, f1, t)

    res, _, t = best
    frames = pd.concat([gt["frame"], pred["frame"]]).nunique() if len(gt) or len(pred) else 0
    report = dict(
        orientation=t,
        match_dist_m=cfg.match_dist_m,
        n_gt=int(len(gt)),
        n_pred=int(len(pred)),
        n_frames=int(frames),
        detection=res["detection"],
        localization=res["localization"],
        identity=res["identity"],
        attributes=res["attributes"],
        flip_search=search,
    )
    return report


def summarize(report: dict) -> str:
    """One-screen human summary of an :func:`evaluate` report."""
    d, loc, idn = report["detection"], report["localization"], report["identity"]
    lines = [
        f"orientation      : {report['orientation']}  "
        f"(gate {report['match_dist_m']} m)",
        f"detections       : {report['n_gt']} gt / {report['n_pred']} pred "
        f"over {report['n_frames']} frames",
        f"detection        : P={d['precision']}  R={d['recall']}  F1={d['f1']}  "
        f"(TP={d['tp']} FP={d['fp']} FN={d['fn']})",
        f"localization     : mean={loc['mean_m']} m  rmse={loc['rmse_m']} m  "
        f"p95={loc['p95_m']} m  (n={loc['n']})",
        f"identity (IDF1)  : IDF1={idn['idf1']}  IDP={idn['idp']}  IDR={idn['idr']}",
    ]
    attrs = report.get("attributes", {})
    if "role" in attrs:
        lines.append(
            f"role accuracy    : {attrs['role']['accuracy']}  (n={attrs['role']['n']})"
        )
    if "team" in attrs and attrs["team"].get("accuracy") is not None:
        lines.append(
            f"team accuracy    : {attrs['team']['accuracy']}  "
            f"(mapping={attrs['team']['mapping']}, n={attrs['team']['n']})"
        )
    return "\n".join(lines)
