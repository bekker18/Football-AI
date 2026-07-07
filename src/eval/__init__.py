"""Layer 1 quality benchmark — evaluate extracted game state against ground truth.

The stage that answers "how good is the extraction?" quantitatively, so anything
built on top of Layer 1 (eventing, valuation) rests on measured numbers rather
than eyeballed video. Compares predicted per-frame detections against ground
truth **in pitch-metre space** and reports:

- **detection** — precision / recall / F1 at a pitch-distance gate,
- **localization** — mean / RMSE position error over matched pairs (metres),
- **identity** — IDF1 / IDP / IDR (association quality; penalises id switches),
- **attributes** — role and (permutation-invariant) team accuracy on matches.

The metric engine (:mod:`matching`, :mod:`metrics`, :mod:`pipeline`) is fully
format-agnostic: it consumes a canonical detection table. Format adapters
(:mod:`soccernet`) convert ground truth / our tracking into that table. Depends
only on pandas / numpy / scipy — not the Layer 1 CV stack.

    from src.eval import EvalConfig, evaluate, as_detections

Ground-truth coordinates and predicted coordinates must live in the **same pitch
frame**; attacking-direction / orientation ambiguity is handled by the optional
flip search (see :class:`EvalConfig.flip_candidates`).
"""

from __future__ import annotations

from .adapters import load_soccernet_gsr, load_tracking
from .config import CANON_COLS, EvalConfig
from .detections import apply_transform, as_detections
from .matching import match_all, match_frame
from .metrics import (
    attribute_accuracy,
    detection_metrics,
    idf1,
    localization_metrics,
)
from .pipeline import evaluate, summarize

__all__ = [
    "EvalConfig",
    "CANON_COLS",
    "as_detections",
    "apply_transform",
    "match_frame",
    "match_all",
    "detection_metrics",
    "localization_metrics",
    "attribute_accuracy",
    "idf1",
    "evaluate",
    "summarize",
    "load_tracking",
    "load_soccernet_gsr",
]
