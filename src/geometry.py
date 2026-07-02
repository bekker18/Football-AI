"""Homography between image pixels and pitch metres.

Heavy imports (supervision / roboflow-sports) are deferred into
``build_transformer`` so that ``to_pitch_m`` — a pure numpy function — can be
imported and unit-tested without the full CV stack installed.
"""

from __future__ import annotations

import numpy as np


def build_transformer(pitch_result):
    """Homography from detected pitch keypoints; None if <4 reliable points."""
    import supervision as sv
    from sports.common.view import ViewTransformer

    from .config import PITCH_VERTICES

    kp = sv.KeyPoints.from_ultralytics(pitch_result)
    if kp.xy is None or len(kp.xy) == 0:
        return None
    pts = kp.xy[0]
    mask = (pts[:, 0] > 1) & (pts[:, 1] > 1)
    if int(mask.sum()) < 4:
        return None
    try:
        return ViewTransformer(
            source=pts[mask].astype(np.float32),
            target=PITCH_VERTICES[mask].astype(np.float32),
        )
    except Exception:
        return None


def to_pitch_m(transformer, img_x: float, img_y: float):
    """Image bottom-center -> pitch (x_m, y_m).

    Returns (None, None, False) when there is no homography for the frame.
    """
    if transformer is None:
        return None, None, False
    t = transformer.transform_points(np.array([[img_x, img_y]], dtype=np.float32))[0]
    return float(t[0] / 100.0), float(t[1] / 100.0), True
