"""Cropping helpers that feed the team classifier."""

from __future__ import annotations

from typing import List

import numpy as np
import supervision as sv


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """Full bounding-box crops for each detection."""
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def jersey_crop(frame: np.ndarray, xyxy) -> np.ndarray:
    """Upper-torso ('jersey') crop, to focus the team classifier on kit colour
    rather than grass / shorts / skin. Falls back to the full box when tiny."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 12:  # too small to carve up reliably
        return sv.crop_image(frame, xyxy)
    # skip the head, stop before the shorts, and trim side background
    band = np.array(
        [x1 + 0.15 * w, y1 + 0.12 * h, x2 - 0.15 * w, y1 + 0.55 * h],
        dtype=np.float32,
    )
    return sv.crop_image(frame, band)


def get_team_crops(
    frame: np.ndarray, detections: sv.Detections, jersey: bool
) -> List[np.ndarray]:
    """Crops fed to the TeamClassifier — jersey band by default, full box if off."""
    if not jersey:
        return get_crops(frame, detections)
    return [jersey_crop(frame, xyxy) for xyxy in detections.xyxy]
