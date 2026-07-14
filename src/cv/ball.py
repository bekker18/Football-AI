"""Optional ball detection: tiled slicing + a simple proximity tracker."""

from __future__ import annotations

import inspect

import numpy as np


class BallDetector:
    """Wraps the ball YOLO model, a tiled InferenceSlicer, and a BallTracker.

    The ball is small and noisy on broadcast video, so it is detected by slicing
    each frame into tiles and running the ball model per tile.
    """

    def __init__(self, ball_pt: str, device: str, ball_imgsz: int, use_half: bool):
        import supervision as sv
        from sports.common.ball import BallTracker
        from ultralytics import YOLO

        self.model = YOLO(ball_pt).to(device)
        self.tracker = BallTracker(buffer_size=20)

        def _callback(image_slice: np.ndarray):
            r = self.model(
                image_slice, imgsz=ball_imgsz, verbose=False, half=use_half
            )[0]
            return sv.Detections.from_ultralytics(r)

        kwargs = dict(callback=_callback, slice_wh=(ball_imgsz, ball_imgsz))
        # supervision renamed overlap_filter_strategy -> overlap_filter (>=0.26);
        # support both so this runs outside the pinned image (e.g. Kaggle).
        params = inspect.signature(sv.InferenceSlicer.__init__).parameters
        if "overlap_filter" in params:
            kwargs["overlap_filter"] = sv.OverlapFilter.NONE
        elif "overlap_filter_strategy" in params:
            kwargs["overlap_filter_strategy"] = sv.OverlapFilter.NONE
        self.slicer = sv.InferenceSlicer(**kwargs)

    def detect(self, frame: np.ndarray):
        """Detect and track the ball in one frame; returns sv.Detections."""
        bdet = self.slicer(frame).with_nms(threshold=0.1)
        return self.tracker.update(bdet)
