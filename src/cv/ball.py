"""Optional ball detection: tiled slicing, keeping every ball-class candidate.

A broadcast frame contains several ball-class detections — the in-play ball plus
static spare balls by the touchline and behind the goals. This module's job is to
find them all; deciding *which one is the game ball* is
:mod:`src.cv.ball_select`'s job, and it needs to see all of them to do it.

The legacy path (``sports.common.ball.BallTracker``) is kept behind ``legacy=True``
purely so the two can be A/B'd. Do not ship on it: it returns the detection
nearest the centroid of a rolling buffer of *all* recent detections, and a static
spare ball — contributing the same point on every frame — drags that centroid
onto itself and then wins the argmin indefinitely.
"""

from __future__ import annotations

import inspect

import numpy as np


class BallDetector:
    """Wraps the ball YOLO model and a tiled InferenceSlicer.

    The ball is small and noisy on broadcast video, so it is detected by slicing
    each frame into tiles and running the ball model per tile.
    """

    def __init__(
        self,
        ball_pt: str,
        device: str,
        ball_imgsz: int,
        use_half: bool,
        legacy: bool = False,
    ):
        import supervision as sv
        from ultralytics import YOLO

        self.model = YOLO(ball_pt).to(device)
        self.legacy = legacy
        self.tracker = None
        if legacy:
            from sports.common.ball import BallTracker

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
        """Every ball-class detection in one frame; returns sv.Detections.

        Returns *all* candidates (0..N). In legacy mode it collapses to the single
        nearest-the-centroid detection, reproducing the old — broken — behaviour.
        """
        bdet = self.slicer(frame).with_nms(threshold=0.1)
        if self.tracker is not None:
            return self.tracker.update(bdet)
        return bdet
