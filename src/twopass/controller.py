"""Pass 2 executor — run the ball detector on only the planned frames.

Needs the Layer 1 CV stack (ultralytics / supervision + the checkpoints), so it
is imported lazily by the CLI and never by ``src.twopass.__init__`` — the gate
half of the package stays dependency-light. Emits a **sparse** ball table in the
Layer 1 ball-row schema, covering just the high-value windows, ready to feed the
prerequisites' ball smoothing.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import TwoPassConfig

# Layer 1 ball-row schema (subset of the tracking columns, ball only).
BALL_COLUMNS = [
    "frame", "time_s", "object_id", "role", "team", "img_x", "img_y",
    "pitch_x_m", "pitch_y_m", "pitch_valid",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]


def run_ball_on_frames(
    source: str,
    frames: Iterable[int],
    *,
    pitch_pt: str,
    ball_pt: str,
    cfg: TwoPassConfig,
    fps: Optional[float] = None,
) -> pd.DataFrame:
    """Detect the ball on ``frames`` only, returning Layer 1 ball rows.

    Decodes the clip sequentially but runs the (expensive) pitch + ball models
    *only* on target frames — the tiled ball slicer is the real cost, so skipping
    it on non-target frames is where the two-pass saving comes from. Frames past
    the last target are not decoded.
    """
    import supervision as sv
    from ultralytics import YOLO

    from src.cv import config as l1cfg
    from src.cv.ball import BallDetector
    from src.cv.geometry import build_transformer, to_pitch_m

    target = set(int(f) for f in frames)
    if not target:
        return pd.DataFrame(columns=BALL_COLUMNS)
    last = max(target)

    info = sv.VideoInfo.from_video_path(source)
    fps = fps or info.fps or 25.0
    use_half = cfg.device == "cuda"
    pitch_model = YOLO(pitch_pt).to(cfg.device)
    ball_detector = BallDetector(ball_pt, cfg.device, cfg.ball_imgsz, use_half)

    rows: List[dict] = []
    for fidx, frame in enumerate(sv.get_video_frames_generator(source)):
        if fidx > last:
            break
        if fidx not in target:
            continue
        transformer = build_transformer(
            pitch_model(frame, verbose=False, half=use_half)[0]
        )
        bdet = ball_detector.detect(frame)
        for i in range(len(bdet)):
            rows.append(
                _ball_row(fidx, fidx / fps, bdet.xyxy[i], transformer, to_pitch_m,
                          l1cfg.BALL_OBJECT_ID)
            )

    return pd.DataFrame(rows, columns=BALL_COLUMNS)


def _ball_row(fidx, time_s, xyxy, transformer, to_pitch_m, ball_id) -> dict:
    """One ball row, using the bottom-center anchor like Layer 1's ``_build_row``."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    ix, iy = (x1 + x2) / 2.0, y2
    px, py, ok = to_pitch_m(transformer, ix, iy)
    return dict(
        frame=fidx, time_s=round(time_s, 4), object_id=ball_id, role="ball", team=None,
        img_x=round(ix, 2), img_y=round(iy, 2),
        pitch_x_m=(round(px, 3) if ok else None),
        pitch_y_m=(round(py, 3) if ok else None),
        pitch_valid=ok,
        bbox_x1=round(x1, 1), bbox_y1=round(y1, 1),
        bbox_x2=round(x2, 1), bbox_y2=round(y2, 1),
    )
