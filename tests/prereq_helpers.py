"""Tiny builders for the prerequisites unit tests (numpy/pandas only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# The full Layer 1 per-row schema, so built frames look like real game state.
COLUMNS = [
    "frame", "time_s", "object_id", "role", "team", "img_x", "img_y",
    "pitch_x_m", "pitch_y_m", "pitch_valid", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]


def row(frame, object_id, role, team, px, py, *, valid=True, fps=25.0):
    """One tracking row with sensible defaults for the fields tests ignore."""
    return dict(
        frame=int(frame),
        time_s=round(frame / fps, 4),
        object_id=int(object_id),
        role=role,
        team=(np.nan if team is None else float(team)),
        img_x=0.0,
        img_y=0.0,
        pitch_x_m=(np.nan if px is None else float(px)),
        pitch_y_m=(np.nan if py is None else float(py)),
        pitch_valid=bool(valid),
        bbox_x1=0.0, bbox_y1=0.0, bbox_x2=1.0, bbox_y2=1.0,
    )


def make_df(rows):
    """Assemble rows into a DataFrame with the Layer 1 column order + dtypes."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["object_id"] = df["object_id"].astype("Int64")
    return df
