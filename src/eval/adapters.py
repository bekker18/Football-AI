"""Format adapters — turn real files into the canonical detection table.

Two sources today:

- :func:`load_tracking` — our own Layer 1 output (``tracking.parquet``) or the
  prepared output (``tracking_prepared.parquet``). Prefers the prepared, target-
  frame coordinates and ``stable_id`` when present.
- :func:`load_soccernet_gsr` — SoccerNet Game State Reconstruction ground truth
  (``Labels-GameState.json``), the benchmark this harness targets.

The adapters are deliberately thin and defensive: the durable value is the
format-agnostic metric engine, and the exact GSR field layout should be
confirmed against a real sample (see the note on :func:`load_soccernet_gsr`).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

from .config import EvalConfig
from .detections import as_detections

# GSR role vocabulary -> ours, so role accuracy compares like with like.
_GSR_ROLE_MAP = {
    "player": "player",
    "goalkeeper": "goalkeeper",
    "goalkeepers": "goalkeeper",
    "referee": "referee",
    "ball": "ball",
    "other": None,
}


def load_tracking(
    path: str, cfg: EvalConfig, *, prefer_prepared: bool = True
) -> pd.DataFrame:
    """Load our tracking output into the canonical detection table.

    ``path`` may be a directory (we look for ``tracking_prepared.parquet`` then
    ``tracking.parquet``) or a direct parquet/csv file. When the prepared,
    target-frame columns (``pitch_x_t_m`` / ``stable_id``) are present they are
    used, so coordinates land in the same frame ``cfg`` describes.
    """
    df = _read_tracking_frame(path, prefer_prepared)

    x_col = "pitch_x_t_m" if "pitch_x_t_m" in df.columns else "pitch_x_m"
    y_col = "pitch_y_t_m" if "pitch_y_t_m" in df.columns else "pitch_y_m"
    id_col = "stable_id" if "stable_id" in df.columns else "object_id"

    return as_detections(
        df,
        colmap=dict(frame="frame", track_id=id_col, x=x_col, y=y_col,
                    role="role", team="team"),
        roles=cfg.roles,
    )


def _read_tracking_frame(path: str, prefer_prepared: bool) -> pd.DataFrame:
    if os.path.isdir(path):
        candidates = (
            ["tracking_prepared.parquet", "tracking.parquet", "tracking.csv"]
            if prefer_prepared
            else ["tracking.parquet", "tracking.csv"]
        )
        for name in candidates:
            p = os.path.join(path, name)
            if os.path.exists(p):
                path = p
                break
        else:
            raise SystemExit(f"no tracking parquet/csv found in {path!r}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_soccernet_gsr(
    path: str,
    cfg: EvalConfig,
    *,
    pitch_units: str = "cm",
    center_origin: bool = True,
) -> pd.DataFrame:
    """Load SoccerNet-GSR ground truth into the canonical detection table.

    Expects a ``Labels-GameState.json`` (or a sequence directory containing one).
    Its ``annotations`` carry a per-detection ``track_id``, an ``attributes``
    block (``role`` / ``team``), and pitch coordinates under ``bbox_pitch``
    (``x_bottom_middle`` / ``y_bottom_middle``); ``images`` carry the frame index.

    Coordinates are mapped into ``cfg``'s corner-origin metre frame:
    ``pitch_units`` scales to metres (GSR pitch coords are centimetres), and
    ``center_origin`` shifts a pitch-centre origin to the corner (x += L/2,
    y += W/2). **Confirm these two against a real sample** — they are the only
    assumptions the metric engine can't self-check.
    """
    doc = _read_gsr_json(path)
    images = doc.get("images", [])
    frame_of = {img["image_id"]: _image_frame(img) for img in images}

    scale = {"cm": 0.01, "m": 1.0, "mm": 0.001}.get(pitch_units)
    if scale is None:
        raise ValueError(f"unknown pitch_units {pitch_units!r} (cm/m/mm)")
    L, W = cfg.pitch()

    recs = []
    for ann in doc.get("annotations", []):
        bp = ann.get("bbox_pitch") or {}
        if bp.get("x_bottom_middle") is None or bp.get("y_bottom_middle") is None:
            continue  # no pitch fix for this detection
        x = float(bp["x_bottom_middle"]) * scale
        y = float(bp["y_bottom_middle"]) * scale
        if center_origin:
            x += L / 2.0
            y += W / 2.0
        attrs = ann.get("attributes") or {}
        recs.append(
            dict(
                frame=frame_of.get(ann.get("image_id")),
                track_id=ann.get("track_id"),
                x=x,
                y=y,
                role=_GSR_ROLE_MAP.get(str(attrs.get("role", "")).lower(),
                                       attrs.get("role")),
                team=_gsr_team(attrs.get("team")),
            )
        )
    df = pd.DataFrame(recs)
    return as_detections(df, roles=cfg.roles)


def _read_gsr_json(path: str) -> dict:
    if os.path.isdir(path):
        for name in ("Labels-GameState.json", "labels-gamestate.json"):
            p = os.path.join(path, name)
            if os.path.exists(p):
                path = p
                break
        else:
            raise SystemExit(f"no Labels-GameState.json in {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _image_frame(img: dict) -> Optional[int]:
    """Frame index for a GSR image record (explicit field, else parsed name)."""
    for key in ("frame_index", "frame", "frame_id"):
        if key in img and img[key] is not None:
            return int(img[key])
    name = os.path.splitext(os.path.basename(str(img.get("file_name", ""))))[0]
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None


def _gsr_team(team) -> Optional[int]:
    """Map GSR team ('left'/'right' or 0/1) to a 0/1 cluster id (team-invariant
    accuracy handles which is which)."""
    if team is None:
        return None
    s = str(team).lower()
    if s in ("left", "0", "home"):
        return 0
    if s in ("right", "1", "away"):
        return 1
    try:
        return int(team)
    except (TypeError, ValueError):
        return None
