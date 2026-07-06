"""Load Layer 1 game state and write the prepared (event-ready) outputs.

Non-destructive: the prepared tracking table keeps every original column and
row (synthetic interpolated ball rows may be *added*, never removed) and simply
carries the extra columns the transforms produce.
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# Frame-level fields promoted to the top of each frames.jsonl record (the rest
# of a row's columns are nested per object, mirroring Layer 1's frames.jsonl).
FRAME_LEVEL_FIELDS = ("time_s", "pitch_valid", "in_play", "in_play_conf")


def load_gamestate(in_dir: str) -> Tuple[pd.DataFrame, dict]:
    """Read ``tracking.parquet`` (or ``tracking.csv``) and ``meta.json``.

    Parquet is preferred; the CSV is used only as a fallback so the layer still
    works if parquet/pyarrow is unavailable.
    """
    parquet = os.path.join(in_dir, "tracking.parquet")
    csv = os.path.join(in_dir, "tracking.csv")
    if os.path.exists(parquet):
        df = pd.read_parquet(parquet)
    elif os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        raise SystemExit(
            f"No tracking.parquet or tracking.csv found in {in_dir!r}. "
            f"Run Layer 1 first."
        )

    meta_path = os.path.join(in_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        print(f"[warn] no meta.json in {in_dir!r}; using default pitch/fps assumptions")
    return df, meta


def _clean(value):
    """Make one value JSON-safe: numpy scalars -> python, NaN/NA/None -> None."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    # pandas nullable scalars
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def frames_from_df(df: pd.DataFrame) -> List[dict]:
    """Rebuild a frames.jsonl-style list from the (prepared) tracking table.

    One record per frame: the frame-level fields promoted to the top, all of a
    row's columns nested under ``objects``. Rows are sorted by object_id so ball
    (id 0) leads, matching Layer 1's ordering intent.
    """
    records: List[dict] = []
    object_cols = [c for c in df.columns]
    for frame, g in df.groupby("frame", sort=True):
        g = g.sort_values("object_id", kind="stable")
        first = g.iloc[0]
        rec = {"frame": int(frame)}
        for field_name in FRAME_LEVEL_FIELDS:
            if field_name in g.columns:
                rec[field_name] = _clean(first[field_name])
        rec["objects"] = [
            {c: _clean(row[c]) for c in object_cols} for _, row in g.iterrows()
        ]
        records.append(rec)
    return records


def write_prepared(
    out_dir: str, df: pd.DataFrame, prep_meta: dict
) -> dict:
    """Write the three prepared artifacts and return their paths.

    - ``tracking_prepared.parquet`` — originals + every added column.
    - ``frames_prepared.jsonl`` — per-frame nested view.
    - ``prep_meta.json`` — all params + resolved directions + target pitch +
      stitching summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "parquet": os.path.join(out_dir, "tracking_prepared.parquet"),
        "jsonl": os.path.join(out_dir, "frames_prepared.jsonl"),
        "meta": os.path.join(out_dir, "prep_meta.json"),
    }

    df = df.sort_values(["frame", "object_id"], kind="stable").reset_index(drop=True)
    df.to_parquet(paths["parquet"], index=False)

    with open(paths["jsonl"], "w", encoding="utf-8") as f:
        for rec in frames_from_df(df):
            f.write(json.dumps(rec) + "\n")

    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(prep_meta, f, indent=2, default=_clean)

    return paths
