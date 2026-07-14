"""Writing the output artifacts and collecting environment metadata."""

from __future__ import annotations

import json
import os
from importlib import metadata
from typing import List, Tuple

import pandas as pd


def pkg_versions() -> dict:
    """Pinned versions of the key libraries, for the run manifest."""
    out = {}
    for p in (
        "torch",
        "ultralytics",
        "supervision",
        "transformers",
        "umap-learn",
        "scikit-learn",
        "opencv-python-headless",
        "numpy",
    ):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def write_outputs(
    out_dir: str,
    rows: List[dict],
    frames_jsonl: List[dict],
    meta: dict,
    ball_debug: List[dict] | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """Write tracking.parquet/csv, frames.jsonl and meta.json.

    ``ball_debug`` (every ball candidate, selected or rejected, with its scores)
    is written to a *separate* ball_candidates.parquet rather than into
    tracking.parquet — that file's one-ball-row-per-frame shape is a contract six
    downstream modules rely on when they select ``object_id == BALL_OBJECT_ID``.

    Returns (dataframe, {name: path}). The dataframe is returned so the caller
    can report summary stats without re-reading the file.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    if "object_id" in df.columns:
        # Single-typed column so arrow/parquet can serialise it: tracker ids stay
        # integers, and any untracked/ball rows remain clean integers too.
        df["object_id"] = df["object_id"].astype("Int64")
    if "ball_track_id" in df.columns:
        df["ball_track_id"] = df["ball_track_id"].astype("Int64")

    paths = {
        "parquet": os.path.join(out_dir, "tracking.parquet"),
        "csv": os.path.join(out_dir, "tracking.csv"),
        "jsonl": os.path.join(out_dir, "frames.jsonl"),
        "meta": os.path.join(out_dir, "meta.json"),
    }
    if ball_debug:
        paths["ball_candidates"] = os.path.join(out_dir, "ball_candidates.parquet")
        pd.DataFrame(ball_debug).to_parquet(paths["ball_candidates"], index=False)
    df.to_parquet(paths["parquet"], index=False)
    df.to_csv(paths["csv"], index=False)
    with open(paths["jsonl"], "w") as f:
        for rec in frames_jsonl:
            f.write(json.dumps(rec) + "\n")
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    return df, paths
