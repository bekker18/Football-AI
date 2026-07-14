"""Smoke test: run the whole pipeline on the real Layer 1 output.

Skips cleanly if the sample tracking file or the tabular deps aren't present, so
the pure-logic tests still run in a minimal environment.
"""

import json
import os

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pandas as pd  # noqa: E402

from src.prerequisites import (  # noqa: E402
    config_from_meta,
    load_gamestate,
    run_prerequisites,
    write_prepared,
)
from src.prerequisites.config import (  # noqa: E402
    COL_ATTACK_DIR,
    COL_IN_PLAY,
    COL_PITCH_X_T,
    COL_PITCH_Y_T,
    COL_STABLE_ID,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMESTATE = os.path.join(REPO, "data", "gamestate")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not os.path.exists(os.path.join(GAMESTATE, "tracking.parquet")):
        pytest.skip("no sample tracking.parquet in data/gamestate")
    df, meta = load_gamestate(GAMESTATE)
    cfg = config_from_meta(meta)
    out_df, prep_meta = run_prerequisites(df.copy(), cfg)
    out_dir = tmp_path_factory.mktemp("prepared")
    paths = write_prepared(str(out_dir), out_df, prep_meta)
    return df, out_df, prep_meta, cfg, paths


def test_no_original_rows_dropped(prepared):
    df, out_df, *_ = prepared
    original = set(zip(df["frame"].tolist(), df["object_id"].astype("Int64").tolist()))
    got = set(zip(out_df["frame"].tolist(), out_df["object_id"].astype("Int64").tolist()))
    assert original <= got  # synthetic ball rows may be added, none removed
    assert len(out_df) >= len(df)


def test_added_columns_populated(prepared):
    _, out_df, *_ = prepared
    for col in (COL_STABLE_ID, COL_ATTACK_DIR, COL_PITCH_X_T, COL_PITCH_Y_T, COL_IN_PLAY):
        assert col in out_df.columns
    assert out_df[COL_STABLE_ID].notna().all()          # every row has a stable id
    assert out_df[COL_IN_PLAY].notna().all()            # every row has a play flag


def test_attack_dir_domain(prepared):
    _, out_df, *_ = prepared
    vals = set(out_df[COL_ATTACK_DIR].dropna().unique().tolist())
    assert vals <= {-1.0, 1.0}


def test_rescaled_within_target_bounds(prepared):
    _, out_df, _, cfg, _ = prepared
    valid = out_df.dropna(subset=[COL_PITCH_X_T, COL_PITCH_Y_T])
    margin = 12.0  # homography spill tolerance (coords are known to leak out)
    assert valid[COL_PITCH_X_T].between(-margin, cfg.target_length_m + margin).all()
    assert valid[COL_PITCH_Y_T].between(-margin, cfg.target_width_m + margin).all()


def test_outputs_written_and_reloadable(prepared):
    *_, paths = prepared
    for key in ("parquet", "jsonl", "meta"):
        assert os.path.exists(paths[key])
    reloaded = pd.read_parquet(paths["parquet"])
    assert COL_STABLE_ID in reloaded.columns
    with open(paths["meta"], encoding="utf-8") as f:
        meta = json.load(f)
    assert "steps" in meta and "resolve_direction" in meta["steps"]
    # frames_prepared.jsonl: one json object per line
    with open(paths["jsonl"], encoding="utf-8") as f:
        first = json.loads(f.readline())
    assert "objects" in first and "in_play" in first
