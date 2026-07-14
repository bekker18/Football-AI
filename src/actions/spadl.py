"""The SPADL vocabulary and table contract, mirrored from ``socceraction``.

This module is the **single place** that knows what SPADL is. It exists so the
event layer can serialize a schema-valid actions table *without importing
socceraction at runtime* (that library pulls in pandera/xgboost; the stage
itself only needs pandas/numpy, like every other stage here).

Mirroring a vocabulary is a promise that can rot, so it is pinned by a test:
``tests/test_actions_spadl.py`` asserts these lists are **identical** to
``socceraction.spadl.config.actiontypes / results / bodyparts`` and that
:data:`SPADL_COLUMNS` matches ``socceraction.spadl.schema.SPADLSchema``. If
socceraction ever changes its enums, that test fails rather than us silently
emitting a table it can no longer load.

The schema (socceraction 1.5.3, ``SPADLSchema``) in one paragraph: ``strict =
True``, so the actions table may contain **only** the columns below -- any extra
column raises. That is why our confidence/occluded flags live in a *separate*
provenance table keyed by ``action_id`` (see :mod:`src.actions.emit`) rather
than as extra columns here. Coordinates are validated as ``0 <= x <= 105`` and
``0 <= y <= 68``, so they must be clipped; ``period_id`` is ``1..5``;
``time_seconds`` is ``>= 0``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# --- the SPADL enums (socceraction.spadl.config) ------------------------- #
# Order is load-bearing: the *_id columns are indices into these lists.
ACTIONTYPES: List[str] = [
    "pass",              # 0
    "cross",             # 1
    "throw_in",          # 2
    "freekick_crossed",  # 3
    "freekick_short",    # 4
    "corner_crossed",    # 5
    "corner_short",      # 6
    "take_on",           # 7
    "foul",              # 8
    "tackle",            # 9
    "interception",      # 10
    "shot",              # 11
    "shot_penalty",      # 12
    "shot_freekick",     # 13
    "keeper_save",       # 14
    "keeper_claim",      # 15
    "keeper_punch",      # 16
    "keeper_pick_up",    # 17
    "clearance",         # 18
    "bad_touch",         # 19
    "non_action",        # 20
    "dribble",           # 21
    "goalkick",          # 22
]

RESULTS: List[str] = [
    "fail",        # 0
    "success",     # 1
    "offside",     # 2
    "owngoal",     # 3
    "yellow_card", # 4
    "red_card",    # 5
]

BODYPARTS: List[str] = [
    "foot",        # 0
    "head",        # 1
    "other",       # 2
    "head/other",  # 3
    "foot_left",   # 4
    "foot_right",  # 5
]

# SPADL's default pitch -- the same 105x68 the prerequisites rescale INTO, which
# is why this layer needs no further rescale (see the coordinate contract in
# config.py).
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0

# --- the action types this milestone is allowed to emit ------------------- #
# Shots, set pieces and duel resolution are deliberately out of scope; the
# pipeline asserts the emitted table stays inside this set, so a future stage
# adding shots has to widen it consciously rather than by accident.
EMITTED_ACTIONTYPES = ("pass", "cross", "dribble", "interception", "tackle", "bad_touch")


def type_id(name: str) -> int:
    """SPADL ``type_id`` for an action-type name."""
    return ACTIONTYPES.index(name)


def result_id(name: str) -> int:
    """SPADL ``result_id`` for a result name."""
    return RESULTS.index(name)


def bodypart_id(name: str) -> int:
    """SPADL ``bodypart_id`` for a bodypart name."""
    return BODYPARTS.index(name)


# --- the table contract --------------------------------------------------- #
# Required by SPADLSchema, in socceraction's own declaration order. The three
# *_name columns are Optional in the schema; we emit them because they cost
# nothing and make the parquet readable without a lookup table.
SPADL_REQUIRED_COLUMNS: List[str] = [
    "game_id",
    "original_event_id",
    "action_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "bodypart_id",
    "type_id",
    "result_id",
]

SPADL_OPTIONAL_COLUMNS: List[str] = ["bodypart_name", "type_name", "result_name"]

SPADL_COLUMNS: List[str] = SPADL_REQUIRED_COLUMNS + SPADL_OPTIONAL_COLUMNS


def clip_x(x: np.ndarray | pd.Series | float) -> np.ndarray:
    """Clip an x coordinate into SPADL's ``[0, field_length]``.

    Tracking coordinates come from a homography and legitimately land a metre or
    two off the pitch (the prepared clip has players at ``y = -1.27`` and ``y =
    70.18``). SPADLSchema rejects those outright, so every coordinate is clipped
    on the way out. The count of clipped values is reported in the stage meta --
    a large count means the homography, not the event layer, needs looking at.
    """
    return np.clip(np.asarray(x, dtype=float), 0.0, FIELD_LENGTH_M)


def clip_y(y: np.ndarray | pd.Series | float) -> np.ndarray:
    """Clip a y coordinate into SPADL's ``[0, field_width]``. See :func:`clip_x`."""
    return np.clip(np.asarray(y, dtype=float), 0.0, FIELD_WIDTH_M)


def add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Fill ``type_name`` / ``result_name`` / ``bodypart_name`` from the id columns.

    The equivalent of ``socceraction.spadl.add_names``, done locally so the
    stage has no runtime dependency on it.
    """
    out = actions.copy()
    out["type_name"] = [ACTIONTYPES[i] for i in out["type_id"]]
    out["result_name"] = [RESULTS[i] for i in out["result_id"]]
    out["bodypart_name"] = [BODYPARTS[i] for i in out["bodypart_id"]]
    return out


def empty_actions() -> pd.DataFrame:
    """A correctly typed, empty SPADL table (single-segment clips emit this)."""
    dtypes = {
        "game_id": object,
        "original_event_id": object,
        "action_id": "int64",
        "period_id": "int64",
        "time_seconds": "float64",
        "team_id": "int64",
        "player_id": "int64",
        "start_x": "float64",
        "start_y": "float64",
        "end_x": "float64",
        "end_y": "float64",
        "bodypart_id": "int64",
        "type_id": "int64",
        "result_id": "int64",
        "bodypart_name": object,
        "type_name": object,
        "result_name": object,
    }
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})[SPADL_COLUMNS]


def enforce_schema(actions: pd.DataFrame) -> pd.DataFrame:
    """Coerce a built actions frame to the SPADL dtypes and column order.

    Cheap local equivalent of what ``SPADLSchema.validate(coerce=True)`` would
    do, so the parquet is already conformant before socceraction ever sees it.
    Raises if a coordinate escaped clipping or an out-of-scope action type crept
    in -- both are bugs in this layer, not bad input.
    """
    if actions.empty:
        return empty_actions()

    out = actions.copy()
    for col in ("action_id", "period_id", "bodypart_id", "type_id", "result_id"):
        out[col] = out[col].astype("int64")
    for col in ("team_id", "player_id"):
        out[col] = out[col].astype("int64")
    for col in ("time_seconds", "start_x", "start_y", "end_x", "end_y"):
        out[col] = out[col].astype("float64")

    bad_type = set(out["type_id"]) - {type_id(t) for t in EMITTED_ACTIONTYPES}
    if bad_type:
        names = sorted(ACTIONTYPES[i] for i in bad_type)
        raise ValueError(
            f"milestone 1 must not emit {names}; shots, set pieces and duel "
            f"resolution are out of scope (see EMITTED_ACTIONTYPES)."
        )

    for col, hi in (("start_x", FIELD_LENGTH_M), ("end_x", FIELD_LENGTH_M),
                    ("start_y", FIELD_WIDTH_M), ("end_y", FIELD_WIDTH_M)):
        v = out[col].to_numpy()
        if np.isnan(v).any() or (v < 0).any() or (v > hi).any():
            raise ValueError(f"{col} outside SPADL's [0, {hi}] after clipping")
    if (out["time_seconds"].to_numpy() < 0).any():
        raise ValueError("time_seconds must be >= 0 (SPADLSchema)")
    if not out["period_id"].between(1, 5).all():
        raise ValueError("period_id must be in 1..5 (SPADLSchema)")

    return out[SPADL_COLUMNS].reset_index(drop=True)
