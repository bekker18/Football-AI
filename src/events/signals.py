"""Per-frame ball-free attacking signals.

For each frame and team, in the team's *attacking-normalized* frame (target goal
at +x, via the prerequisites' ``attack_dir``), we measure how far and how heavily
the team has committed into the opponent's half: players in the final third,
players in the box, and the deepest attacker. A single per-frame value score is
the max over the two teams — "how threatening is the most threatening team right
now" — which the window emitter thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prerequisites import normalize_to_attack

from .config import (
    COL_ATTACK_DIR,
    COL_PITCH_X_T,
    COL_PITCH_Y_T,
    EventConfig,
    PEOPLE_ROLES,
)

TEAMS = (0.0, 1.0)


def attacking_signals(df: pd.DataFrame, cfg: EventConfig) -> pd.DataFrame:
    """One row per frame: the most-threatening team and its value score.

    Columns: ``frame, time_s, attacking_team, value_score, n_final_third, n_box,
    deepest_frac, n_players``. Frames with no usable team signal get value 0.
    Requires the prepared columns ``attack_dir`` and ``pitch_x_t_m`` /
    ``pitch_y_t_m``.
    """
    for col in (COL_ATTACK_DIR, COL_PITCH_X_T, COL_PITCH_Y_T):
        if col not in df.columns:
            raise KeyError(
                f"events layer needs prepared column {col!r}; run "
                f"src.prerequisites first."
            )

    all_frames = np.sort(df["frame"].unique())
    time_by_frame = df.groupby("frame")["time_s"].first() if "time_s" in df else None

    people = df[
        df["role"].isin(PEOPLE_ROLES)
        & df["team"].isin(TEAMS)
        & df[COL_ATTACK_DIR].notna()
        & df[COL_PITCH_X_T].notna()
    ].copy()

    per_team = _per_frame_team(people, cfg) if len(people) else _empty_team_table()

    # keep, per frame, only the team with the higher value score
    per_team = per_team.sort_values(["frame", "value"], kind="stable")
    best = per_team.groupby("frame", as_index=False).last()

    out = pd.DataFrame({"frame": all_frames})
    out = out.merge(best, on="frame", how="left")
    out["value_score"] = out["value"].fillna(0.0)
    out["attacking_team"] = out["team"]
    for c in ("n_final_third", "n_box", "n_players"):
        out[c] = out[c].fillna(0).astype(int)
    out["deepest_frac"] = out["deepest_frac"].fillna(0.0)

    if time_by_frame is not None:
        out["time_s"] = out["frame"].map(time_by_frame)
    else:
        out["time_s"] = out["frame"] / cfg.fps

    return out[[
        "frame", "time_s", "attacking_team", "value_score",
        "n_final_third", "n_box", "deepest_frac", "n_players",
    ]]


def _per_frame_team(people: pd.DataFrame, cfg: EventConfig) -> pd.DataFrame:
    """Aggregate the attacking signal per (frame, team)."""
    norm_x = normalize_to_attack(
        people[COL_PITCH_X_T].to_numpy(float),
        people[COL_ATTACK_DIR].to_numpy(float),
        cfg.pitch_length_m,
    )
    people = people.assign(
        _norm_x=norm_x,
        _in_third=(norm_x >= cfg.final_third_x()).astype(int),
        _in_box=(norm_x >= cfg.box_x()).astype(int),
    )
    g = people.groupby(["frame", "team"])
    agg = g.agg(
        n_players=("_norm_x", "size"),
        n_final_third=("_in_third", "sum"),
        n_box=("_in_box", "sum"),
        deepest_x=("_norm_x", "max"),
    ).reset_index()

    agg["deepest_frac"] = (agg["deepest_x"] / cfg.pitch_length_m).clip(0.0, 1.0)
    raw = (
        cfg.w_third * agg["n_final_third"]
        + cfg.w_box * agg["n_box"]
        + cfg.w_depth * agg["deepest_frac"]
    )
    # a frame we can't trust (too few players tracked for this team) scores 0
    trusted = agg["n_players"] >= cfg.min_players_for_signal
    agg["value"] = np.where(trusted, raw, 0.0)
    return agg


def _empty_team_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["frame", "team", "n_players", "n_final_third", "n_box",
                 "deepest_x", "deepest_frac", "value"]
    )
