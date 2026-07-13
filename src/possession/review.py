"""Render a possession-review video — the eyeball check on the detector.

The summary numbers (coverage / clean / duel) tell you *how much* was attributed,
never whether it was attributed to the **right player**. Only watching it does.
This renders ``possession_review.mp4``: the source clip with, per frame,

- the **possessor ringed** in white and labelled with its ``stable_id``;
- every *other* candidate inside R_pz ringed in orange (so a ``contested`` frame
  visibly shows the duel rather than just asserting one);
- the ball marked, with a line to the possessor labelled with ``dist_m``;
- a **state banner** (possession / contested / loose / no_ball), colour-coded;
- a **minimap** in the target (105x68 m) frame -- the only place the R_pz zone can
  be drawn honestly as a circle, because metres are linear there. In the image
  the same zone is a perspective-warped ellipse we have no homography to compute
  post-hoc, so we draw a distance line instead of faking a circle;
- a **timeline strip**: one column per frame, coloured by state, with a cursor.
  The whole clip's possession structure is visible at a glance.

Everything is drawn from the *prepared* table's image-space columns (``img_x`` /
``img_y`` / ``bbox_*``, kept intact by the prerequisite stage) plus this stage's
own per-frame output. The geometry helpers below are pure numpy and unit-tested;
``cv2`` is imported lazily so importing this module never requires it.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    BALL_OBJECT_ID,
    COL_BALL_XTS,
    COL_BALL_YTS,
    COL_PITCH_X_T,
    COL_PITCH_Y_T,
    COL_STABLE_ID,
    PEOPLE_ROLES,
    STATE_CONTESTED,
    STATE_LOOSE,
    STATE_NO_BALL,
    STATE_POSSESSION,
    PossessionConfig,
)
from .zone import ball_to_player_distances

# --- palette (BGR, for cv2) ------------------------------------------------ #
# team colours match src/annotate.py's so the two overlays read the same way.
TEAM_BGR = {0.0: (147, 20, 255), 1.0: (255, 191, 0)}  # pink / deep-sky-blue
UNKNOWN_BGR = (128, 128, 128)
BALL_BGR = (255, 255, 255)
POSSESSOR_BGR = (255, 255, 255)  # white ring on the possessor
RIVAL_BGR = (0, 165, 255)        # orange ring on the other players in the zone

STATE_BGR = {
    STATE_POSSESSION: (80, 200, 80),    # green  — one player, unambiguous
    STATE_CONTESTED: (0, 165, 255),     # orange — a duel, flagged not resolved
    STATE_LOOSE: (190, 190, 190),       # grey   — nobody on it
    STATE_NO_BALL: (60, 60, 190),       # red    — occluded, NOT a stoppage
}


# --------------------------------------------------------------------------- #
# pure geometry (no cv2) — unit-tested
# --------------------------------------------------------------------------- #

def minimap_geometry(
    cfg: PossessionConfig, width: int = 460, pad: int = 16
) -> Tuple[float, int, int, int]:
    """Uniform metres->pixels scale for the minimap.

    Returns ``(scale_px_per_m, width, height, pad)``. The scale is deliberately
    **uniform** in x and y: an anisotropic minimap would render the R_pz zone as
    an ellipse and quietly misrepresent the one thing the minimap exists to show.
    """
    scale = (width - 2 * pad) / cfg.pitch_length_m
    height = int(round(cfg.pitch_width_m * scale)) + 2 * pad
    return scale, width, height, pad


def pitch_to_px(x_m: float, y_m: float, scale: float, pad: int) -> Tuple[int, int]:
    """One target-frame (metres) point -> minimap pixel."""
    return int(round(pad + x_m * scale)), int(round(pad + y_m * scale))


def state_strip(
    states: np.ndarray, width: int, height: int = 14
) -> np.ndarray:
    """The whole clip's states as a width x height BGR strip (one column/frame).

    Frames are resampled onto the strip width, so a 90-minute match and a 30 s
    clip both render legibly.
    """
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(states)
    if n == 0:
        return strip
    # map each output column back to the frame it represents
    idx = np.minimum((np.arange(width) * n) // width, n - 1)
    for i, s in enumerate(states[idx]):
        strip[:, i] = STATE_BGR.get(s, UNKNOWN_BGR)
    return strip


def team_color(team) -> Tuple[int, int, int]:
    """Team colour, tolerant of the null team the tracker sometimes emits."""
    if team is None or (isinstance(team, float) and np.isnan(team)):
        return UNKNOWN_BGR
    return TEAM_BGR.get(float(team), UNKNOWN_BGR)


# --------------------------------------------------------------------------- #
# rendering (needs cv2)
# --------------------------------------------------------------------------- #

def _draw_minimap(cv2, cfg, people, ball_xy, possessor_id, state):
    """Top-down view in the target frame, with an honest R_pz circle."""
    scale, w, h, pad = minimap_geometry(cfg)
    mm = np.full((h, w, 3), 30, dtype=np.uint8)

    # pitch outline + halfway line
    tl = pitch_to_px(0, 0, scale, pad)
    br = pitch_to_px(cfg.pitch_length_m, cfg.pitch_width_m, scale, pad)
    cv2.rectangle(mm, tl, br, (90, 90, 90), 1)
    mid_top = pitch_to_px(cfg.pitch_length_m / 2, 0, scale, pad)
    mid_bot = pitch_to_px(cfg.pitch_length_m / 2, cfg.pitch_width_m, scale, pad)
    cv2.line(mm, mid_top, mid_bot, (90, 90, 90), 1)
    cv2.circle(mm, pitch_to_px(cfg.pitch_length_m / 2, cfg.pitch_width_m / 2,
                               scale, pad), int(9.15 * scale), (90, 90, 90), 1)

    # the possession zone: a true circle, because metres are linear here
    if ball_xy is not None:
        bx, by = pitch_to_px(ball_xy[0], ball_xy[1], scale, pad)
        cv2.circle(mm, (bx, by), max(1, int(round(cfg.r_pz_m * scale))),
                   STATE_BGR.get(state, UNKNOWN_BGR), 1)

    for r in people.itertuples():
        px, py = pitch_to_px(getattr(r, COL_PITCH_X_T), getattr(r, COL_PITCH_Y_T),
                             scale, pad)
        sid = getattr(r, COL_STABLE_ID)
        is_possessor = possessor_id is not None and sid == possessor_id
        cv2.circle(mm, (px, py), 4, team_color(r.team), -1)
        if is_possessor:
            cv2.circle(mm, (px, py), 7, POSSESSOR_BGR, 2)

    if ball_xy is not None:
        bx, by = pitch_to_px(ball_xy[0], ball_xy[1], scale, pad)
        cv2.circle(mm, (bx, by), 3, BALL_BGR, -1)

    return mm


def _draw_banner(cv2, frame, row, cfg):
    """Colour-coded state banner, top-left."""
    state = row["state"]
    color = STATE_BGR.get(state, UNKNOWN_BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (470, 108), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (18, 18), (470, 108), color, 2)

    cv2.putText(frame, state.upper(), (32, 52), font, 0.9, color, 2, cv2.LINE_AA)
    head = f"f={int(row['frame'])}  t={row['time_s']:.2f}s"
    cv2.putText(frame, head, (210, 50), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    if pd.notna(row["possessor_id"]):
        team = row["possessor_team"]
        team_txt = "?" if pd.isna(team) else str(int(team))
        sub = (f"#{int(row['possessor_id'])} (team {team_txt})   "
               f"d={row['dist_m']:.2f}m   in_zone={int(row['n_in_zone'])}")
    elif state == STATE_LOOSE:
        nearest = "-" if pd.isna(row["dist_m"]) else f"{row['dist_m']:.2f}m"
        sub = f"no possessor - nearest {nearest} (R_pz={cfg.r_pz_m:g}m)"
    else:
        sub = "ball not visible (occlusion, not a stoppage)"
    cv2.putText(frame, sub, (32, 88), font, 0.6, (235, 235, 235), 1, cv2.LINE_AA)


def _draw_players(cv2, frame, people, pairs_f, row, cfg):
    """Ring the possessor (white) and any rival inside the zone (orange)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    possessor_id = row["possessor_id"] if pd.notna(row["possessor_id"]) else None
    in_zone = set()
    if pairs_f is not None and len(pairs_f):
        in_zone = set(pairs_f.loc[pairs_f["_in_zone"], COL_STABLE_ID].tolist())

    for r in people.itertuples():
        if pd.isna(r.img_x) or pd.isna(r.img_y):
            continue
        x, y = int(r.img_x), int(r.img_y)
        sid = getattr(r, COL_STABLE_ID)
        cv2.ellipse(frame, (x, y), (16, 6), 0, -30, 240, team_color(r.team), 2)

        if possessor_id is not None and sid == possessor_id:
            cv2.ellipse(frame, (x, y), (24, 10), 0, 0, 360, POSSESSOR_BGR, 3)
            cv2.putText(frame, f"#{int(sid)}", (x - 16, y - 22), font, 0.6,
                        POSSESSOR_BGR, 2, cv2.LINE_AA)
        elif sid in in_zone:
            # inside R_pz but not the nearest: this is what "contested" means
            cv2.ellipse(frame, (x, y), (22, 9), 0, 0, 360, RIVAL_BGR, 2)


def _draw_ball(cv2, frame, ball_row, people, row):
    """Mark the ball and connect it to the possessor with the measured distance."""
    if ball_row is None or pd.isna(ball_row.get("img_x")):
        return  # interpolated/occluded: nothing to point at in image space
    bx, by = int(ball_row["img_x"]), int(ball_row["img_y"])
    cv2.circle(frame, (bx, by), 7, (0, 0, 0), -1)
    cv2.circle(frame, (bx, by), 5, BALL_BGR, -1)

    if pd.isna(row["possessor_id"]):
        return
    holder = people[people[COL_STABLE_ID] == row["possessor_id"]]
    if holder.empty or pd.isna(holder.iloc[0]["img_x"]):
        return
    hx, hy = int(holder.iloc[0]["img_x"]), int(holder.iloc[0]["img_y"])
    color = STATE_BGR.get(row["state"], UNKNOWN_BGR)
    cv2.line(frame, (bx, by), (hx, hy), color, 2, cv2.LINE_AA)
    mx, my = (bx + hx) // 2, (by + hy) // 2
    cv2.putText(frame, f"{row['dist_m']:.1f}m", (mx + 6, my - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def render_review(
    df: pd.DataFrame,
    frames: pd.DataFrame,
    cfg: PossessionConfig,
    video_in: str,
    video_out: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> dict:
    """Render the possession-review video. Returns a small render summary.

    ``df`` is the prepared tracking table (for image-space positions), ``frames``
    the per-frame possession table from this stage. Frames are matched by decode
    order, exactly as Layer 1 numbered them.
    """
    import cv2  # lazy: importing this module must not require the CV stack

    if not os.path.exists(video_in):
        raise SystemExit(f"no video at {video_in!r} to draw on.")

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_in!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(os.path.abspath(video_out)), exist_ok=True)
    writer = cv2.VideoWriter(
        video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    by_frame = frames.set_index("frame", drop=False)  # keep 'frame' for the banner
    states_all = frames.sort_values("frame")["state"].to_numpy()
    all_frames = frames["frame"].to_numpy()

    people_all = df[df["role"].isin(PEOPLE_ROLES)]
    ball_all = df[df["object_id"] == BALL_OBJECT_ID].set_index("frame")
    pairs = ball_to_player_distances(df, cfg)
    pairs_by_frame = dict(tuple(pairs.groupby("frame"))) if len(pairs) else {}
    people_by_frame = dict(tuple(people_all.groupby("frame")))

    strip_h = 14
    n_written, idx = 0, 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        f = idx
        idx += 1
        if f < start_frame:
            continue
        if end_frame is not None and f > end_frame:
            break
        if f not in by_frame.index:
            continue

        row = by_frame.loc[f]
        people = people_by_frame.get(f, people_all.iloc[0:0])
        ball_row = ball_all.loc[f].to_dict() if f in ball_all.index else None
        ball_xy = None
        if ball_row is not None and pd.notna(ball_row.get(COL_BALL_XTS)):
            ball_xy = (ball_row[COL_BALL_XTS], ball_row[COL_BALL_YTS])

        _draw_players(cv2, image, people, pairs_by_frame.get(f), row, cfg)
        _draw_ball(cv2, image, ball_row, people, row)
        _draw_banner(cv2, image, row, cfg)

        # minimap, bottom-right
        mm = _draw_minimap(
            cv2, cfg, people, ball_xy,
            row["possessor_id"] if pd.notna(row["possessor_id"]) else None,
            row["state"],
        )
        mh, mw, _ = mm.shape
        y0, x0 = height - mh - strip_h - 12, width - mw - 18
        roi = image[y0:y0 + mh, x0:x0 + mw]
        cv2.addWeighted(mm, 0.82, roi, 0.18, 0, roi)
        cv2.rectangle(image, (x0, y0), (x0 + mw, y0 + mh), (110, 110, 110), 1)

        # timeline strip along the bottom + a cursor at the current frame
        strip = state_strip(states_all, width, strip_h)
        cursor = int(np.searchsorted(all_frames, f) * width / max(len(all_frames), 1))
        cv2.line(strip, (cursor, 0), (cursor, strip_h), (255, 255, 255), 1)
        image[height - strip_h:height] = strip

        writer.write(image)
        n_written += 1

    cap.release()
    writer.release()
    return dict(
        video_out=video_out, n_frames_written=n_written,
        width=width, height=height, fps=float(fps), r_pz_m=cfg.r_pz_m,
    )
