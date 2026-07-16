"""Render an actions-review video -- the eyeball check on the event layer.

The counts (33 actions, 10 refused) tell you *how many* events were named, never
whether they were the **right** events. Only watching it does. This renders
``actions_review.mp4``: the source clip with, per frame,

- the **possessor ringed** in white and labelled with its ``stable_id``;
- the **active action** called out -- its type, result, actor and confidence --
  and the actor ringed in that action's colour;
- a **minimap** in the target (105x68 m) frame carrying the action geometry: the
  active action drawn **start -> end as an arrow**, with the previous few actions
  fading behind it, so the passing chain is visible as a chain;
- **three stacked timeline strips** -- ``SEGMENTS``, ``TOUCHES``, ``ACTIONS`` --
  which is the whole derivation of this layer in one picture: raw possession runs,
  the same runs after coalescing, and the events that came out of the gaps between
  them. Where the TOUCHES strip has *fewer* boundaries than SEGMENTS, that is a
  blip or a knock-and-chase being absorbed; where ACTIONS is dark, that is a gap
  the layer **refused** to name.

Why the arrows live on the minimap and not on the video
-------------------------------------------------------
An action is a statement about two points in *pitch metres* (release and
reception). Drawing it onto the image would need a pitch->image homography, and we
have none post-hoc -- the prepared table keeps image space and pitch space side by
side but no map between them. So the minimap, where metres are linear and an arrow
is honest, is where the geometry goes. The video itself carries only what image
space can actually support: who the possessor is, and who is acting.

``cv2`` is imported lazily, so importing this module never requires the CV stack.
The pure-geometry helpers below are unit-tested without it.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Pure drawing primitives, shared with the possession review so the two overlays
# read the same way. These are pitch-drawing helpers, NOT possession logic -- the
# event layer's independence from the zone detector (see source.py) is about where
# the possessor comes from, and is untouched by sharing a minimap.
from ..possession.review import (
    BALL_BGR,
    POSSESSOR_BGR,
    UNKNOWN_BGR,
    minimap_geometry,
    pitch_to_px,
    team_color,
)
from .config import (
    BALL_OBJECT_ID,
    COL_BALL_XTS,
    COL_BALL_YTS,
    COL_PITCH_X_T,
    COL_PITCH_Y_T,
    COL_STABLE_ID,
    PEOPLE_ROLES,
    ActionConfig,
)
from .pipeline import ActionStages

# --- palette (BGR, for cv2) ------------------------------------------------- #
# One colour per SPADL action type. Moves are warm/green, the two halves of a
# turnover are red (lost) and orange (won), so a change of possession reads as a
# colour flip on the ACTIONS strip.
ACTION_BGR: Dict[str, Tuple[int, int, int]] = {
    "pass": (90, 210, 90),          # green
    "cross": (220, 200, 60),        # cyan-ish
    "dribble": (60, 220, 235),      # yellow
    "interception": (0, 165, 255),  # orange -- the ball was WON
    "tackle": (40, 120, 255),       # deep orange -- also won, off a settled touch
    "bad_touch": (150, 90, 220),    # purple -- the ball was LOST
}
IDLE_BGR = (38, 38, 38)             # a gap the layer named nothing for

# Segments/touches strips: shaded by team, alternating light/dark per run so the
# BOUNDARIES are visible (which is the entire point of showing both strips).
SEG_ALT = (0.62, 1.0)


# --------------------------------------------------------------------------- #
# pure geometry (no cv2) -- unit-tested
# --------------------------------------------------------------------------- #

def spans_to_lookup(
    spans: List[Tuple[int, int, object]], frames: np.ndarray
) -> List[Optional[object]]:
    """Map every frame to the payload of the span containing it (or ``None``).

    ``spans`` is ``[(start_frame, end_frame, payload), ...]``, inclusive on both
    ends. Later spans win on overlap, which never happens for segments or touches
    (they are disjoint by construction) but can for actions: a turnover's two rows
    share a timestamp, and a carry can end on the same frame its touch's pass
    begins. Showing the *last* one is the right call -- it is the one that ends the
    touch, and therefore the one that explains what happens next.
    """
    out: List[Optional[object]] = [None] * len(frames)
    index = {int(f): i for i, f in enumerate(frames)}
    for start, end, payload in spans:
        for f in range(int(start), int(end) + 1):
            i = index.get(f)
            if i is not None:
                out[i] = payload
    return out


def _shade(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


def run_strip(
    payloads: List[Optional[object]],
    width: int,
    height: int,
    color_of,
) -> np.ndarray:
    """A width x height BGR strip: one column per frame, coloured by payload.

    Frames are resampled onto the strip width, so a 90-minute match and a 30 s clip
    both render legibly. ``color_of(payload)`` returns the BGR for a frame.
    """
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(payloads)
    if n == 0:
        return strip
    idx = np.minimum((np.arange(width) * n) // width, n - 1)
    for col, i in enumerate(idx):
        strip[:, col] = color_of(payloads[int(i)])
    return strip


def segment_color(payload) -> Tuple[int, int, int]:
    """Colour a SEGMENTS/TOUCHES column: team hue, alternating shade per run.

    The alternating shade is what makes the *boundaries* visible -- and the
    boundaries are the whole point of showing SEGMENTS and TOUCHES as two strips:
    where TOUCHES has fewer of them, a blip or a knock-and-chase was coalesced away.
    """
    if payload is None:
        return IDLE_BGR
    run_id, team, _player = payload
    return _shade(team_color(team), SEG_ALT[int(run_id) % 2])


def action_color(payload) -> Tuple[int, int, int]:
    """Colour an ACTIONS column by SPADL action type."""
    if payload is None:
        return IDLE_BGR
    return ACTION_BGR.get(str(payload[1]), UNKNOWN_BGR)


# --------------------------------------------------------------------------- #
# rendering (needs cv2)
# --------------------------------------------------------------------------- #

def _draw_minimap(cv2, cfg: ActionConfig, people, ball_xy, possessor_id,
                  recent_actions):
    """Top-down view carrying the ACTION geometry: arrows from start to end.

    ``recent_actions`` is oldest-first; the last is the active one and is drawn
    solid, the rest fade. The chain of arrows IS the passing move.
    """
    scale, w, h, pad = minimap_geometry(cfg)
    mm = np.full((h, w, 3), 30, dtype=np.uint8)

    tl = pitch_to_px(0, 0, scale, pad)
    br = pitch_to_px(cfg.pitch_length_m, cfg.pitch_width_m, scale, pad)
    cv2.rectangle(mm, tl, br, (90, 90, 90), 1)
    mid_top = pitch_to_px(cfg.pitch_length_m / 2, 0, scale, pad)
    mid_bot = pitch_to_px(cfg.pitch_length_m / 2, cfg.pitch_width_m, scale, pad)
    cv2.line(mm, mid_top, mid_bot, (90, 90, 90), 1)
    cv2.circle(mm, pitch_to_px(cfg.pitch_length_m / 2, cfg.pitch_width_m / 2,
                               scale, pad), int(9.15 * scale), (90, 90, 90), 1)

    # the two goal mouths -- the geometry progression is measured against
    for gx in (0.0, cfg.pitch_length_m):
        g_top = pitch_to_px(gx, cfg.pitch_width_m / 2 - 3.66, scale, pad)
        g_bot = pitch_to_px(gx, cfg.pitch_width_m / 2 + 3.66, scale, pad)
        cv2.line(mm, g_top, g_bot, (200, 200, 200), 2)

    # players first, so the action arrows sit on top of them
    for r in people.itertuples():
        x_t, y_t = getattr(r, COL_PITCH_X_T), getattr(r, COL_PITCH_Y_T)
        # Failed-homography frames carry NaN target coords -- no minimap point.
        if pd.isna(x_t) or pd.isna(y_t):
            continue
        px, py = pitch_to_px(x_t, y_t, scale, pad)
        cv2.circle(mm, (px, py), 4, team_color(r.team), -1)
        if possessor_id is not None and getattr(r, COL_STABLE_ID) == possessor_id:
            cv2.circle(mm, (px, py), 7, POSSESSOR_BGR, 2)

    n = len(recent_actions)
    for i, act in enumerate(recent_actions):
        is_active = (i == n - 1)
        color = ACTION_BGR.get(act["type_name"], UNKNOWN_BGR)
        if not is_active:
            # fade the trail: older actions dimmer
            color = _shade(color, 0.30 + 0.25 * (i / max(n - 1, 1)))
        p0 = pitch_to_px(act["start_x"], act["start_y"], scale, pad)
        p1 = pitch_to_px(act["end_x"], act["end_y"], scale, pad)

        if (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 < 9:
            # interception / tackle: won ON the spot, so there is no arrow to draw
            cv2.circle(mm, p1, 6 if is_active else 4, color, 2 if is_active else 1)
        else:
            cv2.arrowedLine(mm, p0, p1, color, 2 if is_active else 1,
                            cv2.LINE_AA, tipLength=0.18)
        if is_active:
            cv2.circle(mm, p0, 3, color, -1)

    if ball_xy is not None:
        bx, by = pitch_to_px(ball_xy[0], ball_xy[1], scale, pad)
        cv2.circle(mm, (bx, by), 3, BALL_BGR, -1)

    return mm


def _draw_banner(cv2, image, frame_no, time_s, seg, touch, action):
    """Segment / touch / action, top-left. The three stages, on every frame.

    ``action`` is a row of the actions table joined to its provenance, so the
    confidence and occlusion flags are right there on it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (
        ACTION_BGR.get(action["type_name"], UNKNOWN_BGR) if action is not None
        else (150, 150, 150)
    )

    overlay = image.copy()
    cv2.rectangle(overlay, (18, 18), (560, 146), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    cv2.rectangle(image, (18, 18), (560, 146), color, 2)

    if action is not None:
        head = f"{action['type_name'].upper()}  {action['result_name']}"
    else:
        head = "-- no action --"
    # The header gets the whole first row to itself: "INTERCEPTION success" is
    # wide enough to run into anything sharing the line with it.
    cv2.putText(image, head, (32, 52), font, 0.8, color, 2, cv2.LINE_AA)

    if action is not None:
        sub = (f"#{int(action['player_id'])} (team {int(action['team_id'])})   "
               f"conf {action['confidence']:.2f}"
               f"{'   OCCLUDED' if action['occluded'] else ''}"
               f"{'   DUEL?' if action['duel_candidate'] else ''}")
        cv2.putText(image, sub, (32, 84), font, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
        det = (f"{action['direction']} / {action['length_class']}   "
               f"ball {action['ball_travel_m']:.1f}m  "
               f"coherence {action['path_coherence']:.2f}")
        cv2.putText(image, det, (32, 110), font, 0.5, (190, 190, 190), 1, cv2.LINE_AA)

    seg_txt = "-" if seg is None else f"#{seg[0]} (player {seg[2]})"
    touch_txt = "-" if touch is None else f"#{touch[0]} (player {touch[2]})"
    cv2.putText(image, f"segment {seg_txt}   touch {touch_txt}", (32, 136),
                font, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(image, f"f={frame_no}  t={time_s:.2f}s", (416, 136), font, 0.5,
                (215, 215, 215), 1, cv2.LINE_AA)


def _draw_players(cv2, image, people, possessor_id, actor_id, action):
    """Ring the possessor (white); ring the ACTOR in its action's colour."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    act_color = (
        ACTION_BGR.get(action["type_name"], UNKNOWN_BGR) if action is not None
        else UNKNOWN_BGR
    )
    for r in people.itertuples():
        if pd.isna(r.img_x) or pd.isna(r.img_y):
            continue
        x, y = int(r.img_x), int(r.img_y)
        sid = getattr(r, COL_STABLE_ID)
        cv2.ellipse(image, (x, y), (16, 6), 0, -30, 240, team_color(r.team), 2)

        if possessor_id is not None and sid == possessor_id:
            cv2.ellipse(image, (x, y), (24, 10), 0, 0, 360, POSSESSOR_BGR, 3)
            cv2.putText(image, f"#{int(sid)}", (x - 16, y - 22), font, 0.6,
                        POSSESSOR_BGR, 2, cv2.LINE_AA)
        if actor_id is not None and sid == actor_id and action is not None:
            cv2.ellipse(image, (x, y), (30, 13), 0, 0, 360, act_color, 2)
            cv2.putText(image, action["type_name"], (x - 24, y + 32), font, 0.5,
                        act_color, 2, cv2.LINE_AA)


def _legend(cv2, image, width):
    """What the colours mean. An overlay nobody can read is not a check."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x0, y0 = width - 210, 22
    overlay = image.copy()
    cv2.rectangle(overlay, (x0 - 12, y0 - 10),
                  (width - 18, y0 + 20 * len(ACTION_BGR) + 8), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    for i, (name, color) in enumerate(ACTION_BGR.items()):
        y = y0 + 20 * i + 8
        cv2.rectangle(image, (x0 - 4, y - 8), (x0 + 8, y + 2), color, -1)
        cv2.putText(image, name, (x0 + 16, y + 2), font, 0.45, (225, 225, 225), 1,
                    cv2.LINE_AA)


def render_review(
    df: pd.DataFrame,
    stages: ActionStages,
    cfg: ActionConfig,
    video_in: str,
    video_out: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    trail: int = 4,
) -> dict:
    """Render the actions-review video. Returns a small render summary.

    ``df`` is the prepared tracking table (for image-space positions), ``stages``
    the full derivation from :func:`~src.actions.pipeline.run_stages` -- segments,
    touches and actions together, so the video shows how each became the next.
    Frames are matched by decode order, exactly as Layer 1 numbered them.
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

    frames_tbl = stages.frames.sort_values("frame")
    all_frames = frames_tbl["frame"].to_numpy()
    time_by_frame = dict(zip(frames_tbl["frame"], frames_tbl["time_s"]))

    # --- the three strips: segments -> touches -> actions ------------------- #
    seg_spans = [
        (int(s.start_frame), int(s.end_frame),
         (int(s.segment_id), s.team, int(s.possessor_id)))
        for s in stages.segments.itertuples()
    ]
    touch_spans = [
        (t.start_frame, t.end_frame, (t.touch_id, t.team, t.possessor_id))
        for t in stages.touches
    ]
    merged = stages.provenance.merge(stages.actions, on="action_id")
    act_spans = [
        (int(a.start_frame), int(a.end_frame), (int(a.action_id), a.type_name))
        for a in merged.itertuples()
    ]

    seg_by_frame = spans_to_lookup(seg_spans, all_frames)
    touch_by_frame = spans_to_lookup(touch_spans, all_frames)
    act_by_frame = spans_to_lookup(act_spans, all_frames)

    strip_h = 12
    strips = [
        ("SEGMENTS", run_strip(seg_by_frame, width, strip_h, segment_color)),
        ("TOUCHES", run_strip(touch_by_frame, width, strip_h, segment_color)),
        ("ACTIONS", run_strip(act_by_frame, width, strip_h, action_color)),
    ]
    strip_block = strip_h * len(strips)

    actions_by_id = {int(r.action_id): r._asdict() for r in merged.itertuples()}
    possessor_by_frame = dict(
        zip(frames_tbl["frame"], frames_tbl["possessor_id"])
    )

    people_all = df[df["role"].isin(PEOPLE_ROLES)]
    ball_all = df[df["object_id"] == BALL_OBJECT_ID].set_index("frame")
    people_by_frame = dict(tuple(people_all.groupby("frame")))

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
        i = int(np.searchsorted(all_frames, f))
        if i >= len(all_frames) or all_frames[i] != f:
            continue

        people = people_by_frame.get(f, people_all.iloc[0:0])
        ball_xy = None
        if f in ball_all.index:
            brow = ball_all.loc[f]
            if pd.notna(brow[COL_BALL_XTS]):
                ball_xy = (brow[COL_BALL_XTS], brow[COL_BALL_YTS])

        pid = possessor_by_frame.get(f)
        possessor_id = None if pd.isna(pid) else int(pid)

        act_here = act_by_frame[i]
        action = actions_by_id[act_here[0]] if act_here is not None else None
        actor_id = None if action is None else int(action["player_id"])

        # the action chain so far: the active one plus a fading trail behind it
        upto = act_here[0] if act_here is not None else -1
        if act_here is None:
            # between actions: keep showing the chain that led here
            earlier = [a for a in act_spans if a[1] < f]
            upto = earlier[-1][2][0] if earlier else -1
        recent = [
            actions_by_id[j]
            for j in range(max(0, upto - trail + 1), upto + 1)
            if j in actions_by_id
        ]

        _draw_players(cv2, image, people, possessor_id, actor_id, action)
        _draw_banner(
            cv2, image, f, float(time_by_frame.get(f, f / cfg.fps)),
            seg_by_frame[i], touch_by_frame[i], action,
        )
        _legend(cv2, image, width)

        mm = _draw_minimap(cv2, cfg, people, ball_xy, possessor_id, recent)
        mh, mw, _ = mm.shape
        y0, x0 = height - mh - strip_block - 12, width - mw - 18
        roi = image[y0:y0 + mh, x0:x0 + mw]
        cv2.addWeighted(mm, 0.85, roi, 0.15, 0, roi)
        cv2.rectangle(image, (x0, y0), (x0 + mw, y0 + mh), (110, 110, 110), 1)

        # the three strips, stacked, with a shared cursor
        cursor = int(i * width / max(len(all_frames), 1))
        for k, (label, strip) in enumerate(strips):
            band = strip.copy()
            cv2.line(band, (cursor, 0), (cursor, strip_h), (255, 255, 255), 1)
            top = height - strip_block + k * strip_h
            image[top:top + strip_h] = band
            cv2.putText(image, label, (6, top + strip_h - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1,
                        cv2.LINE_AA)

        writer.write(image)
        n_written += 1

    cap.release()
    writer.release()
    return dict(
        video_out=video_out, n_frames_written=n_written,
        width=width, height=height, fps=float(fps),
        n_segments=len(seg_spans), n_touches=len(touch_spans),
        n_actions=len(act_spans),
    )
