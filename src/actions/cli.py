"""Command-line interface for the event layer (possession -> SPADL actions).

Mirrors the prerequisites / possession argparse style: an input dir in, an output
dir out, guard thresholds overridable.

    # emit the SPADL actions table
    python -m src.actions detect_actions --in data/gamestate --out data/gamestate

    # loosen the guards on scrappier footage
    python -m src.actions detect_actions --in data/gamestate --out /tmp/out \
        --min-gap-frames 3 --min-ball-travel-m 2.0 --min-carry-m 3.0

    # eyeball check: render segments -> touches -> actions onto the clip
    python -m src.actions review_actions --in data/gamestate --out data/gamestate \
        --video data/raw/2e57b9_0.mp4

Reads the possession stream (``possession_frames.parquet``) and the prepared
tracking (``tracking_prepared.parquet`` + ``prep_meta.json``); writes
``spadl_actions.parquet``, ``spadl_actions_provenance.parquet``,
``ball_aerial.parquet`` and ``actions_meta.json`` (or ``actions_review.mp4`` in
review mode). The possession and prerequisite stages' own outputs are never
touched. Only ``review_actions`` needs ``cv2``.

EXTENSION POINT (feed ``airborne`` back upstream) -- deliberately NOT built
---------------------------------------------------------------------------
``ball_aerial.parquet`` is a **sidecar**: ``frame, airborne, aerial_conf``, one
row per ball frame, joinable onto ``tracking_prepared.parquet`` on ``frame``. It
is written here, by the *event* layer, rather than as two extra columns emitted by
``src.prerequisites``, and that is a deliberate scoping decision, not an oversight.

Two upstream consumers genuinely want it:

* ``prerequisites/ball.py`` (``smooth_ball``) rejects points whose implied GROUND
  speed is impossible -- which is exactly what an airborne ball's stretched
  back-projection looks like. It is currently deleting flight frames as "outliers"
  (on the sample clip, all of frames 104-122 and 150-165 of the aerial pass), and
  then Savitzky-Golay-smoothing across the hole. Knowing a frame is airborne, it
  could hold those positions out as *untrusted* rather than *impossible*, and
  interpolate the ground track across the flight instead of trying to fit it.
* ``prerequisites/deadball.py`` (``synth_dead_ball``) reads "ball near a boundary
  and slow" as a stoppage. A ball in flight over the touchline is neither.

Wiring it in there would make the prerequisites depend on the possession stream
(to know which frames are loose), which today runs *after* them -- so it would
either invert the stage order or force a second prerequisites pass over every
clip. That is a pipeline change, not a feature, and it is out of scope here. The
flag exists, it is persisted, and it is joinable; whoever takes that on starts
from data, not from scratch.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .config import ActionConfig, config_from_prep_meta
from .pipeline import detect_actions
from .source import ZonePossessionSource
from .spadl import BODYPARTS

#: CLI flags that map 1:1 onto ActionConfig fields.
_OVERRIDE_KEYS = (
    "game_id", "period_id",
    "bridge_max_gap_frames", "bridge_max_ball_dist_m",
    "min_gap_frames", "min_ball_travel_m", "min_path_coherence", "max_gap_frames",
    "flight_min_travel_m", "min_carry_m", "max_carry_m", "duel_max_touch_frames",
    "cross_wide_y_frac", "cross_min_x_frac",
    "progressive_min_m", "long_pass_m", "default_bodypart",
    # aerial (airborne-ball) detection
    "aerial_enabled", "aerial_min_run_frames", "aerial_max_run_frames",
    "aerial_min_curvature", "aerial_min_r2", "aerial_min_amplitude_px",
    "aerial_min_speed_ms", "aerial_bbox_min_corr", "aerial_min_path_coherence",
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m src.actions",
        description="Event layer: possession transitions -> SPADL actions.",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser(
        "detect_actions",
        help="emit the SPADL actions table from the possession stream",
    )
    d.add_argument("--in", dest="in_dir", default="data/gamestate",
                   help="dir with possession_frames.parquet + "
                        "tracking_prepared.parquet + prep_meta.json")
    d.add_argument("--out", dest="out_dir", default="data/gamestate",
                   help="output dir for the spadl_actions* artifacts")

    ids = d.add_argument_group("SPADL identifiers")
    ids.add_argument("--game-id", dest="game_id", default=None,
                     help="SPADL game_id (default: the source clip's stem)")
    ids.add_argument("--period-id", dest="period_id", type=int, default=None,
                     help="SPADL period_id, 1..5 (default 1: the clip is one period)")
    ids.add_argument("--bodypart", dest="default_bodypart", default=None,
                     choices=BODYPARTS,
                     help="bodypart for every action (default 'foot'). NOT "
                          "observable without pose data -- see the README.")

    g = d.add_argument_group(
        "gap guards",
        "loose is ~37%% of ball-frames; these stop a blip past the zone radius "
        "from being minted as a pass",
    )
    g.add_argument("--bridge-max-gap-frames", type=int, default=None,
                   help="same-possessor segments this close are ONE touch, so a "
                        "knock-and-chase is a carry not a pass (default 12)")
    g.add_argument("--bridge-max-ball-dist-m", type=float, default=None,
                   help="...and only if the ball stayed this close to the carrier "
                        "throughout (default 10.0)")
    g.add_argument("--min-gap-frames", type=int, default=None,
                   help="loose frames needed for a possessor change to count, "
                        "unless the ball travelled (default 2)")
    g.add_argument("--min-ball-travel-m", type=float, default=None,
                   help="...or this much ball travel (default 1.5)")
    g.add_argument("--min-path-coherence", type=float, default=None,
                   help="straight/polyline ratio below which the ball's path is an "
                        "aimless deflection -> low confidence (default 0.5)")
    g.add_argument("--max-gap-frames", type=int, default=None,
                   help="gaps longer than this are too unobserved to name (default 100)")

    c = d.add_argument_group("classification")
    c.add_argument("--duel-max-touch-frames", type=int, default=None,
                   help="a turnover out of a contested touch this short is flagged "
                        "an unresolved duel (default 2)")
    c.add_argument("--flight-min-travel-m", type=float, default=None,
                   help="ball travel at/above which it was IN FLIGHT: picks "
                        "interception over tackle (default 3.0)")
    c.add_argument("--min-carry-m", type=float, default=None,
                   help="ball movement within a touch needed to call it a dribble; "
                        "below this the ball was parked, not carried (default 2.0)")
    c.add_argument("--max-carry-m", type=float, default=None,
                   help="above this it is a tracking gap, not a carry (default 60.0)")
    c.add_argument("--cross-wide-y-frac", type=float, default=None,
                   help="a cross ORIGINATES within this fraction of the width of a "
                        "touchline (default 0.2)")
    c.add_argument("--cross-min-x-frac", type=float, default=None,
                   help="...and at/beyond this fraction of the length toward the "
                        "target goal (default 0.66)")
    c.add_argument("--progressive-min-m", type=float, default=None,
                   help="metres gained toward goal to call a pass progressive "
                        "(default 5.0)")
    c.add_argument("--long-pass-m", type=float, default=None,
                   help="pass length at/above which it is long (default 25.0)")

    a = d.add_argument_group(
        "aerial (airborne ball)",
        "the homography maps the image to the GROUND plane (z=0), so an airborne "
        "ball's pitch coords are stretched away from the camera. These flag WHICH "
        "frames are untrustworthy -- from the arc the ball traces in img_y. A "
        "HEURISTIC detector, not height recovery.",
    )
    a.add_argument("--no-aerial", dest="aerial_enabled", action="store_false",
                   default=None,
                   help="disable airborne detection entirely; every guard then "
                        "behaves as it did before the flag existed")
    a.add_argument("--aerial-min-run-frames", type=int, default=None,
                   help="usable img_y samples a loose run needs before its "
                        "parabola means anything (default 8)")
    a.add_argument("--aerial-max-run-frames", type=int, default=None,
                   help="loose runs longer than this are DROPPED: over several "
                        "seconds a camera pan traces an arc of its own (default 125)")
    a.add_argument("--aerial-min-curvature", type=float, default=None,
                   help="upward-opening img_y curvature, px/frame^2. img_y "
                        "DECREASES as the ball rises, so a flight is a local "
                        "MINIMUM in img_y (default 0.02)")
    a.add_argument("--aerial-min-r2", type=float, default=None,
                   help="quadratic fit quality floor (default 0.80)")
    a.add_argument("--aerial-min-amplitude-px", type=float, default=None,
                   help="how deep the img_y arc must be to be a flight rather "
                        "than a wobble (default 8.0)")
    a.add_argument("--aerial-min-speed-ms", type=float, default=None,
                   help="apparent ground-speed floor: the main defence against a "
                        "camera pan reading as an arc (default 12.0)")
    a.add_argument("--aerial-bbox-min-corr", type=float, default=None,
                   help="corr(img_y, bbox height) at/above which the shrinking "
                        "box corroborates the arc (default 0.30)")
    a.add_argument("--aerial-min-path-coherence", type=float, default=None,
                   help="coherence floor applied to gaps the ball FLEW across. "
                        "Default 0.0 = bypassed: straightness is a test of the "
                        "GROUND path, and an airborne ball has no trustworthy one")

    r = sub.add_parser(
        "review_actions",
        help="render the actions-review video (segments -> touches -> SPADL actions)",
    )
    r.add_argument("--in", dest="in_dir", default="data/gamestate",
                   help="dir with possession_frames.parquet + "
                        "tracking_prepared.parquet + prep_meta.json")
    r.add_argument("--out", dest="out_dir", default="data/gamestate",
                   help="output dir (default location of actions_review.mp4)")
    r.add_argument("--video", default=None,
                   help="video to draw on (default: data/raw/<meta source>)")
    r.add_argument("--video-out", default=None,
                   help="output path (default: <out>/actions_review.mp4)")
    r.add_argument("--start-frame", type=int, default=0,
                   help="first frame to render (default 0)")
    r.add_argument("--end-frame", type=int, default=None,
                   help="last frame to render (default: all)")
    r.add_argument("--trail", type=int, default=4,
                   help="how many previous actions to fade in behind the active "
                        "one on the minimap (default 4)")
    return ap


def _load_inputs(in_dir: str):
    """The possession stream + the prepared tracking + the prerequisites' meta."""
    tracking_path = os.path.join(in_dir, "tracking_prepared.parquet")
    if not os.path.exists(tracking_path):
        raise SystemExit(
            f"no tracking_prepared.parquet in {in_dir!r}; run "
            f"`python -m src.prerequisites run_prerequisites` first."
        )
    tracking = pd.read_parquet(tracking_path)

    # The possession SOURCE is swappable: this is the only line that names the
    # zone detector. A ball-free source would be constructed here instead, and
    # nothing downstream would change.
    source = ZonePossessionSource.from_dir(in_dir)

    meta_path = os.path.join(in_dir, "prep_meta.json")
    prep_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            prep_meta = json.load(f)
    else:
        print(f"[warn] no prep_meta.json in {in_dir!r}; using default fps/pitch and "
              f"attack_dir=+1 for every team (progression will be unreliable)")
    return source, tracking, prep_meta


def _overrides(args) -> dict:
    return {
        k: getattr(args, k)
        for k in _OVERRIDE_KEYS
        if getattr(args, k, None) is not None
    }


def _default_video(prep_meta: dict) -> str:
    """Where Layer 1's source clip normally lives (from the recorded meta)."""
    source = (prep_meta.get("source_meta", {}) or {}).get("source") or ""
    return os.path.join("data", "raw", os.path.basename(source))


def _run_review(source, tracking, cfg, prep_meta, args) -> None:
    from .pipeline import run_stages
    from .review import render_review

    video_in = args.video or _default_video(prep_meta)
    video_out = args.video_out or os.path.join(args.out_dir, "actions_review.mp4")

    stages = run_stages(source, tracking, cfg)
    info = render_review(
        tracking, stages, cfg, video_in, video_out,
        start_frame=args.start_frame, end_frame=args.end_frame, trail=args.trail,
    )
    print(
        f"[actions] reviewed {info['n_frames_written']} frames: "
        f"{info['n_segments']} segments -> {info['n_touches']} touches -> "
        f"{info['n_actions']} actions"
    )
    print(
        "[actions] white ring = possessor; coloured ring = the actor of the active "
        "action. The minimap carries the action geometry (start->end arrows, with "
        "a fading trail) -- there is no pitch->image homography to draw it on the "
        "video itself."
    )
    print(
        "[actions] the three strips ARE the derivation: SEGMENTS -> TOUCHES "
        "(fewer boundaries = a blip or knock-and-chase coalesced away) -> ACTIONS "
        "(dark = a gap the layer refused to name)."
    )
    print(f"[actions] wrote:\n  {info['video_out']}")


def _run_detect(source, tracking, cfg, args) -> None:
    from .pipeline import run_stages

    stages = run_stages(source, tracking, cfg)
    actions, provenance, meta = stages.actions, stages.provenance, stages.meta

    os.makedirs(args.out_dir, exist_ok=True)
    actions_path = os.path.join(args.out_dir, "spadl_actions.parquet")
    prov_path = os.path.join(args.out_dir, "spadl_actions_provenance.parquet")
    meta_path = os.path.join(args.out_dir, "actions_meta.json")
    # The ball annotation is written as its OWN file rather than as two extra
    # columns on tracking_prepared.parquet, so the prerequisite and possession
    # stages' outputs stay byte-for-byte what they were. Join it on `frame` to
    # recover the columns. See the EXTENSION POINT below.
    aerial_path = os.path.join(args.out_dir, "ball_aerial.parquet")

    actions.to_parquet(actions_path, index=False)
    provenance.to_parquet(prov_path, index=False)
    stages.aerial.to_frame().to_parquet(aerial_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    by_type = "  ".join(
        f"{t}: {n}" for t, n in sorted(meta["actions_by_type"].items())
    )
    print(
        f"[actions] {meta['n_segments']} segments -> {meta['n_touches']} touches "
        f"-> {meta['n_actions']} SPADL actions "
        f"({meta['n_gaps_skipped']} gaps refused)"
    )
    if by_type:
        print(f"[actions] {by_type}")
    print(
        f"[actions] occluded {meta['occluded_pct']:.1f}%  "
        f"low-confidence {meta['low_confidence_pct']:.1f}%  "
        f"mean confidence {meta['mean_confidence']:.2f}"
    )

    aer = meta["aerial"]
    if aer["enabled"]:
        print(
            f"[actions] aerial: {aer['n_aerial_runs']}/{aer['n_loose_runs']} loose "
            f"runs were airborne ({aer['n_full_arcs']} full arcs, "
            f"{aer['n_partial_arcs']} partial), {aer['n_aerial_frames']} ball "
            f"frames flagged"
        )
        if meta["n_aerial_actions"]:
            print(
                f"[actions] {meta['n_aerial_actions']} actions crossed an airborne "
                f"gap ({meta['n_aerial_passes']} passes, "
                f"{meta['n_aerial_crosses']} crosses; mean aerial confidence "
                f"{meta['mean_aerial_conf']:.2f}). Their endpoints come from the "
                f"PLAYERS, and they bypass the ground-path coherence guard."
            )
        print(
            "[actions] airborne is a HEURISTIC flag (the arc the ball traces in "
            "img_y), NOT a height measurement -- the homography only knows z=0."
        )

    if meta["n_duel_candidates"]:
        print(
            f"[actions] {meta['n_duel_candidates']} actions "
            f"({meta['duel_candidate_pct']:.1f}%) came out of a fleeting contested "
            f"touch -- unresolved duels. Duel resolution is a later milestone."
        )
    if meta["skipped_by_reason"]:
        refused = "  ".join(
            f"{r}: {n}" for r, n in sorted(meta["skipped_by_reason"].items())
        )
        print(f"[actions] refused: {refused}")
    print(
        f"[actions] bodypart is NOT observable without pose data -- every action "
        f"is '{cfg.default_bodypart}'."
    )
    print(
        f"[actions] wrote:\n  {actions_path}\n  {prov_path}\n  {aerial_path}\n"
        f"  {meta_path}"
    )


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    source, tracking, prep_meta = _load_inputs(args.in_dir)
    cfg: ActionConfig = config_from_prep_meta(prep_meta, **_overrides(args))

    if args.command == "review_actions":
        _run_review(source, tracking, cfg, prep_meta, args)
    else:
        _run_detect(source, tracking, cfg, args)


if __name__ == "__main__":
    main()
