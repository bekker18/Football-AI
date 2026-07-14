"""Airborne-ball detection: which ball frames are mid-flight, and how sure are we.

Why this exists
---------------
The homography maps the image onto the **ground plane (z=0)**. Every pitch
coordinate this project produces is therefore the answer to "where would this
pixel be *if the thing in it were lying on the grass*". For a ball on the grass
that is right. For a ball **in the air** it is wrong in a specific, systematic
way: the back-projected point is stretched **away from the camera** along the
viewing ray, by an amount that grows with the ball's height. There is no height
channel anywhere in the pipeline to correct it with.

So an aerial pass is currently recorded as a *flat, distorted ground track*: the
ball appears to accelerate away, curve, and decelerate back, none of which
happened. Downstream, that fake ground path sets pass endpoints, fails
straightness tests written for rolling balls, and feeds xT/VAEP.

We cannot recover the height (see the EXTENSION POINT at the bottom). What we
*can* do is **notice** the ball is airborne, so that everything downstream can
refuse to trust its ground coordinates. That is all this module does.

The signal
----------
The camera looks *down* at the pitch, so a ball that goes up moves **up the
image**, and image-y **decreases**. A ball that rises and falls therefore traces
a **local MINIMUM in img_y** — an upward-opening parabola. That vertical arc is
visible even though the horizontal geometry is ruined, because it lives in image
space, where no homography has been applied.

Concretely, per loose run (the ball attributed to nobody):

1. **Clean img_y robustly** (:func:`_clean_img_y`) — a single bad detection must
   not be able to break the fit.
2. **Fit a quadratic** to img_y over the run. Airborne when the curvature opens
   upward (a local minimum), the vertex is *observed inside* the run, and the fit
   is good (R^2).
3. **Corroborate** with elevated-but-smooth apparent ground speed and with the
   bbox height varying coherently across the arc (the ball is further away near
   the apex, so its box is smaller). These raise confidence — and the speed floor
   also *gates*, because it is what separates a real flight from a camera tilt.
4. **Partial arcs** (ball already up on entry, or only the descent visible) emit
   ``airborne=True`` with LOW confidence rather than forcing a full parabola onto
   half an arc.

Why ``ball_outlier`` is NOT used to clean img_y
-----------------------------------------------
It is tempting to reuse the prerequisites' ``ball_outlier`` flag as the spike
filter. **It would be circular, and it would delete the very thing we are looking
for.** ``ball_outlier`` is a *pitch-space* speed gate: it rejects points whose
implied ground speed is impossible. But an airborne ball's ground projection is
stretched away from the camera, which is *exactly* what makes its apparent ground
speed impossible — so the gate rejects airborne balls **because** they are
airborne.

On the sample clip this is not hypothetical: across the aerial pass at frames
104-166, frames 104-122 and 150-165 are all ``ball_outlier=True`` while their
``img_y`` is perfectly smooth. Cleaning img_y with that flag would throw away two
thirds of the arc and keep only its middle.

So the spike filter here is a **robust filter in image space** (median + MAD),
which rejects the genuine image-space glitch (frame 136: ``img_y=449`` amid
``~325``) and keeps the smooth ramps. ``ball_outlier`` is still *reported* per
run, because "these frames were rejected as impossible ground speed AND they are
airborne" is a coherent story, not a contradiction.

Honesty about what this is
--------------------------
This is a **heuristic, single-camera detector, not height recovery**. It answers
"was the ball probably off the ground here?" with a confidence, and nothing more.
It does not estimate height, and it cannot: a single broadcast camera with a
ground-plane homography has no depth channel. Camera pan/tilt also moves img_y,
which is why airborne requires the ball to be *loose*, the run to be bounded, and
the speed/bbox evidence to agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    BALL_OBJECT_ID,
    COL_BALL_OUTLIER,
    COL_BALL_SPEED,
    COL_BBOX_Y1,
    COL_BBOX_Y2,
    COL_IMG_Y,
    ActionConfig,
)

#: Columns of the ball annotation this module produces.
AERIAL_COLUMNS = ["frame", "airborne", "aerial_conf"]

#: Why a run was (not) called airborne. Reported per run in the stage meta.
ARC_FULL = "arc"          # rose and fell: the vertex was observed inside the run
ARC_PARTIAL = "partial"   # only half the arc is visible -> airborne, LOW confidence
ARC_NONE = "none"         # not airborne


@dataclass(frozen=True)
class AerialRun:
    """One loose run, and the verdict on whether the ball was flying through it."""

    start_frame: int
    end_frame: int
    n_frames: int
    n_samples: int          # usable img_y samples after cleaning
    n_rejected: int         # img_y samples thrown out as image-space glitches
    airborne: bool
    confidence: float
    kind: str               # ARC_FULL | ARC_PARTIAL | ARC_NONE
    curvature: float        # px / frame^2; > 0 is an upward-opening img_y parabola
    r2: float
    vertex_frame: float     # apex, in frames; NaN when there is no fit
    amplitude_px: float     # depth of the arc (or the ramp) in image pixels
    median_speed_ms: float  # apparent GROUND speed -- distorted, but elevated
    bbox_corr: float        # corr(img_y, bbox height): + means smaller near the apex
    n_ball_outlier: int     # how many of these frames the ground gate had rejected
    reasons: List[str] = field(default_factory=list)

    def as_meta(self) -> dict:
        return dict(
            start_frame=int(self.start_frame),
            end_frame=int(self.end_frame),
            n_frames=int(self.n_frames),
            n_samples=int(self.n_samples),
            airborne=bool(self.airborne),
            confidence=float(self.confidence),
            kind=self.kind,
            curvature=_round(self.curvature, 4),
            r2=_round(self.r2, 3),
            vertex_frame=_round(self.vertex_frame, 1),
            amplitude_px=_round(self.amplitude_px, 1),
            median_speed_ms=_round(self.median_speed_ms, 2),
            bbox_corr=_round(self.bbox_corr, 2),
            n_ball_outlier=int(self.n_ball_outlier),
            reasons=list(self.reasons),
        )


def _round(v: float, n: int):
    """``round`` that survives NaN (json.dump cannot serialize numpy NaN nicely)."""
    return None if not np.isfinite(v) else round(float(v), n)


class AerialTrack:
    """``airborne`` / ``aerial_conf`` by ball frame. Missing frame => not airborne.

    The event layer's read-side view of this module. Mirrors
    :class:`~src.actions.geometry.BallTrack`: built once per run, queried per gap.
    """

    def __init__(
        self,
        flags: Optional[Dict[int, Tuple[bool, float]]] = None,
        runs: Optional[List[AerialRun]] = None,
    ):
        self._flags: Dict[int, Tuple[bool, float]] = dict(flags or {})
        self.runs: List[AerialRun] = list(runs or [])

    @classmethod
    def empty(cls) -> "AerialTrack":
        """No aerial information at all: every frame reads as not-airborne.

        The honest default when ``img_y`` is absent (a ball-free possession
        source, or a tracking table without image columns). Not-airborne is the
        safe answer: it changes nothing, rather than relaxing a guard on a guess.
        """
        return cls()

    def airborne(self, frame: int) -> bool:
        return bool(self._flags.get(int(frame), (False, 0.0))[0])

    def conf(self, frame: int) -> float:
        return float(self._flags.get(int(frame), (False, 0.0))[1])

    def over(self, first: int, last: int) -> Tuple[bool, float]:
        """Was the ball airborne ANYWHERE in ``[first, last]``, and how sure?

        A gap is aerial if any part of the ball's journey across it was — a ball
        chipped over a defender and rolling on is still an aerial pass, and its
        ground path is still distorted where it flew. The confidence returned is
        the strongest evidence found in the window, not an average, because a
        confident arc in the middle of a long gap is not made less real by the
        rolling frames either side of it.
        """
        best = 0.0
        hit = False
        for f in range(int(first), int(last) + 1):
            flag, conf = self._flags.get(f, (False, 0.0))
            if flag:
                hit = True
                best = max(best, conf)
        return hit, best

    def to_frame(self) -> pd.DataFrame:
        """The ball annotation: one row per ball frame, ``frame/airborne/aerial_conf``."""
        if not self._flags:
            return pd.DataFrame(
                {
                    "frame": pd.Series(dtype="int64"),
                    "airborne": pd.Series(dtype="bool"),
                    "aerial_conf": pd.Series(dtype="float64"),
                }
            )
        rows = [
            dict(frame=int(f), airborne=bool(a), aerial_conf=float(c))
            for f, (a, c) in sorted(self._flags.items())
        ]
        return pd.DataFrame(rows)[AERIAL_COLUMNS]

    def __len__(self) -> int:
        return sum(1 for a, _c in self._flags.values() if a)


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
def _median_filter(y: np.ndarray, window: int) -> np.ndarray:
    """Running median, edge-clamped. Small, dependency-free, and robust."""
    n = len(y)
    if n == 0 or window < 3:
        return y.copy()
    half = min(window // 2, max(0, (n - 1) // 2))
    if half == 0:
        return y.copy()
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.median(y[lo:hi]))
    return out


def _clean_img_y(
    frames: np.ndarray, img_y: np.ndarray, cfg: ActionConfig
) -> np.ndarray:
    """Boolean mask of img_y samples to KEEP: reject image-space glitches.

    A single bad detection must not be able to break the fit. On the sample clip
    the aerial pass at frames 123-149 carries exactly one: frame 136 reads
    ``img_y=449`` in the middle of a run sitting around 325 — the detector briefly
    latched onto something else. A least-squares parabola is not robust, so that
    one sample would drag the vertex and wreck the R^2 if it survived.

    Robust, in image space, and deliberately NOT the ``ball_outlier`` flag (see
    the module docstring — that flag is confounded with airborne-ness). Residual
    against a running median, scaled by the MAD of those residuals: a normal
    detection sits within a few MADs of its neighbours; a glitch is orders out.
    """
    keep = np.ones(len(img_y), dtype=bool)
    if len(img_y) < 3:
        return keep

    resid = img_y - _median_filter(img_y, cfg.aerial_median_window)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    # 1.4826 * MAD is the MAD-based estimate of sigma for a normal distribution.
    # The floor keeps a *perfectly* smooth run (MAD == 0, common on a clean arc)
    # from declaring every non-identical sample an outlier.
    scale = max(1.4826 * mad, cfg.aerial_min_spike_px)
    return np.abs(resid) <= cfg.aerial_spike_mad * scale


def _fit_quadratic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Least-squares ``y = a*x^2 + b*x + c``. Returns ``(a, vertex_x, r2, amplitude)``.

    ``a > 0`` is an upward-opening parabola, i.e. a local MINIMUM in img_y, i.e.
    the ball rose and then fell. ``amplitude`` is the fitted depth of the arc over
    the observed span — how far the ball actually climbed, in pixels.
    """
    # Centre x before fitting: frame numbers are ~1e2-1e5 and squaring them into a
    # Vandermonde matrix is numerically nasty. Centring keeps it well conditioned.
    x0 = float(np.mean(x))
    xc = x - x0
    try:
        a, b, c = np.polyfit(xc, y, 2)
    except Exception:  # pragma: no cover - degenerate input (all-equal x)
        return 0.0, float("nan"), 0.0, 0.0

    pred = np.polyval([a, b, c], xc)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    # A flat run has nothing to explain; calling that a perfect fit would let a
    # motionless ball score R^2 = 1. It has no arc, so it explains no variance.
    r2 = 0.0 if ss_tot <= 1e-9 else float(1.0 - ss_res / ss_tot)

    if abs(a) < 1e-12:
        return float(a), float("nan"), r2, 0.0

    vertex_c = -b / (2.0 * a)
    # The fitted arc's depth over the span we actually observed -- not the
    # parabola's mathematical depth, which for a vertex far outside the window is
    # a fantasy about frames nobody saw.
    span_pred = np.polyval([a, b, c], np.array([xc.min(), xc.max(), vertex_c]))
    if xc.min() <= vertex_c <= xc.max():
        amplitude = float(max(span_pred[0], span_pred[1]) - span_pred[2])
    else:
        amplitude = float(abs(span_pred[1] - span_pred[0]))
    return float(a), float(vertex_c + x0), r2, amplitude


def _bbox_corr(img_y: np.ndarray, bbox_h: np.ndarray) -> float:
    """corr(img_y, bbox height). Positive => the box is SMALLER near the apex.

    A ball at the top of its arc is further from the camera, so it images smaller.
    img_y is *low* at the apex, and the bbox height should be low there too — the
    two move together. Independent physics from the parabola fit, which is why it
    is worth corroborating with: a camera tilt moves img_y without touching the
    ball's apparent size.
    """
    ok = np.isfinite(img_y) & np.isfinite(bbox_h)
    if ok.sum() < 4:
        return float("nan")
    a, b = img_y[ok], bbox_h[ok]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _confidence(
    kind: str,
    r2: float,
    speed_ok: bool,
    bbox_ok: bool,
    cfg: ActionConfig,
) -> float:
    """A blunt, honest 0-1 score. Not a probability -- a triage flag.

    Same spirit as the event layer's own ``_confidence``: a base for clearing the
    gate, plus credit for each independent line of evidence that agrees. A partial
    arc starts lower and is **capped**, because half an arc is genuinely weaker
    evidence than a whole one and no amount of corroboration should let it
    masquerade as a clean parabola.
    """
    if kind == ARC_PARTIAL:
        score = cfg.aerial_partial_base_conf
        score += 0.10 if speed_ok else 0.0
        score += 0.10 if bbox_ok else 0.0
        return float(min(cfg.aerial_partial_max_conf, round(score, 2)))

    score = 0.45
    # fit quality, scaled across the range that is actually admissible: at the R^2
    # floor this contributes nothing, at a perfect fit it contributes 0.25.
    span = max(1e-6, 1.0 - cfg.aerial_min_r2)
    score += 0.25 * float(np.clip((r2 - cfg.aerial_min_r2) / span, 0.0, 1.0))
    score += 0.15 if speed_ok else 0.0
    score += 0.15 if bbox_ok else 0.0
    return float(min(1.0, round(score, 2)))


# --------------------------------------------------------------------------- #
# loose runs
# --------------------------------------------------------------------------- #
def loose_runs(frames: pd.DataFrame, cfg: ActionConfig) -> List[Tuple[int, int]]:
    """Maximal runs of consecutive frames where the ball belongs to **nobody**.

    "Loose" here means *unattributed* -- the possession stream's ``loose`` AND its
    ``no_ball``. Both are states with no possessor, and lumping them is not a
    convenience: **a ball high in the air is exactly the ball the ground-plane
    pipeline loses**. On the sample clip the aerial pass reads as ``no_ball``
    (frames 102-122, 150-165) either side of ``loose`` (123-149), because the
    stretched ground coordinates were rejected by the prerequisites' speed gate.
    Splitting the run on that boundary would cut the arc into three and fit a
    parabola to its middle third.

    Runs longer than ``aerial_max_run_frames`` are dropped, not truncated: over
    several seconds a camera pan can trace an img_y curve of its own, and a window
    that long is no longer evidence about a single ball flight.
    """
    if frames.empty:
        return []
    df = frames.sort_values("frame", kind="stable")
    fr = df["frame"].to_numpy(dtype=int)
    unattributed = df["possessor_id"].isna().to_numpy()

    runs: List[Tuple[int, int]] = []
    i = 0
    while i < len(fr):
        if not unattributed[i]:
            i += 1
            continue
        j = i
        while (
            j + 1 < len(fr)
            and unattributed[j + 1]
            and fr[j + 1] == fr[j] + 1
        ):
            j += 1
        first, last = int(fr[i]), int(fr[j])
        if (last - first + 1) <= cfg.aerial_max_run_frames:
            runs.append((first, last))
        i = j + 1
    return runs


def _ball_samples(ball: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """The ball rows in ``[first, last]`` that carry a usable ``img_y``."""
    sl = ball[(ball["frame"] >= first) & (ball["frame"] <= last)]
    return sl[sl[COL_IMG_Y].notna()]


def _classify_run(
    ball: pd.DataFrame, first: int, last: int, cfg: ActionConfig
) -> AerialRun:
    """Fit one loose run and decide: full arc, partial arc, or not airborne."""
    n_frames = last - first + 1
    sl = _ball_samples(ball, first, last)

    def verdict(**kw) -> AerialRun:
        base = dict(
            start_frame=first, end_frame=last, n_frames=n_frames,
            n_samples=0, n_rejected=0, airborne=False, confidence=0.0,
            kind=ARC_NONE, curvature=0.0, r2=0.0, vertex_frame=float("nan"),
            amplitude_px=0.0, median_speed_ms=float("nan"),
            bbox_corr=float("nan"), n_ball_outlier=0, reasons=[],
        )
        base.update(kw)
        return AerialRun(**base)

    if len(sl) < cfg.aerial_min_run_frames:
        return verdict(n_samples=len(sl), reasons=["too_few_img_y_samples"])

    frames = sl["frame"].to_numpy(dtype=float)
    img_y = sl[COL_IMG_Y].to_numpy(dtype=float)

    keep = _clean_img_y(frames, img_y, cfg)
    n_rejected = int((~keep).sum())
    frames, img_y = frames[keep], img_y[keep]
    if len(frames) < cfg.aerial_min_run_frames:
        return verdict(
            n_samples=len(frames), n_rejected=n_rejected,
            reasons=["too_few_img_y_samples_after_cleaning"],
        )

    sl = sl[keep]
    n_outlier = (
        int((sl[COL_BALL_OUTLIER] == True).sum())  # noqa: E712
        if COL_BALL_OUTLIER in sl.columns else 0
    )

    # --- the two corroborating signals ------------------------------------ #
    # Apparent ground speed. It is DISTORTED (that is the whole problem), but a
    # ball that is flying still reads fast, and a camera panning over a ball
    # sitting in the grass does not. Missing speed is not evidence of slowness:
    # the ground gate nulls the speed of exactly the frames we care about, so an
    # absent speed neither gates nor corroborates.
    speeds = (
        sl[COL_BALL_SPEED].dropna().to_numpy(dtype=float)
        if COL_BALL_SPEED in sl.columns else np.array([])
    )
    median_speed = float(np.median(speeds)) if len(speeds) else float("nan")
    speed_known = np.isfinite(median_speed)
    speed_ok = bool(speed_known and median_speed >= cfg.aerial_min_speed_ms)

    if COL_BBOX_Y1 in sl.columns and COL_BBOX_Y2 in sl.columns:
        bbox_h = (sl[COL_BBOX_Y2] - sl[COL_BBOX_Y1]).to_numpy(dtype=float)
    else:
        bbox_h = np.full(len(sl), np.nan)
    corr = _bbox_corr(img_y, bbox_h)
    bbox_ok = bool(np.isfinite(corr) and corr >= cfg.aerial_bbox_min_corr)

    a, vertex, r2, amplitude = _fit_quadratic(frames, img_y)

    # A known-slow ball is not flying, whatever shape its img_y traces. This is
    # the main defence against a camera pan/tilt being read as an arc, so it gates
    # rather than merely informing the confidence.
    if speed_known and not speed_ok:
        return verdict(
            n_samples=len(frames), n_rejected=n_rejected, curvature=a, r2=r2,
            vertex_frame=vertex, amplitude_px=amplitude,
            median_speed_ms=median_speed, bbox_corr=corr,
            n_ball_outlier=n_outlier, reasons=["below_speed_floor"],
        )

    # --- 1. the full arc: rose AND fell, and we saw the apex --------------- #
    vertex_inside = bool(
        np.isfinite(vertex) and frames.min() <= vertex <= frames.max()
    )
    if (
        a >= cfg.aerial_min_curvature
        and r2 >= cfg.aerial_min_r2
        and vertex_inside
        and amplitude >= cfg.aerial_min_amplitude_px
    ):
        reasons = ["img_y_arc"]
        if speed_ok:
            reasons.append("elevated_speed")
        if bbox_ok:
            reasons.append("bbox_shrinks_at_apex")
        return AerialRun(
            start_frame=first, end_frame=last, n_frames=n_frames,
            n_samples=len(frames), n_rejected=n_rejected, airborne=True,
            confidence=_confidence(ARC_FULL, r2, speed_ok, bbox_ok, cfg),
            kind=ARC_FULL, curvature=a, r2=r2, vertex_frame=vertex,
            amplitude_px=amplitude, median_speed_ms=median_speed, bbox_corr=corr,
            n_ball_outlier=n_outlier, reasons=reasons,
        )

    # --- 2. the partial arc: only half the flight is in the window --------- #
    # The ball was already up when the run began, or we only ever saw it coming
    # down. Forcing a parabola onto half an arc invents a vertex nobody observed,
    # so instead: a large, sustained, one-way img_y ramp on a FAST loose ball,
    # with the bbox changing coherently -- airborne, and say plainly that we are
    # much less sure. A monotonic ramp is also what a pan looks like, which is why
    # this branch demands the speed AND the bbox evidence, not just the shape.
    ramp = float(img_y[-1] - img_y[0])
    monotone = _monotone_frac(img_y)
    if (
        speed_ok
        and bbox_ok
        and abs(ramp) >= cfg.aerial_partial_min_ramp_px
        and monotone >= cfg.aerial_partial_min_monotone
    ):
        return AerialRun(
            start_frame=first, end_frame=last, n_frames=n_frames,
            n_samples=len(frames), n_rejected=n_rejected, airborne=True,
            confidence=_confidence(ARC_PARTIAL, r2, speed_ok, bbox_ok, cfg),
            kind=ARC_PARTIAL, curvature=a, r2=r2, vertex_frame=vertex,
            amplitude_px=abs(ramp), median_speed_ms=median_speed, bbox_corr=corr,
            n_ball_outlier=n_outlier,
            reasons=["partial_arc", "descending" if ramp > 0 else "ascending"],
        )

    # --- 3. not airborne --------------------------------------------------- #
    why: List[str] = []
    if a < cfg.aerial_min_curvature:
        why.append("no_upward_curvature")
    if not vertex_inside:
        why.append("vertex_outside_run")
    if r2 < cfg.aerial_min_r2:
        why.append("poor_fit")
    if amplitude < cfg.aerial_min_amplitude_px:
        why.append("arc_too_shallow")
    return verdict(
        n_samples=len(frames), n_rejected=n_rejected, curvature=a, r2=r2,
        vertex_frame=vertex, amplitude_px=amplitude,
        median_speed_ms=median_speed, bbox_corr=corr, n_ball_outlier=n_outlier,
        reasons=why or ["no_arc"],
    )


def _monotone_frac(y: np.ndarray) -> float:
    """Fraction of consecutive steps that go the same way as the overall ramp."""
    if len(y) < 2:
        return 0.0
    steps = np.diff(y)
    direction = np.sign(y[-1] - y[0])
    if direction == 0:
        return 0.0
    return float(np.mean(np.sign(steps) == direction))


# --------------------------------------------------------------------------- #
# the entry point
# --------------------------------------------------------------------------- #
def detect_airborne(
    tracking: pd.DataFrame, frames: pd.DataFrame, cfg: ActionConfig
) -> AerialTrack:
    """Flag the ball frames on which the ball was probably off the ground.

    ``tracking`` is the prepared tracking table (needs the **image-space** ball
    columns ``img_y`` / ``bbox_y1`` / ``bbox_y2`` -- this is the one place in the
    event layer that reads image space rather than pitch metres, and it does so
    precisely because image space is where the homography has not yet destroyed
    the evidence). ``frames`` is the possession stream, which says which frames
    the ball belonged to nobody.

    Degrades to :meth:`AerialTrack.empty` when the image columns are absent, so a
    ball-free possession source (or an older prepared table) still runs -- it just
    never flags anything, which changes no behaviour anywhere.
    """
    if not cfg.aerial_enabled:
        return AerialTrack.empty()
    if COL_IMG_Y not in tracking.columns or "object_id" not in tracking.columns:
        return AerialTrack.empty()
    if frames.empty or "possessor_id" not in frames.columns:
        return AerialTrack.empty()

    ball = tracking[tracking["object_id"] == BALL_OBJECT_ID].sort_values(
        "frame", kind="stable"
    )
    if ball.empty:
        return AerialTrack.empty()

    ball_frames = set(int(f) for f in ball["frame"])
    flags: Dict[int, Tuple[bool, float]] = {f: (False, 0.0) for f in ball_frames}

    runs: List[AerialRun] = []
    for first, last in loose_runs(frames, cfg):
        run = _classify_run(ball, first, last, cfg)
        runs.append(run)
        if not run.airborne:
            continue
        # The verdict is about the RUN, so every ball frame in it is a flight
        # frame -- including the interpolated ones, whose ground coordinates are
        # the ones a downstream consumer most needs to be warned off.
        for f in range(first, last + 1):
            if f in ball_frames:
                flags[f] = (True, run.confidence)

    return AerialTrack(flags, runs)


def summarize_runs(track: AerialTrack, cfg: ActionConfig) -> dict:
    """The aerial half of the stage meta: what was found, and on what thresholds."""
    airborne = [r for r in track.runs if r.airborne]
    return dict(
        enabled=bool(cfg.aerial_enabled),
        n_loose_runs=len(track.runs),
        n_aerial_runs=len(airborne),
        n_aerial_frames=len(track),
        n_full_arcs=sum(1 for r in airborne if r.kind == ARC_FULL),
        n_partial_arcs=sum(1 for r in airborne if r.kind == ARC_PARTIAL),
        mean_aerial_conf=(
            round(float(np.mean([r.confidence for r in airborne])), 3)
            if airborne else 0.0
        ),
        thresholds=dict(
            aerial_min_run_frames=cfg.aerial_min_run_frames,
            aerial_max_run_frames=cfg.aerial_max_run_frames,
            aerial_min_curvature=cfg.aerial_min_curvature,
            aerial_min_r2=cfg.aerial_min_r2,
            aerial_min_amplitude_px=cfg.aerial_min_amplitude_px,
            aerial_min_speed_ms=cfg.aerial_min_speed_ms,
            aerial_bbox_min_corr=cfg.aerial_bbox_min_corr,
            aerial_median_window=cfg.aerial_median_window,
            aerial_spike_mad=cfg.aerial_spike_mad,
            aerial_partial_min_ramp_px=cfg.aerial_partial_min_ramp_px,
            aerial_partial_max_conf=cfg.aerial_partial_max_conf,
        ),
        runs=[r.as_meta() for r in track.runs if r.airborne],
        note=(
            "HEURISTIC, single-camera detector -- NOT height recovery. The "
            "homography maps the image to the ground plane (z=0), so an airborne "
            "ball's pitch coordinates are stretched away from the camera and "
            "there is no height channel to correct them with. This flags WHICH "
            "frames are untrustworthy (an upward-opening img_y arc on a fast, "
            "loose ball); it does not estimate how high the ball was. Camera "
            "pan/tilt moves img_y too, which is why airborne also requires the "
            "ball to be loose, the run to be bounded, and the speed/bbox evidence "
            "to agree."
        ),
    )


# --------------------------------------------------------------------------- #
# EXTENSION POINT (ballistic reconstruction) -- deliberately NOT built.
#
# With the flight frames identified, the obvious next step is to RECOVER the
# height: fit a ballistic parabola z(t) = z0 + vz*t - g*t^2/2 anchored on the two
# endpoints (which are on the ground, and which -- after CHANGE 1 -- we now take
# from the passer and the receiver rather than from the distorted ball track),
# then back-project each mid-flight image point onto that parabola instead of onto
# z=0 to recover a corrected (x, y, z).
#
# It is NOT built, and it is not needed:
#
#   * SPADL actions are start->end, not trajectories. Anchoring the endpoints on
#     the players (CHANGE 1) already keeps every distorted mid-flight coordinate
#     out of the emitted event stream, and therefore out of xT and VAEP. There is
#     no consumer in this project for a corrected mid-flight ball position.
#   * It would need the ball's true 3D launch point and a calibrated camera to be
#     anything better than a curve that happens to join two dots.
#
# What it WOULD unlock, if a consumer ever appears: true ball height for
# aerial-duel detection, header/volley bodypart inference (see the bodypart
# limitation in the README), and shot trajectories over/under the bar.
#
# The inputs it needs already exist: `AerialRun` gives the flight window and the
# apex frame; `GapPath.start` / `.end` give the ground-truth endpoints; `img_x` /
# `img_y` give the image ray; `meta.json` records fps. What is missing is the
# camera intrinsics/extrinsics -- the homography alone cannot invert a ray to a
# height -- not the shape of the code.
# --------------------------------------------------------------------------- #
