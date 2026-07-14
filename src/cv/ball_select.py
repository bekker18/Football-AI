"""Resolve the single in-play ball from the many ball-class detections per frame.

Broadcast clips contain several ball-class detections at once: the in-play ball,
plus static spare balls by the touchline and behind the goals. Upstream
(``sports.common.ball.BallTracker``) collapsed them to the detection nearest the
centroid of a rolling buffer, which systematically locks onto a *static* spare
ball — a stationary ball contributes the same point on every frame, so it
accumulates buffer mass and drags the centroid toward itself, while the moving
game ball's contributions smear out and cancel. This module replaces that rule
with multi-candidate track scoring.

Stages:

1. **associate** per-frame candidates into candidate *tracks* (greedy nearest
   neighbour, gated);
2. **score** each track over a window on
     - motion energy — a static spare ball scores ~0, the game ball high. The
       primary discriminator;
     - on-pitch fraction — spare balls sit off the pitch and score ~0. This is
       what rescues the *resting* game ball (static, but on the pitch), which
       motion energy alone would throw away;
3. **select** a winner per window with hysteresis, so the identity does not
   flicker between candidates; and
4. **allow no winner.** When the real ball is occluded or genuinely out of play,
   emitting null is correct — it feeds the existing gap handling in
   :mod:`src.prerequisites.ball`. A static candidate is never selected merely
   because it is the only thing left.

Two deliberate coordinate choices:

*Motion energy is measured in pitch metres, not image pixels.* The camera pans,
so a static spare ball moves across the image. Only pitch coordinates remove
camera motion and let "static" actually mean static.

*Association runs in image pixels.* It needs frame-to-frame continuity even where
the homography is missing or noisy — which is exactly the touchline, where the
spare balls sit — and adjacent-frame pixel displacement stays small regardless.

On-pitch fraction is a weighted signal over a window, never a hard per-frame cut:
the homography is noisiest at the touchline, and the real ball legitimately
crosses that line on throw-ins and corners.

Online mode: ``mode="trailing"`` makes every window causal (``[t-W+1, t]``), so
the same scorer drives the real-time path — warm up, then commit on candidate
identity. ``"centered"`` is the offline default.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class BallSelectConfig:
    """Tunables for in-play ball selection.

    Defaults are for 25 fps, 1080p broadcast. ``motion_ref_m`` and ``min_score``
    are the two that decide the false-lock rate; see the module docstring in
    ``tests/cv/test_ball_select.py`` for the cases they are pinned against.
    """

    fps: float = 25.0

    # --- association ---
    assoc_gate_px: float = 80.0  # max frame-to-frame pixel step within one track
    assoc_max_gap: int = 8  # frames a track may go unseen before it is closed

    # --- windowing ---
    window_frames: int = 51  # ~2 s @ 25 fps
    window_hop: int = 10  # frames committed per window
    mode: str = "centered"  # "centered" (offline) | "trailing" (causal/online)
    warmup_frames: int = 15  # trailing mode: emit nothing until this much history

    # --- scoring ---
    min_track_frames: int = 5  # eligibility floor, NOT a score term (see below)
    motion_ref_m: float = 2.0  # gyration radius (m) that saturates the motion score
    pitch_margin_m: float = 1.0  # how far outside the lines still counts as on-pitch
    w_motion: float = 1.0
    w_onpitch: float = 1.0
    min_score: float = 0.20  # below this, NO ball is emitted for the window
    hysteresis: float = 0.15  # challenger must beat the incumbent by this to switch

    # --- pitch bounds (metres; matches cv.config PITCH_LEN_M / PITCH_WID_M) ---
    pitch_len_m: float = 120.0
    pitch_wid_m: float = 70.0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One ball-class detection in one frame."""

    frame: int
    img_x: float  # bbox centre (association + rendering)
    img_y: float
    pitch_x: Optional[float]  # None when the frame has no homography
    pitch_y: Optional[float]
    pitch_valid: bool
    conf: float
    bbox: Sequence[float]  # (x1, y1, x2, y2)
    track_id: int = -1  # filled in by associate()


@dataclass
class Track:
    """A candidate track: the same physical ball across consecutive frames."""

    track_id: int
    cands: List[Candidate] = field(default_factory=list)
    _frames: List[int] = field(default_factory=list)  # parallel to cands, ascending

    def append(self, c: Candidate) -> None:
        self.cands.append(c)
        self._frames.append(c.frame)

    @property
    def last_frame(self) -> int:
        return self.cands[-1].frame

    @property
    def last_xy(self):
        c = self.cands[-1]
        return c.img_x, c.img_y

    def between(self, lo: int, hi: int) -> List[Candidate]:
        """Candidates with ``lo <= frame <= hi``.

        Bisected, not scanned: this is called once per track per window, so a
        linear scan makes selection quadratic in clip length — fine on a 300-frame
        smoke test, hopeless on a 90-minute match (~135k frames).
        """
        i = bisect_left(self._frames, lo)
        j = bisect_right(self._frames, hi)
        return self.cands[i:j]


# ---------------------------------------------------------------------------
# 1. association
# ---------------------------------------------------------------------------


def associate(cands: Sequence[Candidate], cfg: BallSelectConfig) -> Dict[int, Track]:
    """Greedy nearest-neighbour association of per-frame candidates into tracks.

    A static ball yields one long, stable track; the game ball a moving,
    occasionally fragmented one. Fragmentation is tolerable — a fragment still
    carries the motion and on-pitch evidence that wins it the window.

    Greedy (rather than Hungarian) is deliberate: there are only a handful of
    ball candidates per frame, they are far apart in the image, and greedy has no
    pathological case at that size.
    """
    by_frame: Dict[int, List[Candidate]] = {}
    for c in cands:
        by_frame.setdefault(c.frame, []).append(c)

    tracks: Dict[int, Track] = {}
    active: List[Track] = []
    next_id = 0

    for frame in sorted(by_frame):
        # retire tracks that have been unseen for too long
        active = [t for t in active if frame - t.last_frame <= cfg.assoc_max_gap]
        frame_cands = by_frame[frame]

        # all (distance, track, candidate) pairs inside the gate, nearest first
        pairs = []
        for ti, t in enumerate(active):
            tx, ty = t.last_xy
            for ci, c in enumerate(frame_cands):
                d = float(np.hypot(c.img_x - tx, c.img_y - ty))
                if d <= cfg.assoc_gate_px:
                    pairs.append((d, ti, ci))
        pairs.sort(key=lambda p: p[0])

        used_t: set = set()
        used_c: set = set()
        for _, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            t = active[ti]
            c = frame_cands[ci]
            c.track_id = t.track_id
            t.append(c)

        # unmatched candidates open new tracks
        for ci, c in enumerate(frame_cands):
            if ci in used_c:
                continue
            t = Track(track_id=next_id)
            next_id += 1
            c.track_id = t.track_id
            t.append(c)
            tracks[t.track_id] = t
            active.append(t)

    return tracks


# ---------------------------------------------------------------------------
# 2. scoring
# ---------------------------------------------------------------------------


@dataclass
class TrackScore:
    track_id: int
    score: float
    motion: float  # saturated motion energy in [0, 1]
    onpitch: float  # on-pitch fraction in [0, 1]
    gyration_m: float  # raw robust spread, metres (for debugging)
    n_frames: int
    eligible: bool


def _gyration_m(pts: np.ndarray) -> float:
    """Robust spread of a point set: median distance from the median point.

    Chosen over path length because it is insensitive to homography jitter, which
    inflates a static ball's *path* (a jittery point retraces the same few metres
    over and over) far more than it inflates its *spread*. A ball genuinely in
    play sweeps a large area and scores high on both; a jittering static ball
    scores high only on path length. So spread is the honest discriminator, and
    it is what keeps a touchline spare ball — sitting exactly where the
    homography is worst — from faking motion energy.
    """
    if len(pts) < 2:
        return 0.0
    centre = np.median(pts, axis=0)
    return float(np.median(np.linalg.norm(pts - centre, axis=1)))


def _onpitch_fraction(pts: np.ndarray, cfg: BallSelectConfig) -> float:
    """Fraction of a track's points lying inside the pitch (+ a small margin)."""
    if len(pts) == 0:
        return 0.0
    m = cfg.pitch_margin_m
    inside = (
        (pts[:, 0] >= -m)
        & (pts[:, 0] <= cfg.pitch_len_m + m)
        & (pts[:, 1] >= -m)
        & (pts[:, 1] <= cfg.pitch_wid_m + m)
    )
    return float(inside.mean())


def score_track(
    track: Track, lo: int, hi: int, cfg: BallSelectConfig
) -> Optional[TrackScore]:
    """Score one track over the window ``[lo, hi]``. None if it has no presence."""
    win = track.between(lo, hi)
    if not win:
        return None

    pts = np.array(
        [(c.pitch_x, c.pitch_y) for c in win if c.pitch_valid and c.pitch_x is not None],
        dtype=float,
    ).reshape(-1, 2)

    gyr = _gyration_m(pts)
    motion = min(1.0, gyr / cfg.motion_ref_m) if cfg.motion_ref_m > 0 else 0.0
    onpitch = _onpitch_fraction(pts, cfg)

    # Support (how many frames the track appears in) is an eligibility FLOOR, not
    # a score term — on purpose. Static balls are detected consistently and would
    # win any support- or confidence-weighted vote outright; that is the very
    # failure mode this stage exists to remove.
    eligible = len(win) >= cfg.min_track_frames and len(pts) > 0

    wsum = cfg.w_motion + cfg.w_onpitch
    score = (cfg.w_motion * motion + cfg.w_onpitch * onpitch) / wsum if wsum else 0.0

    return TrackScore(
        track_id=track.track_id,
        score=score,
        motion=motion,
        onpitch=onpitch,
        gyration_m=gyr,
        n_frames=len(win),
        eligible=eligible,
    )


# ---------------------------------------------------------------------------
# 3. windowed selection with hysteresis
# ---------------------------------------------------------------------------


def _windows(frames: Sequence[int], cfg: BallSelectConfig):
    """Yield ``(lo, hi, commit_lo, commit_hi)`` window/commit-range pairs.

    ``centered`` looks both ways around the block it commits (offline).
    ``trailing`` only ever looks backwards, so the identical scorer runs causally
    in the online path — the window is ``[t - W + 1, t]`` and it commits the
    newest hop of frames.
    """
    if not frames:
        return
    f0, f1 = min(frames), max(frames)
    half = cfg.window_frames // 2
    hop = max(1, cfg.window_hop)

    for start in range(f0, f1 + 1, hop):
        commit_lo = start
        commit_hi = min(start + hop - 1, f1)
        if cfg.mode == "trailing":
            lo = max(f0, commit_hi - cfg.window_frames + 1)
            hi = commit_hi  # causal: never reads past the committed frame
            if commit_hi - f0 + 1 < cfg.warmup_frames:
                continue  # warm-up: not enough history to commit an identity yet
        else:
            mid = (commit_lo + commit_hi) // 2
            lo, hi = mid - half, mid + half
        yield lo, hi, commit_lo, commit_hi


@dataclass
class Selection:
    """The in-play ball for one frame (or its absence)."""

    frame: int
    cand: Optional[Candidate]  # None => no in-play ball this frame
    track_id: int = -1
    score: float = 0.0
    margin: float = 0.0  # winner score minus runner-up; low => ambiguous


def select_in_play_ball(
    cands: Sequence[Candidate], cfg: BallSelectConfig
) -> tuple[List[Selection], Dict[int, Track], Dict[int, TrackScore]]:
    """Resolve one in-play ball per frame (or none).

    Returns ``(selections, tracks, last_scores)``. ``selections`` has one entry
    per frame that had any candidate at all; ``cand`` is None on frames where the
    winning track has no detection (occlusion) or no track cleared ``min_score``
    (genuinely no ball in play).
    """
    if not cands:
        return [], {}, {}

    tracks = associate(cands, cfg)
    frames = sorted({c.frame for c in cands})

    winner_by_frame: Dict[int, Optional[TrackScore]] = {}
    margin_by_frame: Dict[int, float] = {}
    last_scores: Dict[int, TrackScore] = {}

    incumbent: Optional[int] = None  # track_id currently holding the ball identity

    for lo, hi, clo, chi in _windows(frames, cfg):
        scores = [
            s
            for s in (score_track(t, lo, hi, cfg) for t in tracks.values())
            if s is not None and s.eligible
        ]
        for s in scores:
            last_scores[s.track_id] = s

        ranked = sorted(scores, key=lambda s: s.score, reverse=True)
        viable = [s for s in ranked if s.score >= cfg.min_score]

        win: Optional[TrackScore] = viable[0] if viable else None

        # Hysteresis: keep the incumbent unless a challenger clears it by a
        # margin. Without this the identity flickers between candidates whenever
        # two scores sit close together, and every flicker is a teleport in the
        # emitted track.
        if win is not None and incumbent is not None and win.track_id != incumbent:
            inc = next((s for s in viable if s.track_id == incumbent), None)
            if inc is not None and win.score - inc.score < cfg.hysteresis:
                win = inc

        # A window with no viable track hands back the identity: the next window
        # is free to pick a different ball rather than being anchored to a stale
        # incumbent from before the gap.
        incumbent = win.track_id if win is not None else None

        runner_up = next((s.score for s in ranked if win and s.track_id != win.track_id), 0.0)
        for f in range(clo, chi + 1):
            winner_by_frame[f] = win
            margin_by_frame[f] = (win.score - runner_up) if win else 0.0

    # materialise: the winning track's detection on each frame, if it has one
    by_track_frame: Dict[tuple, Candidate] = {
        (c.track_id, c.frame): c for c in cands
    }
    selections: List[Selection] = []
    for f in frames:
        win = winner_by_frame.get(f)
        if win is None:
            selections.append(Selection(frame=f, cand=None))
            continue
        cand = by_track_frame.get((win.track_id, f))
        selections.append(
            Selection(
                frame=f,
                cand=cand,  # None when the winner is occluded on this frame
                track_id=win.track_id,
                score=win.score,
                margin=margin_by_frame.get(f, 0.0),
            )
        )
    return selections, tracks, last_scores
