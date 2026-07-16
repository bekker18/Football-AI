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
2. **score** each track over a window on four combined signals — none a hard gate:
     - distance to the nearest player (pitch metres) — the *being-played* signal
       and the primary discriminator. The played ball lives inside the player
       cluster; a spare/stray ball drifts away from it. This is what catches a
       static ball resting just inside the goal line (on the pitch, low motion,
       but 20 m from anyone) that motion + on-pitch alone lock onto for hundreds
       of frames;
     - motion energy — a static spare ball scores ~0, a sweeping ball high;
     - on-pitch fraction — a weak tie-breaker (see ``w_onpitch``), not a lifeline;
     - trajectory physics — a soft, multiplicative plausibility factor in
       ``[physics_floor, 1]``: a coherent path that respects the ~36 m/s cap keeps
       its score; a track glued together from teleporting static-ball noise is
       down-weighted. It never *adds* score, so it cannot floor a static ball;
3. **select** a winner per window with hysteresis, so the identity does not
   flicker between candidates;
4. **bridge** short dropouts. A null run no longer than ``bridge_max_gap`` that is
   bracketed by the *same* selected track is recovered rather than emitted null —
   any sub-threshold detection on those frames is taken, and a genuine no-detection
   gap is linearly interpolated (flagged ``bridged``); and
5. **allow no winner.** When the real ball is occluded or genuinely out of play,
   emitting null is correct — it feeds the existing gap handling in
   :mod:`src.prerequisites.ball`. A static candidate is never selected merely
   because it is the only thing left.

Distance-to-player needs per-frame player pitch positions, passed to
:func:`select_in_play_ball` as ``player_pos_by_frame``. Without them (e.g. the unit
tests) the signal is simply dropped and the score renormalises over the rest, so
the selector degrades to the motion + on-pitch behaviour it had before.

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
    # Eligibility floor, NOT a score term (see below). Kept low because the
    # distance-to-player factor, not track length, is now what rejects spare balls;
    # this floor only has to suppress 1-2 frame spurious detections, and set higher
    # it would discard the many short real-ball fragments a fast, occluded game ball
    # is broken into.
    min_track_frames: int = 3
    motion_ref_m: float = 2.0  # gyration radius (m) that saturates the motion score
    pitch_margin_m: float = 1.0  # how far outside the lines still counts as on-pitch
    player_dist_ref_m: float = 12.0  # nearest-player distance (m) that saturates dist to the floor
    dist_floor: float = 0.10  # a ball far from every player keeps this fraction (soft, not a gate)
    ball_max_speed_ms: float = 36.0  # physics cap; steps above this are teleports
    physics_floor: float = 0.5  # a fully-teleporting track keeps this factor (soft, not a gate)
    # motion + on-pitch form the additive "ball quality" base (moving and/or inbounds).
    w_motion: float = 1.0
    w_onpitch: float = 1.0
    # Distance-to-player is the primary "being played" discriminator, so it enters as
    # a MULTIPLICATIVE factor, not another additive vote: motion + on-pitch alone
    # describe a ball that is moving and inbounds, which a spare ball kicked along the
    # sideline also is — only proximity to the play tells them apart, and it has to be
    # able to veto a high motion/on-pitch score, which an additive term (renormalised
    # away) cannot. The ``dist_floor`` keeps it soft: a far ball is suppressed below
    # ``min_score``, never hard-zeroed. Physics is the same shape, one level down.
    min_score: float = 0.20  # below this, NO ball is emitted for the window
    hysteresis: float = 0.15  # challenger must beat the incumbent by this to switch

    # --- continuity / bridging: recover short dropouts instead of emitting null ---
    bridge_max_gap: int = 12  # null run <= this, bracketed by the ball, is bridged
    # Slack (m) added to the physically reachable step when matching the next
    # detection to the locked trajectory. It absorbs the position jump across a
    # fragment / track-id boundary (a different detection of the same ball, off a
    # noisy homography) without being wide enough to swallow a *second* in-play ball,
    # which sits tens of metres away. Below this + speed*dt is "the same ball"; a
    # jump beyond it holds (and bridges) rather than teleporting onto another ball.
    continuity_slack_m: float = 6.0

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
    dist: float  # being-played factor in [dist_floor, 1] (1.0 == on top of a player)
    physics: float  # trajectory plausibility factor in [physics_floor, 1]
    gyration_m: float  # raw robust spread, metres (for debugging)
    player_dist_m: float  # raw median nearest-player distance, metres (nan if unknown)
    dist_available: bool  # whether player positions were available for this window
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


def _player_dist_factor(
    win: Sequence[Candidate],
    player_pos_by_frame: Optional[Dict[int, np.ndarray]],
    cfg: BallSelectConfig,
):
    """Soft "being played" factor in ``[dist_floor, 1]`` from ball-to-player distance.

    The played ball lives inside the player cluster, so its distance to the
    *nearest* player is small; a spare ball off the pitch or resting behind the
    goal is metres from anyone. We take the median nearest-player distance over the
    window — deliberately median, so the transient frames of a long pass or shot,
    when even the game ball is briefly far from every player, do not define the
    track — and map it to a factor that is 1.0 on top of a player and decays to
    ``dist_floor`` at ``player_dist_ref_m``. It multiplies the score so that far
    from the play *vetoes* an otherwise high motion/on-pitch base, which is the
    whole point of the signal; the floor keeps the veto soft rather than a hard cut.

    Returns ``(factor, median_dist_m, available)``. ``available`` is False — factor
    then defaults to 1.0 (no veto) — when no player positions were supplied, or none
    of the window's ball frames had both a valid pitch point and a player on the pitch.
    """
    if not player_pos_by_frame:
        return 1.0, float("nan"), False
    dists = []
    for c in win:
        if not (c.pitch_valid and c.pitch_x is not None):
            continue
        P = player_pos_by_frame.get(c.frame)
        if P is None or len(P) == 0:
            continue
        dists.append(float(np.min(np.hypot(P[:, 0] - c.pitch_x, P[:, 1] - c.pitch_y))))
    if not dists:
        return 1.0, float("nan"), False
    med = float(np.median(dists))
    ref = cfg.player_dist_ref_m
    near = max(0.0, 1.0 - med / ref) if ref > 0 else 0.0  # 1 on a player, 0 by ref
    factor = cfg.dist_floor + (1.0 - cfg.dist_floor) * near
    return factor, med, True


def _physics_factor(win: Sequence[Candidate], cfg: BallSelectConfig) -> float:
    """Soft trajectory-plausibility factor in ``[physics_floor, 1]``.

    A played ball traces a coherent path: consecutive detections imply a speed
    under the ~36 m/s cap. A track stitched out of static-ball noise and ID
    switches teleports, so a large fraction of its steps blow the cap. We return
    the fraction of steps within the cap, floored at ``physics_floor`` so the term
    only ever *down-weights* an implausible track — it is multiplicative and never
    a hard gate, and a static (zero-speed) coherent track scores a clean 1.0.
    """
    pts = [
        (c.frame, c.pitch_x, c.pitch_y)
        for c in win
        if c.pitch_valid and c.pitch_x is not None
    ]
    if len(pts) < 2:
        return 1.0  # nothing to contradict: no penalty
    pts.sort(key=lambda p: p[0])
    fps = cfg.fps if cfg.fps > 0 else 25.0
    vmax = cfg.ball_max_speed_ms
    ok = tot = 0
    for (f0, x0, y0), (f1, x1, y1) in zip(pts[:-1], pts[1:]):
        dt = (f1 - f0) / fps
        if dt <= 0:
            continue
        speed = float(np.hypot(x1 - x0, y1 - y0)) / dt
        tot += 1
        if speed <= vmax:
            ok += 1
    if tot == 0:
        return 1.0
    coherence = ok / tot
    return cfg.physics_floor + (1.0 - cfg.physics_floor) * coherence


def score_track(
    track: Track,
    lo: int,
    hi: int,
    cfg: BallSelectConfig,
    player_pos_by_frame: Optional[Dict[int, np.ndarray]] = None,
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
    dist, player_dist_m, dist_avail = _player_dist_factor(win, player_pos_by_frame, cfg)
    physics = _physics_factor(win, cfg)

    # Support (how many frames the track appears in) is an eligibility FLOOR, not
    # a score term — on purpose. Static balls are detected consistently and would
    # win any support- or confidence-weighted vote outright; that is the very
    # failure mode this stage exists to remove.
    eligible = len(win) >= cfg.min_track_frames and len(pts) > 0

    # "Ball quality" base (moving and/or inbounds), then the two soft plausibility
    # factors: proximity to the play (the being-played veto) and trajectory physics.
    wsum = cfg.w_motion + cfg.w_onpitch
    base = (cfg.w_motion * motion + cfg.w_onpitch * onpitch) / wsum if wsum else 0.0
    score = base * dist * physics

    return TrackScore(
        track_id=track.track_id,
        score=score,
        motion=motion,
        onpitch=onpitch,
        dist=dist,
        physics=physics,
        gyration_m=gyr,
        player_dist_m=player_dist_m,
        dist_available=dist_avail,
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
    cand: Optional[Candidate]  # None => no *detected* in-play ball this frame
    track_id: int = -1
    score: float = 0.0
    margin: float = 0.0  # winner score minus runner-up; low => ambiguous
    bridged: bool = False  # position interpolated across a short same-track dropout
    pitch_x: Optional[float] = None  # bridged frames carry an interpolated position
    pitch_y: Optional[float] = None
    img_x: Optional[float] = None
    img_y: Optional[float] = None


def select_in_play_ball(
    cands: Sequence[Candidate],
    cfg: BallSelectConfig,
    player_pos_by_frame: Optional[Dict[int, np.ndarray]] = None,
) -> tuple[List[Selection], Dict[int, Track], Dict[int, TrackScore]]:
    """Resolve one in-play ball per frame (or none).

    ``player_pos_by_frame`` maps a frame to an ``(N, 2)`` array of on-pitch player
    positions (pitch metres); it powers the distance-to-player signal and may be
    omitted (the signal is then dropped — see :func:`score_track`).

    Returns ``(selections, tracks, last_scores)``. ``selections`` has one entry
    per frame that had any candidate at all; ``cand`` is None on frames where the
    winning track has no detection (occlusion) or no track cleared ``min_score``
    (genuinely no ball in play). A ``bridged`` selection has no ``cand`` but does
    carry an interpolated position across a short same-track dropout.
    """
    if not cands:
        return [], {}, {}

    tracks = associate(cands, cfg)
    frames = sorted({c.frame for c in cands})

    # Per committed frame: the viable tracks and their score, plus the preferred
    # "acquire" identity (the top track, with cross-window hysteresis). Selection
    # then follows the ball by POSITIONAL continuity across these, because the game
    # ball is routinely fragmented into a chain of short tracks (an ID switch every
    # time YOLO drops it for a frame) — locking to a single winning track id caps
    # coverage at one fragment's length and drops the ball onto null the moment the
    # committed fragment's detections fall outside the block, even while a sibling
    # fragment sits right on the ball.
    viable_by_frame: Dict[int, Dict[int, float]] = {}
    winner_by_frame: Dict[int, Optional[int]] = {}
    margin_by_frame: Dict[int, float] = {}
    last_scores: Dict[int, TrackScore] = {}

    incumbent: Optional[int] = None  # track_id currently holding the ball identity

    for lo, hi, clo, chi in _windows(frames, cfg):
        scores = [
            s
            for s in (
                score_track(t, lo, hi, cfg, player_pos_by_frame)
                for t in tracks.values()
            )
            if s is not None and s.eligible
        ]
        for s in scores:
            last_scores[s.track_id] = s

        ranked = sorted(scores, key=lambda s: s.score, reverse=True)
        viable = [s for s in ranked if s.score >= cfg.min_score]
        viable_map = {s.track_id: s.score for s in viable}

        win: Optional[TrackScore] = viable[0] if viable else None

        # Hysteresis on the acquire identity: keep the incumbent unless a challenger
        # clears it by a margin, so the preferred re-acquisition target does not
        # flip whenever two window scores sit close together.
        if win is not None and incumbent is not None and win.track_id != incumbent:
            inc = next((s for s in viable if s.track_id == incumbent), None)
            if inc is not None and win.score - inc.score < cfg.hysteresis:
                win = inc
        incumbent = win.track_id if win is not None else None

        runner_up = next((s.score for s in ranked if win and s.track_id != win.track_id), 0.0)
        for f in range(clo, chi + 1):
            viable_by_frame[f] = viable_map
            winner_by_frame[f] = win.track_id if win is not None else None
            margin_by_frame[f] = (win.score - runner_up) if win else 0.0

    cands_on: Dict[int, List[Candidate]] = {}
    for c in cands:
        cands_on.setdefault(c.frame, []).append(c)

    # 1) emit a detection per frame by following the ball's trajectory
    chosen = _emit_by_continuity(
        frames, cands_on, viable_by_frame, winner_by_frame, cfg
    )
    # 2) bridge short occlusion gaps between two continuous emitted detections
    bridged_pos = _bridge_gaps(frames, chosen, cfg)

    selections: List[Selection] = []
    for f in frames:
        c = chosen.get(f)
        if c is not None:
            selections.append(
                Selection(
                    frame=f,
                    cand=c,
                    track_id=c.track_id,
                    score=viable_by_frame.get(f, {}).get(c.track_id, 0.0),
                    margin=margin_by_frame.get(f, 0.0),
                )
            )
        elif f in bridged_pos:
            px, py, ix, iy, tid = bridged_pos[f]
            selections.append(
                Selection(frame=f, cand=None, track_id=tid, bridged=True,
                          pitch_x=px, pitch_y=py, img_x=ix, img_y=iy)
            )
        else:
            selections.append(Selection(frame=f, cand=None))
    return selections, tracks, last_scores


def _emit_by_continuity(
    frames: Sequence[int],
    cands_on: Dict[int, List[Candidate]],
    viable_by_frame: Dict[int, Dict[int, float]],
    winner_by_frame: Dict[int, Optional[int]],
    cfg: BallSelectConfig,
) -> Dict[int, Candidate]:
    """Pick one detection per frame, following the trajectory across fragments.

    On each frame only *viable* detections (their track cleared ``min_score`` in the
    covering window — i.e. the "being played" evidence held up) are eligible, so a
    static spare ball is never a candidate here. Among those:

    * fresh acquisition (first frame, or after a gap longer than ``bridge_max_gap``)
      takes the committed winner if it is present, else the highest-scoring viable
      detection — the ball is (re)found by evidence;
    * while locked on, we take the viable detection nearest the last emitted
      position, provided it is within a physically reachable step
      (``ball_max_speed_ms`` over the elapsed time, plus a small margin). If nothing
      reachable is present we emit *nothing* for that frame rather than jumping to a
      viable ball elsewhere — this clip has several balls on the pitch at once, and a
      one-frame dropout must resume the same trajectory (via bridging), not teleport
      onto a different ball. Only once the gap outlives ``bridge_max_gap`` does the
      lock lapse and evidence re-acquire from scratch.

    This is what lets the emitter ride a chain of short fragments as one ball while
    refusing both a spare ball on the far touchline and a flicker onto a second
    in-play ball during a brief occlusion.
    """
    fps = cfg.fps if cfg.fps > 0 else 25.0
    chosen: Dict[int, Candidate] = {}
    last_pos = None
    last_frame = None

    for f in frames:
        viable = viable_by_frame.get(f, {})
        present = [
            c for c in cands_on.get(f, [])
            if c.track_id in viable and c.pitch_valid and c.pitch_x is not None
        ]
        if not present:
            continue
        locked = last_pos is not None and last_frame is not None and (
            f - last_frame <= cfg.bridge_max_gap
        )
        pick: Optional[Candidate] = None
        if locked:
            reach = cfg.ball_max_speed_ms * (f - last_frame) / fps + cfg.continuity_slack_m
            near = min(present, key=lambda c: np.hypot(c.pitch_x - last_pos[0],
                                                       c.pitch_y - last_pos[1]))
            if np.hypot(near.pitch_x - last_pos[0], near.pitch_y - last_pos[1]) <= reach:
                pick = near
            # else: locked but nothing reachable — hold, do not jump (leave for bridging)
        else:  # fresh acquisition by evidence
            win_tid = winner_by_frame.get(f)
            pick = next((c for c in present if c.track_id == win_tid), None)
            if pick is None:
                pick = max(present, key=lambda c: viable[c.track_id])
        if pick is None:
            continue
        chosen[f] = pick
        last_pos = (pick.pitch_x, pick.pitch_y)
        last_frame = f
    return chosen


def _bridge_gaps(
    frames: Sequence[int],
    chosen: Dict[int, Candidate],
    cfg: BallSelectConfig,
) -> Dict[int, tuple]:
    """Linearly interpolate short occlusion gaps between two continuous emissions.

    A run of no-detection frames no longer than ``bridge_max_gap``, bracketed by two
    emitted detections whose separation is physically reachable, is a brief occlusion
    inside one phase of play, not an out-of-play stretch. We fill it with an
    interpolated ``(pitch_x, pitch_y, img_x, img_y, track_id)`` per frame (flagged
    ``bridged`` upstream) so the emitted ball path stays continuous. Longer gaps, or
    gaps whose brackets are too far apart to be the same ball, are left null.
    """
    if cfg.bridge_max_gap <= 0:
        return {}
    fps = cfg.fps if cfg.fps > 0 else 25.0
    order = sorted(frames)
    bridged: Dict[int, tuple] = {}

    i, n = 0, len(order)
    while i < n:
        if order[i] in chosen:
            i += 1
            continue
        j = i
        while j < n and order[j] not in chosen:
            j += 1
        if i > 0 and j < n:
            f_lo, f_hi = order[i - 1], order[j]
            if (f_hi - f_lo - 1) <= cfg.bridge_max_gap:
                a, b = chosen[f_lo], chosen[f_hi]
                reach = cfg.ball_max_speed_ms * (f_hi - f_lo) / fps + cfg.continuity_slack_m
                if np.hypot(b.pitch_x - a.pitch_x, b.pitch_y - a.pitch_y) <= reach:
                    for k in range(i, j):
                        f = order[k]
                        t = (f - f_lo) / (f_hi - f_lo)
                        bridged[f] = (
                            a.pitch_x + t * (b.pitch_x - a.pitch_x),
                            a.pitch_y + t * (b.pitch_y - a.pitch_y),
                            a.img_x + t * (b.img_x - a.img_x),
                            a.img_y + t * (b.img_y - a.img_y),
                            a.track_id,
                        )
        i = j
    return bridged
