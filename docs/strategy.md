# Strategy — ball-free-first eventing, and how the layers fit

This note records an *architectural decision*, not code. It exists so the choice
below is explicit and deliberate, rather than an accident of "the ball detector
is slow." Read it before adding anything to the event/valuation layers.

## The decision (adopted)

**Ball-free / tracking-native processing is the default for full-match coverage.
Ball-based eventing is reserved for high-value deep-dive clips.**

Everything the framework computes over a whole match (possession proxies,
pressing, formations, pitch control, space) is derived from **player/GK/referee
positions only** — the artifact Layer 1 produces reliably. The ball detector,
which is the per-frame bottleneck and is noisy on broadcast footage (tiled
slicing, small fast object, motion blur), is treated as an *opt-in enrichment*
for selected windows, never a prerequisite for coverage.

This is already the de-facto posture of the codebase (`--ball` is off by default;
the README calls Layer 2 / Door 2 "ball-free-friendly"). This note promotes that
from an implementation side-effect to a committed strategy.

### Why

- **Coverage vs. cost.** A full half is ~67k frames at 25 fps. Ball detection is
  the dominant cost and the dominant *error* source on broadcast video. Paying it
  everywhere buys the least reliable signal at the highest price.
- **Layer 1 already gives us the reliable half.** Positions + team labels +
  homography are good enough to drive tracking-native surfaces without a ball.
- **Graceful degradation.** On zoom-ins / replays where homography drops out and
  the ball is unfindable, ball-free metrics degrade smoothly; ball-based ones
  fall off a cliff.

## The two doors

| | Door 2 — tracking-native (default) | Door 1 — ball-based eventing (opt-in) |
|---|---|---|
| Input | player/GK/ref positions | positions **+** ball track |
| Coverage | whole match | selected high-value windows |
| Produces | possession proxy, pressing, formations, pitch control, space | on-ball events (pass/shot/reception), SPADL, xT/VAEP/xG |
| Cost | cheap, one pass | expensive (ball detector + tiled slicing) |
| Fails on | little (degrades smoothly) | zoom/replay/occlusion (ball lost) |

Door 2 is the spine. Door 1 hangs off it where the extra cost is justified by
the value of the moment.

## The two-pass controller (built — see `src/twopass/`)

The concrete mechanism that makes "reserve ball-based eventing for high-value
clips" automatic instead of manual:

1. **Pass 1 — cheap, ball-free, whole match.** Run Layer 1 in player-only mode
   and the tracking-native eventing pass. Emit, among its outputs, a stream of
   **candidate high-value windows** (frame ranges): likely possessions,
   final-third entries, shots-on-goal geometry, transitions.
2. **Gate.** Rank/threshold those windows.
3. **Pass 2 — expensive, ball-based, only the flagged windows.** Re-decode only
   those frame ranges and run the ball detector + on-ball eventing there.

Net effect: you pay the ball cost on a few percent of the match instead of all
of it, and you get valued on-ball events exactly where they matter.

**Prerequisites for building it (why it was *last*, not first):**
- Pass 1 must already produce the window signals — that is the ball-free eventing
  layer (Step 3 below). You cannot build the gate before the thing that feeds it.
- Layer 1 must support a **windowed second decode** (run the ball detector on an
  arbitrary frame range). `src/twopass/controller.py:run_ball_on_frames` now does
  exactly this — it decodes the clip but runs the pitch + ball models only on the
  planned frames, reusing the existing `BallDetector` and homography, and emits a
  sparse ball table in the Layer 1 ball-row schema.

The gate itself (`src/twopass/plan.py`) is pure logic and unit-tested: it selects
windows highest-value-first under a frame budget (`budget_frac`), unions their
frames, and — crucially — truncates an oversized top window around its core
rather than skipping it (so a single big window never yields zero coverage).

## Build order (committed sequencing)

1. **Explicit strategy** — this document. *(done)*
2. **Layer 1 quality benchmark** (`src/eval`) — GSR-style evaluation of the
   extraction against ground truth. We currently have *no quantitative measure*
   of Layer 1 quality; everything above it is built on numbers we should trust
   only once we can measure them. Highest-leverage, lowest-risk, so it is first.
3. **Ball-free eventing pass** (`src/events`) — possession proxy, pressing,
   formations, pitch control from positions; standardized via kloppy; emits the
   candidate high-value windows Pass 1 needs.
4. **Two-pass controller** — the gate + the windowed second decode that runs the
   ball detector only on flagged windows.

Valuation (SPADL → socceraction: xT / VAEP / xG) sits downstream of Step 3/4,
consuming emitted on-ball events; it is not on this critical path.

## Relationship to the online / offline product modes

The eventual product has two modes (see `todo.txt`):

- **Offline** (implemented direction): whole halves / full matches, all metrics.
  The two-pass controller is what makes full-match offline runs affordable — the
  ball cost is confined to flagged windows.
- **Online** (future): streaming translation with a live mini-map and running
  metrics. Online mode is **Door 2 by construction** — it cannot afford a second
  ball pass at low latency, so it runs the cheap ball-free surfaces live and, at
  most, a single-pass ball detector on the current window. The ball-free-first
  decision is what makes an online mode feasible at all.

The through-line: **make the cheap, reliable, ball-free signal the product's
backbone; treat the ball as targeted enrichment.** Both modes fall out of that.
