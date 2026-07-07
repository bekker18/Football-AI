"""Configuration for the Layer 1 benchmark.

Every tunable lives here with a documented default. Distances are in metres (the
pitch frame both prediction and ground truth are expressed in).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Tuple

# Canonical detection-table columns the whole metric engine speaks. Format
# adapters normalise onto exactly these; ``role``/``team`` may be null.
CANON_COLS = ("frame", "track_id", "x", "y", "role", "team")

# Orientation transforms the flip search may try. Our pitch frame's origin/
# orientation relative to ground truth is not known a priori (attacking
# direction is arbitrary — see docs/strategy.md and the prerequisites notes), so
# we evaluate under each candidate and keep the best. Coordinates are assumed to
# lie in [0, length] x [0, width] with the origin at a corner.
VALID_TRANSFORMS = ("none", "rot180", "mirror_x", "mirror_y")


@dataclass
class EvalConfig:
    """Parameters for one evaluation run.

    ``pitch_length_m`` / ``pitch_width_m`` define the shared coordinate frame and
    are only used by the flip transforms; they should match the frame the
    detections are given in (e.g. 105x68 for prepared/target coords).
    """

    # --- shared pitch frame (for the flip transforms) ---
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    # --- matching ---
    match_dist_m: float = 2.0  # a GT/pred pair within this counts as a match (TP)
    # roles evaluated; None => all. Ball/referee excluded by default because the
    # ball is opt-in and refs are not the point of Layer 1 quality.
    roles: Optional[Sequence[str]] = ("player", "goalkeeper")

    # --- orientation search ---
    # transforms applied to the *predictions* before matching; the best-scoring
    # one (by detection F1) is reported. Default tries identity + 180deg rotation.
    flip_candidates: Sequence[str] = ("none", "rot180")

    # --- attributes ---
    # team labels are arbitrary cluster ids (0/1), so team accuracy is measured
    # under the better of the two label permutations (identity vs swapped).
    eval_team: bool = True
    eval_role: bool = True

    # bookkeeping: CLI overrides recorded for the manifest
    overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        for t in self.flip_candidates:
            if t not in VALID_TRANSFORMS:
                raise ValueError(
                    f"unknown flip transform {t!r}; valid: {VALID_TRANSFORMS}"
                )

    def pitch(self) -> Tuple[float, float]:
        return self.pitch_length_m, self.pitch_width_m

    def as_meta(self) -> dict:
        return asdict(self)
