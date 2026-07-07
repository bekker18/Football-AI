"""Configuration for the two-pass controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class TwoPassConfig:
    """Parameters for gating the ball detector onto high-value windows.

    The gate spends a *budget* — at most ``budget_frac`` of the match's frames —
    on the highest-value windows first, so the ball cost is bounded no matter how
    many windows the emitter produced.
    """

    # --- gate / budget ---
    budget_frac: Optional[float] = 0.10  # ball on <= this share of frames (None = no cap)
    max_windows: Optional[int] = None    # optional hard cap on window count
    min_peak_value: float = 0.0          # ignore windows below this peak value

    # --- pass 2 execution (passed through to the ball detector / decode) ---
    ball_imgsz: int = 640
    device: str = "cpu"

    overrides: dict = field(default_factory=dict)

    def as_meta(self) -> dict:
        return asdict(self)
