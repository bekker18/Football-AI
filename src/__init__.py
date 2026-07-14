"""The football-ai package: one subpackage per stage.

Layer 1 (pixels)::

    cv/             video -> raw game state (torch / YOLO / SigLIP)

Layer 2 (tables in, tables out — no pixels, no GPU)::

    prerequisites/  raw game state -> event-ready game state
    possession/     per-frame ball possessor
    actions/        possession transitions -> SPADL actions
    events/         ball-free high-value windows (Door 2)
    twopass/        the ball-only-where-it-matters controller
    eval/           detection/tracking metrics against SoccerNet GSR

Nothing is imported here: ``import src`` must stay free of the heavy Layer 1
stack so the Layer 2 image (no torch, no checkpoints) can import its own
subpackages. Reach for the stage you want, e.g. ``from src.cv import run``.
"""

from __future__ import annotations

__version__ = "0.1.0"
