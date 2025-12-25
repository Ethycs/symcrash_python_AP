"""Material Engine - Nonlinear signal processing.

Includes tape saturation, catastrophe bifurcation, and rank filtering.
"""

from .material_engine import (
    MaterialEngine,
    TapeState,
    CatastropheState,
    RankState,
)

__all__ = [
    'MaterialEngine',
    'TapeState',
    'CatastropheState',
    'RankState',
]
