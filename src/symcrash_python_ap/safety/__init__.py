"""Safety Engine - Audio safety, limiting, and click prevention.

Includes OLA, limiting, DC correction, and unified anti-click rendering.
"""

from .safety_engine import (
    SafetyEngine,
    DegradeLevel,
    RegularityOperator,
    OLAState,
    RGState,
    LimiterState,
    CoeffInvariantResult,
    AudioInvariantResult,
    SymbolTail,
)

from .anti_click import (
    AntiClickOperator,
    AntiClickConfig,
    AntiClickState,
)

from .unified_renderer import (
    UnifiedRenderer,
    SynthesisBackend,
    RenderFrame,
    DiscontinuityFlags,
    VoiceState,
    HarmonicScaffoldBackend,
    WaveletBackend,
    StandardWaveletPipeline,
    StandardHarmonicPipeline,
    CarrierLiftPipeline,
)

__all__ = [
    # Main engine
    'SafetyEngine',
    'DegradeLevel',
    'RegularityOperator',
    'OLAState',
    'RGState',
    'LimiterState',
    'CoeffInvariantResult',
    'AudioInvariantResult',
    'SymbolTail',
    # Anti-click
    'AntiClickOperator',
    'AntiClickConfig',
    'AntiClickState',
    # Unified renderer
    'UnifiedRenderer',
    'SynthesisBackend',
    'RenderFrame',
    'DiscontinuityFlags',
    'VoiceState',
    'HarmonicScaffoldBackend',
    'WaveletBackend',
    'StandardWaveletPipeline',
    'StandardHarmonicPipeline',
    'CarrierLiftPipeline',
]
