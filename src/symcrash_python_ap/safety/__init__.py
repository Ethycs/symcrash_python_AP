"""Safety Engine - Audio safety, limiting, and click prevention.

Includes OLA, limiting, DC correction, and the UnifiedOutputStage closure layer.

The UnifiedOutputStage (formerly CarrierLiftPipeline) is the single authority
for ensuring no discontinuities reach the audio output.
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
    # Main class (v2 architecture)
    UnifiedOutputStage,
    # Backwards compatibility aliases
    CarrierLiftPipeline,
    UnifiedRenderer,  # Deprecated
    # Backend protocol and adapters
    SynthesisBackend,
    HarmonicScaffoldBackend,
    WaveletBackend,
    # Deprecated data class
    RenderFrame,
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
    # Unified output stage (v2 architecture)
    'UnifiedOutputStage',
    # Backwards compatibility
    'CarrierLiftPipeline',
    'UnifiedRenderer',  # Deprecated
    'SynthesisBackend',
    'HarmonicScaffoldBackend',
    'WaveletBackend',
    'RenderFrame',  # Deprecated
]
