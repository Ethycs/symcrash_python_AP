"""Wavelet Engine - Audio synthesis via inverse wavelet packet transform.

Includes carrier oscillators, harmonic injection, and tree management.
"""

from .wavelet_engine import (
    WaveletEngine,
    RegularityOperator,
    IDWPTLift,
)

from .wavelet_packet_trees import (
    PacketTreeSet,
    PacketAnalyzer,
    PacketSynthesizer,
    CDF53Lifting,
    ComplexAtomTransform,
)

from .tree_library import (
    TreeLibrary,
    TreeDescriptor,
)

from .tree_compression import (
    TreeStructure,
    TreeSchedule,
    TreeAnalyzer,
    TreeAnalysis,
    WaveletTreeDescriptor,
    TreeEnvelope,
    TreeADSR,
    TreePreset,
    TreeNode,
    Op,
)

from .oscillator import (
    OscillatorManager,
    WPOState,
    HarmonicScaffold,
    PacketTexture,
    AnalogDrift,
    PhaseAccumulator,
    BLEPState,
    TreeArchetype,
    TreeArchetypeConfig,
    TransitionKind,
)

from .carrier import (
    CarrierOscillator,
)

from .carrier_lift import (
    CarrierState,
    EnvelopeState,
    EnvelopeStage,
    BlendState,
    LeafContent,
    InstrumentTuning,
    TuningMorphState,
    CarrierLiftVoiceState,
)

from .harmonic_injector import (
    HarmonicTargetGenerator,
    HarmonicTarget,
    HarmonicInjectorState,
    WaveType,
)

from .leaf_generator import (
    LeafGenerator,
    Partial,
)

from .sine_core import (
    OscillatorState,
)

__all__ = [
    # Main engine
    'WaveletEngine',
    'RegularityOperator',
    'IDWPTLift',
    # Packet trees
    'PacketTreeSet',
    'PacketAnalyzer',
    'PacketSynthesizer',
    'CDF53Lifting',
    'ComplexAtomTransform',
    # Tree library
    'TreeLibrary',
    'TreeDescriptor',
    # Tree compression
    'TreeStructure',
    'TreeSchedule',
    'TreeAnalyzer',
    'TreeAnalysis',
    'WaveletTreeDescriptor',
    'TreeEnvelope',
    'TreeADSR',
    'TreePreset',
    'TreeNode',
    'Op',
    # Oscillator
    'OscillatorManager',
    'WPOState',
    'HarmonicScaffold',
    'PacketTexture',
    'AnalogDrift',
    'PhaseAccumulator',
    'BLEPState',
    'TreeArchetype',
    'TreeArchetypeConfig',
    'TransitionKind',
    # Carrier
    'CarrierOscillator',
    # Carrier-Lift
    'CarrierState',
    'EnvelopeState',
    'EnvelopeStage',
    'BlendState',
    'LeafContent',
    'InstrumentTuning',
    'TuningMorphState',
    'CarrierLiftVoiceState',
    # Harmonic injection
    'HarmonicTargetGenerator',
    'HarmonicTarget',
    'HarmonicInjectorState',
    'WaveType',
    # Leaf generator
    'LeafGenerator',
    'Partial',
    # Sine core
    'OscillatorState',
]
