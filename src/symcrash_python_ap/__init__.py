"""
symcrash_python_ap - Wavelet Synth Engine Python Implementation
================================================================

A deterministic, real-time audio synthesis system built on wavelet packet
decomposition and catastrophe-theoretic operator families.

Engine Architecture:
- Symbol Engine: Coefficient executor (WSE-STD-030)
- Wavelet Engine: Audio synthesis via IDWPT
- Material Engine: Nonlinear signal processing
- Safety Engine: Audio safety and limiting
- Voice Engine: CMMM voice control (WSE-STD-050, WSE-STD-060)
- Comm: UART protocols (WSE-STD-020, WSE-STD-100)
"""

# Core engines - re-export main classes for convenience
from .symbol import SymbolEngine, SymbolExecutor
from .wavelet import WaveletEngine, PacketTreeSet, HarmonicTargetGenerator, WaveType
from .material import MaterialEngine
from .safety import SafetyEngine, UnifiedRenderer
from .voice import VoiceEngine, VoiceBank, BlockScheduler, BlockOrchestrator

# Foundation modules
from .fixed import (
    sat16, mul_q15_sat, slew_q15,
    Q15_ONE, Q15_HALF, Q15_SHIFT
)
from .frames import (
    VoiceFrame, ParamFrame, VoiceFramePacketV0b, VoiceFlags
)
from .prng import PCG32, VoicePRNG

# Communication
from .comm import VoiceFrameV0, DualUARTLink, UARTPacket

# Package metadata
__version__ = "0.1.0"
__author__ = "Keytone"

__all__ = [
    # Engines
    'SymbolEngine',
    'SymbolExecutor',
    'WaveletEngine',
    'MaterialEngine',
    'SafetyEngine',
    'VoiceEngine',
    'VoiceBank',
    # Rendering
    'PacketTreeSet',
    'HarmonicTargetGenerator',
    'WaveType',
    'UnifiedRenderer',
    # Scheduling
    'BlockScheduler',
    'BlockOrchestrator',
    # Foundation
    'sat16',
    'mul_q15_sat',
    'slew_q15',
    'Q15_ONE',
    'Q15_HALF',
    'Q15_SHIFT',
    # Frames
    'VoiceFrame',
    'ParamFrame',
    'VoiceFramePacketV0b',
    'VoiceFlags',
    # PRNG
    'PCG32',
    'VoicePRNG',
    # Communication
    'VoiceFrameV0',
    'DualUARTLink',
    'UARTPacket',
]
