"""
Hybrid VM Executor (WSE-STD-VM-HYBRID v0)
=========================================

Microcoded block VM with fixed lanes for the Symbol Engine synthesizer.

Executes VoiceFrame as microcode with deterministic, bounded work.
Each frame = one voice for one block.

Usage:
    from symcrash_python_ap.vm import vm_step_block, VMState, DecodedPlan

    # Initialize state for a voice
    state = VMState.create(voice_id=0)

    # Execute one block
    state, output = vm_step_block(state, frame, tables)

Fixed Lane Schedule:
    1. LANE_CARRIER_RENDER - Phase-continuous oscillator
    2. LANE_INJECT - Event injection → coeff_state
    3. LANE_DIAG - Diagonal operators (K-scaling)
    4. LANE_BANDED - Banded operators (K-scaling)
    5. LANE_PERMUTE - Permutation operators (L-scaling)
    6. LANE_LOWRANK - Low-rank operators (L-scaling)
    7. LANE_TAIL - Mandatory tail operators
    8. LANE_IDWPT_RENDER - Wavelet synthesis
    9. LANE_MIX_GAIN - Eta blend + gain ramp

See docs/WSE-STD-VM-HYBRID.md for full specification.
"""

from .state import VMState, EnvelopePhase
from .decode import DecodedPlan, CarrierRole, decode_frame
from .tables import Tables, OPERATOR_LANES, Lane
from .executor import vm_step_block
from .wavetable_engine import (
    WavetableAtlas,
    WavetableBank,
    WavetableState,
    wavetable_generate_coeffs,
    wavetable_generate_block,
    create_basic_waves,
    create_harmonic_series,
    create_default_bank,
)

__all__ = [
    # State
    'VMState',
    'EnvelopePhase',

    # Decode
    'DecodedPlan',
    'CarrierRole',
    'decode_frame',

    # Tables
    'Tables',
    'OPERATOR_LANES',
    'Lane',

    # Executor
    'vm_step_block',

    # Wavetable (Phase 4)
    'WavetableAtlas',
    'WavetableBank',
    'WavetableState',
    'wavetable_generate_coeffs',
    'wavetable_generate_block',
    'create_basic_waves',
    'create_harmonic_series',
    'create_default_bank',
]
