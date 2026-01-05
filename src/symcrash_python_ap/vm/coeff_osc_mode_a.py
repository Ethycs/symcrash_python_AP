"""
CoeffOsc Mode A: Table-Orbit Oscillator (STANDARD-PATCH.MD §3.1)
=================================================================

Per-coefficient oscillation model where each coefficient has its own
phase accumulator advancing at a specific harmonic relationship.

Runtime equation (per coefficient i):
    phi_i += phase_inc * stride_i
    coeff_base[i] = (mag[i] * sin_lut(phi_i)) >> 15

This generates 64 leaf coefficients directly (not audio!), which then
flow through CMMM operators and IDWPT for final audio reconstruction.

This is the TRUE coefficient-domain carrier - each coefficient oscillates
at its own frequency, creating a phase-continuous coefficient orbit.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

from ..wavelet.sine_core import sine_q15
from ..fixed import RSHIFT_RNE

Q15_ONE = 32767
Q15_HALF = 16384


# =============================================================================
# MODE A TABLE (Per-Coefficient Magnitude + Stride)
# =============================================================================

@dataclass
class CoeffOscTableModeA:
    """
    Mode A table: Per-coefficient magnitude and frequency stride.

    Each coefficient oscillates independently:
        coeff[i] = mag[i] * sin(phi[i])
        phi[i] += phase_inc * stride[i]

    For a saw wave:
        - Coefficient 0 (fundamental): mag=high, stride=1.0
        - Coefficient 1 (2nd harmonic): mag=med, stride=2.0
        - etc.
    """
    table_id: int = 0
    name: str = "default"

    # Per-coefficient magnitude (Q15) - 64 leaves
    mag_q15: np.ndarray = field(default_factory=lambda: np.array([Q15_ONE] + [0]*63, dtype=np.int16))

    # Per-coefficient frequency stride (Q16) - 64 leaves
    # stride = 16384 * harmonic_number
    # Example: stride[0] = 16384 (1x), stride[1] = 32768 (2x)
    stride_q16: np.ndarray = field(default_factory=lambda: np.array([16384]*64, dtype=np.int32))


def build_sine_table_mode_a() -> CoeffOscTableModeA:
    """Pure sine: only first coefficient active."""
    mag = np.zeros(64, dtype=np.int16)
    mag[0] = Q15_ONE
    stride = np.full(64, 16384, dtype=np.int32)
    return CoeffOscTableModeA(table_id=0, name="sine", mag_q15=mag, stride_q16=stride)


def build_saw_table_mode_a() -> CoeffOscTableModeA:
    """
    Sawtooth: First 16 harmonics with 1/n amplitude.

    Maps harmonics to leaves in frequency order.
    """
    mag = np.zeros(64, dtype=np.int16)
    stride = np.zeros(64, dtype=np.int32)

    for h in range(1, 17):
        leaf_idx = h - 1
        mag[leaf_idx] = Q15_ONE // h
        stride[leaf_idx] = 16384 * h  # Harmonic h

    # Normalize to prevent clipping
    total = mag.sum()
    if total > 0:
        scale_q15 = (Q15_ONE << 15) // total
        mag = ((mag.astype(np.int32) * scale_q15) >> 15).astype(np.int16)

    return CoeffOscTableModeA(table_id=1, name="saw", mag_q15=mag, stride_q16=stride)


# Pre-built tables
MODE_A_TABLES = {
    0: build_sine_table_mode_a(),
    1: build_saw_table_mode_a(),
}


def get_mode_a_table(table_id: int) -> CoeffOscTableModeA:
    """Get Mode A table by ID."""
    return MODE_A_TABLES.get(table_id, MODE_A_TABLES[0])


# =============================================================================
# MODE A OSCILLATOR STATE
# =============================================================================

@dataclass
class CoeffOscStateModeA:
    """
    Mode A oscillator state with per-coefficient phase accumulators.

    For leaf-shared optimization (recommended v0):
    - 32 node phases (one per WPT node)
    - 8 sub-leaf phases (for fine detail)
    - Total: 40 x 4 bytes = 160 bytes per voice

    Or full: 64 leaf phases = 256 bytes per voice
    """
    # Per-leaf phase accumulators (64 x 32-bit)
    leaf_phases: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.uint32))

    # Base phase increment (from MIDI note)
    base_phase_inc: int = 0

    # Target phase increment (for glide)
    target_phase_inc: int = 0

    # Phase slew rate per sample
    phase_slew_rate: int = 0

    # Table ID
    table_id: int = 0

    def reset(self, phase_inc: int = 0, table_id: int = 0) -> None:
        """Reset for new note."""
        self.leaf_phases.fill(0)
        self.base_phase_inc = phase_inc
        self.target_phase_inc = phase_inc
        self.phase_slew_rate = 0
        self.table_id = table_id

    def set_pitch(self, phase_inc: int, slew_blocks: int = 8) -> None:
        """Set target pitch with optional slew."""
        if slew_blocks <= 0:
            self.base_phase_inc = phase_inc
            self.target_phase_inc = phase_inc
            self.phase_slew_rate = 0
        else:
            self.target_phase_inc = phase_inc
            diff = phase_inc - self.base_phase_inc
            self.phase_slew_rate = diff // (slew_blocks * 64)


# =============================================================================
# MODE A COEFFICIENT GENERATION
# =============================================================================

def generate_coeffs_mode_a(
    state: CoeffOscStateModeA,
    table: CoeffOscTableModeA,
    output_coeffs: np.ndarray,
) -> None:
    """
    Generate 64 leaf coefficients using Mode A (Table-Orbit Oscillator).

    Per STANDARD-PATCH.MD §3.1:
        For each coefficient i:
            phi_i += phase_inc * stride_i
            coeff[i] = (mag[i] * sin(phi_i)) >> 15

    This generates COEFFICIENTS (not audio), which are then passed through
    CMMM operators and finally IDWPT for audio reconstruction.

    Args:
        state: CoeffOscStateModeA (modified in place)
        table: CoeffOscTableModeA with mag/stride per leaf
        output_coeffs: [64] int16 output buffer

    Returns:
        None (output written to output_coeffs)
    """
    # Apply pitch slew if needed
    phase_inc = state.base_phase_inc
    slew_rate = state.phase_slew_rate
    target = state.target_phase_inc

    if slew_rate != 0:
        new_inc = phase_inc + slew_rate * 64
        if slew_rate > 0 and new_inc >= target:
            new_inc = target
            state.phase_slew_rate = 0
        elif slew_rate < 0 and new_inc <= target:
            new_inc = target
            state.phase_slew_rate = 0
        state.base_phase_inc = new_inc
        phase_inc = new_inc

    # Generate coefficients
    for i in range(64):
        mag = table.mag_q15[i]
        if mag == 0:
            output_coeffs[i] = 0
            continue

        stride = table.stride_q16[i]

        # Advance phase: phi_i += phase_inc * stride_i
        # stride is Q16, phase_inc is Q0.32, result is Q16.32 → keep high 32 bits
        # Use Python int to avoid numpy overflow
        phase_delta = int(((int(phase_inc) * int(stride)) >> 16) & 0xFFFFFFFF)
        new_phase = (int(state.leaf_phases[i]) + phase_delta) & 0xFFFFFFFF
        state.leaf_phases[i] = np.uint32(new_phase)

        # Generate coefficient: coeff = mag * sin(phi)
        sin_val = sine_q15(new_phase)

        # Q15 multiply with proper handling
        # Use Python int to avoid overflow, then shift
        product = int(sin_val) * int(mag)
        coeff_val = RSHIFT_RNE(product, 15)

        # Saturate
        if coeff_val > 32767:
            coeff_val = 32767
        elif coeff_val < -32768:
            coeff_val = -32768

        output_coeffs[i] = np.int16(coeff_val)


__all__ = [
    'CoeffOscTableModeA',
    'CoeffOscStateModeA',
    'generate_coeffs_mode_a',
    'get_mode_a_table',
    'build_sine_table_mode_a',
    'build_saw_table_mode_a',
]
