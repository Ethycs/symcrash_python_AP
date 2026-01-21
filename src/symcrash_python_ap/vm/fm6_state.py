"""
FM6 State - 6-Operator FM Synthesis State
==========================================

Per WSE-STD-FM-TOPOLOGY-v0:
- FMOperator: Per-operator state and parameters
- FM6State: Runtime state for 6-operator FM voice
- All arithmetic in Q15/Q16/Q32 fixed-point
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .tables import VM_K

# Import canonical constants from primitives (single source of truth)
from ..primitives import Q15_ONE, Q16_ONE

# Constants
# VM_K imported from tables.py for consistency
# Q15_ONE and Q16_ONE imported from primitives
Q32_MASK = 0xFFFFFFFF


@dataclass
class FMOperator:
    """
    Single FM operator with DX7-compatible parameters.

    All levels/amplitudes in Q15, frequency ratios in Q16, phase in Q32.
    """
    # Identity
    op_id: int = 0                    # 0-5 (operator index)

    # Frequency
    ratio_q16: int = Q16_ONE          # Frequency ratio (Q16: 65536 = 1.0×)
    detune_cents: int = 0             # Fine detune (-99 to +99 cents)

    # Level
    level_q15: int = Q15_ONE          # Output level (Q15)
    velocity_sens: int = 0            # Velocity sensitivity (0-7)

    # Feedback (only meaningful if this is feedback_op)
    feedback_q15: int = 0             # Self-modulation depth (Q15)

    # Phase state
    phase: int = 0                    # Q32 phase accumulator

    # Envelope (DX7-style rate/level) - simplified for now
    env_rates: List[int] = field(default_factory=lambda: [99, 99, 99, 99])
    env_levels: List[int] = field(default_factory=lambda: [99, 99, 99, 0])
    env_phase: int = 0                # Current envelope segment (0-3)
    env_value_q15: int = Q15_ONE      # Current envelope value (Q15)

    def reset(self):
        """Reset operator phase and envelope."""
        self.phase = 0
        self.env_phase = 0
        self.env_value_q15 = Q15_ONE

    def compute_phase_inc(self, base_phase_inc: int) -> int:
        """
        Compute phase increment for this operator.

        Args:
            base_phase_inc: Base phase increment (Q32) for fundamental

        Returns:
            Operator phase increment (Q32) including ratio and detune
        """
        # Apply frequency ratio (Q16)
        phase_inc = (base_phase_inc * self.ratio_q16) >> 16

        # Apply detune if non-zero
        if self.detune_cents != 0:
            # cents_to_ratio = 2^(cents/1200)
            # Approximate: 1 cent ≈ 0.000578 ratio change
            # For ±99 cents: ratio change ≈ ±5.7%
            detune_factor = Q16_ONE + (self.detune_cents * 38)  # ~38 = 65536 * 0.000578
            phase_inc = (phase_inc * detune_factor) >> 16

        return phase_inc & Q32_MASK


@dataclass
class FM6State:
    """
    Runtime state for 6-operator FM synthesis.

    Holds 6 operators plus algorithm selection and intermediate buffers.
    """
    # Operators (always 6)
    operators: List[FMOperator] = field(default_factory=lambda: [
        FMOperator(op_id=i) for i in range(6)
    ])

    # Algorithm selection
    algorithm_id: int = 1             # DX7 algorithm (1-32)

    # Per-block intermediate storage
    # Shape: [6, K] for lane-rate operator outputs
    op_outputs: np.ndarray = field(default_factory=lambda: np.zeros((6, VM_K), dtype=np.int32))

    # Feedback buffer (previous block's feedback op output)
    # Shape: [K] for z⁻¹ block delay
    feedback_buffer: np.ndarray = field(default_factory=lambda: np.zeros(VM_K, dtype=np.int32))

    # Master controls
    master_level_q15: int = Q15_ONE   # Overall output scaling (Q15)
    master_pitch_bend: int = 0        # Pitch bend (cents, ±8192 typical)

    # Post-filter (optional one-pole lowpass)
    # When enabled, softens harsh FM harmonics (like vintage DX7 DAC filter)
    post_filter_enabled: bool = False
    post_filter_cutoff_q15: int = 24576  # ~0.75 coefficient (gentle rolloff)
    _post_filter_state: int = 0          # Filter state (z⁻¹)

    def reset(self):
        """Reset all operators and clear buffers."""
        for op in self.operators:
            op.reset()
        self.op_outputs.fill(0)
        self.feedback_buffer.fill(0)
        self._post_filter_state = 0

    def set_algorithm(self, alg_id: int):
        """Set algorithm ID (1-32)."""
        if 1 <= alg_id <= 32:
            self.algorithm_id = alg_id
        else:
            raise ValueError(f"Algorithm ID must be 1-32, got {alg_id}")

    def set_operator_ratio(self, op_idx: int, ratio_q16: int):
        """Set operator frequency ratio."""
        if 0 <= op_idx < 6:
            self.operators[op_idx].ratio_q16 = ratio_q16

    def set_operator_level(self, op_idx: int, level_q15: int):
        """Set operator output level."""
        if 0 <= op_idx < 6:
            self.operators[op_idx].level_q15 = level_q15

    def set_operator_feedback(self, op_idx: int, feedback_q15: int):
        """Set operator feedback depth (only effective for feedback operator)."""
        if 0 <= op_idx < 6:
            self.operators[op_idx].feedback_q15 = feedback_q15

    def configure_from_frame(self,
                              algorithm_id: int = 1,
                              ratios: Optional[List[int]] = None,
                              levels: Optional[List[int]] = None,
                              feedback_depth: int = 0):
        """
        Configure FM6 state from VoiceFrame parameters.

        Args:
            algorithm_id: Algorithm (1-32)
            ratios: List of 6 Q16 frequency ratios (default: all 1.0)
            levels: List of 6 Q15 output levels (default: all max)
            feedback_depth: Feedback level 0-7 (mapped to Q15)
        """
        self.algorithm_id = algorithm_id

        # Set ratios (default: all 1.0)
        if ratios:
            for i, r in enumerate(ratios[:6]):
                self.operators[i].ratio_q16 = r
        else:
            for op in self.operators:
                op.ratio_q16 = Q16_ONE

        # Set levels (default: all max)
        if levels:
            for i, l in enumerate(levels[:6]):
                self.operators[i].level_q15 = l
        else:
            for op in self.operators:
                op.level_q15 = Q15_ONE

        # Map DX7 feedback level (0-7) to Q15
        # Level 0 = 0, Level 7 = 32767
        feedback_q15 = (feedback_depth * Q15_ONE) // 7 if feedback_depth > 0 else 0

        # Import here to avoid circular dependency
        from .fm_algorithms import get_algorithm
        alg = get_algorithm(algorithm_id)

        # Set feedback on the designated operator
        for op in self.operators:
            op.feedback_q15 = 0
        if alg.feedback_op >= 0:
            self.operators[alg.feedback_op].feedback_q15 = feedback_q15

    def set_post_filter(self, enabled: bool, cutoff_hz: float = 8000.0,
                        sample_rate: int = 48000):
        """
        Configure post-filter for FM output.

        The post-filter is a simple one-pole lowpass that softens harsh
        FM harmonics, similar to the DAC reconstruction filter on vintage
        DX7 hardware.

        Args:
            enabled: Enable/disable post-filter
            cutoff_hz: Filter cutoff frequency in Hz
            sample_rate: Audio sample rate (default 48000)
        """
        self.post_filter_enabled = enabled
        if enabled:
            # One-pole coefficient: a = exp(-2π * fc / fs)
            # For simplicity, use linear approximation for Q15:
            # coeff ≈ 32767 * (1 - 2π * fc / fs)
            import math
            fc_norm = cutoff_hz / sample_rate
            # Clamp to valid range
            fc_norm = max(0.001, min(0.499, fc_norm))
            # One-pole coefficient (inverted for lowpass: coeff = 1 - exp(-2πfc/fs))
            coeff = 1.0 - math.exp(-2.0 * math.pi * fc_norm)
            self.post_filter_cutoff_q15 = int(coeff * Q15_ONE)

    def apply_post_filter(self, audio: 'np.ndarray') -> 'np.ndarray':
        """
        Apply post-filter to FM audio output.

        One-pole lowpass: y[n] = y[n-1] + coeff * (x[n] - y[n-1])

        Args:
            audio: Input audio samples (int16 array)

        Returns:
            Filtered audio samples (int16 array, same shape)
        """
        if not self.post_filter_enabled:
            return audio

        import numpy as np
        output = np.zeros_like(audio)
        state = self._post_filter_state
        coeff = self.post_filter_cutoff_q15

        for i in range(len(audio)):
            # y[n] = y[n-1] + coeff * (x[n] - y[n-1])
            diff = int(audio[i]) - state
            state = state + ((diff * coeff) >> 15)
            # Saturate
            state = max(-32768, min(32767, state))
            output[i] = state

        self._post_filter_state = state
        return output.astype(np.int16)


def create_fm6_state(algorithm_id: int = 1) -> FM6State:
    """
    Factory function to create initialized FM6State.

    Args:
        algorithm_id: Algorithm (1-32)

    Returns:
        Initialized FM6State
    """
    state = FM6State()
    state.set_algorithm(algorithm_id)
    return state


# DX7 frequency ratio presets (common values)
DX7_RATIOS = {
    # Harmonic ratios (Q16)
    'fundamental': Q16_ONE,           # 1.0
    'octave': Q16_ONE * 2,            # 2.0
    'fifth': (Q16_ONE * 3) // 2,      # 1.5
    'fourth': (Q16_ONE * 4) // 3,     # ~1.333
    'major_third': (Q16_ONE * 5) // 4, # 1.25
    'minor_third': (Q16_ONE * 6) // 5, # 1.2

    # Integer ratios (Q16)
    'ratio_1': Q16_ONE,
    'ratio_2': Q16_ONE * 2,
    'ratio_3': Q16_ONE * 3,
    'ratio_4': Q16_ONE * 4,
    'ratio_5': Q16_ONE * 5,
    'ratio_6': Q16_ONE * 6,
    'ratio_7': Q16_ONE * 7,
    'ratio_8': Q16_ONE * 8,

    # Sub-octave
    'half': Q16_ONE // 2,             # 0.5
    'quarter': Q16_ONE // 4,          # 0.25
}


# DX7 feedback level mapping (0-7 to modulation index β)
# Level 7 ≈ β=1.0 (well below chaotic threshold π)
DX7_FEEDBACK_TABLE = [
    0,      # Level 0: no feedback
    4681,   # Level 1: ~0.143
    9362,   # Level 2: ~0.286
    14043,  # Level 3: ~0.429
    18724,  # Level 4: ~0.571
    23405,  # Level 5: ~0.714
    28086,  # Level 6: ~0.857
    32767,  # Level 7: ~1.0
]


def dx7_feedback_to_q15(level: int) -> int:
    """
    Convert DX7 feedback level (0-7) to Q15.

    Args:
        level: DX7 feedback level (0-7)

    Returns:
        Q15 feedback depth
    """
    if 0 <= level <= 7:
        return DX7_FEEDBACK_TABLE[level]
    return 0


__all__ = [
    'FMOperator',
    'FM6State',
    'create_fm6_state',
    'DX7_RATIOS',
    'DX7_FEEDBACK_TABLE',
    'dx7_feedback_to_q15',
    'VM_K',
    'Q15_ONE',
    'Q16_ONE',
    'Q32_MASK',
]
