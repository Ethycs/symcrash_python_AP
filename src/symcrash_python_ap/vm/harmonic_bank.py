"""
Harmonic Bank - Mode B2 Coefficient Oscillator (WSE-STD-210)
============================================================

Mode B2 (Harmonic Basis Bank) generates coefficients dynamically from
precomputed basis frames and runtime harmonic weights.

Per WSE-STD-210 Section 3.2.1, Mode B2 is REQUIRED for:
- PWM modulation
- Audio-rate FM/PM

Architecture:
    basis_frames[anchor] = (DC_frame, Ucos_h[], Usin_h[])
    weights = (dc_q15, a_h[], b_h[])
    coeff_frame = dc*DC_frame + Σ_h (wcos*Ucos_h + wsin*Usin_h)

Where:
    wcos = b_h*cos(hφ) - a_h*sin(hφ)
    wsin = b_h*sin(hφ) + a_h*cos(hφ)

This implements the complex rotation identity from Section 3.2.1:
    b_h·cos(hφ) + a_h·sin(hφ) = Re{(b_h - j·a_h)·e^(jhφ)}
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import numpy as np

from ..fixed import RSHIFT_RNE, SAT16, Q15_ONE
from ..wavelet.sine_core import sin_cos_q15, freq_to_phase_inc, PHASE_MASK
from .tables import COEFF_LEN

# =============================================================================
# CONSTANTS (WSE-STD-210 Section 1)
# =============================================================================

VM_K = 8                    # Polyphase lanes / within-block offsets
NUM_LEAVES = 64             # Number of WPT leaves
BLOCK_SIZE = 64             # Samples per block
SAMPLE_RATE = 48000

# Oscillator modes
OSC_MODE_A = 0              # LUT-based (existing coeff_lut_streaming)
OSC_MODE_B = 1              # Rotator/Harmonic Bank (this module)

# Default harmonic count
DEFAULT_H_MAX = 32


# =============================================================================
# MODE B2 BASIS FRAMES (WSE-STD-210 Section 3.2.1)
# =============================================================================

@dataclass
class HarmonicBasisFrames:
    """
    Precomputed basis frames for one pitch anchor.

    Per WSE-STD-210 Section 3.2.1:
    - DC_frame[LEAVES, K]: reconstructs constant +1.0
    - Ucos_h[H, LEAVES, K]: reconstructs cos(2πh·φ)
    - Usin_h[H, LEAVES, K]: reconstructs sin(2πh·φ)

    Memory: (2H + 1) × 1024 bytes per anchor (K=8, int16)
    """
    DC_frame: np.ndarray      # [64, K] Q15
    Ucos_h: np.ndarray        # [H_max, 64, K] Q15
    Usin_h: np.ndarray        # [H_max, 64, K] Q15
    H_max: int
    anchor_note: int          # MIDI note for this anchor
    anchor_freq_hz: float     # Frequency in Hz


@dataclass
class HarmonicWeights:
    """
    Runtime harmonic weights for waveform definition.

    Per WSE-STD-210 Section 3.2.1:
    - dc_q15: DC offset
    - a_h_q15[]: sine weights (a_h coefficient)
    - b_h_q15[]: cosine weights (b_h coefficient)

    Time-domain target: x(φ) = dc + Σ_h (b_h·cos(hφ) + a_h·sin(hφ))
    """
    dc_q15: int = 0
    a_h_q15: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_H_MAX, dtype=np.int16))
    b_h_q15: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_H_MAX, dtype=np.int16))
    H_active: int = 1

    def set_sine(self) -> None:
        """Configure for pure sine wave: H=1, a_1=1, b_1=0."""
        self.dc_q15 = 0
        self.a_h_q15.fill(0)
        self.b_h_q15.fill(0)
        self.a_h_q15[0] = Q15_ONE  # a_1 = sin component
        self.H_active = 1

    def set_saw(self, H: int = 32) -> None:
        """
        Configure for sawtooth wave.

        Fourier series: x(φ) = -2/π × Σ_h ((-1)^h / h) × sin(hφ)
        Simplified: a_h = 1/h with alternating sign
        """
        self.dc_q15 = 0
        self.a_h_q15.fill(0)
        self.b_h_q15.fill(0)
        for h in range(1, min(H + 1, len(self.a_h_q15) + 1)):
            # a_h = (-1)^(h+1) / h, scaled to Q15
            sign = 1 if (h % 2 == 1) else -1
            self.a_h_q15[h - 1] = int(sign * Q15_ONE / h)
        self.H_active = min(H, len(self.a_h_q15))

    def set_square(self, H: int = 32) -> None:
        """
        Configure for square wave (odd harmonics only).

        Fourier series: x(φ) = 4/π × Σ_odd (1/h) × sin(hφ)
        """
        self.dc_q15 = 0
        self.a_h_q15.fill(0)
        self.b_h_q15.fill(0)
        for h in range(1, min(H + 1, len(self.a_h_q15) + 1)):
            if h % 2 == 1:  # Odd harmonics only
                # a_h = 1/h for odd h, scaled to Q15
                self.a_h_q15[h - 1] = int(Q15_ONE / h)
        self.H_active = min(H, len(self.a_h_q15))

    def set_triangle(self, H: int = 32) -> None:
        """
        Configure for triangle wave (odd harmonics, 1/h² rolloff).

        Fourier series: x(φ) = 8/π² × Σ_odd ((-1)^((h-1)/2) / h²) × sin(hφ)
        """
        self.dc_q15 = 0
        self.a_h_q15.fill(0)
        self.b_h_q15.fill(0)
        for h in range(1, min(H + 1, len(self.a_h_q15) + 1)):
            if h % 2 == 1:  # Odd harmonics only
                # a_h = ±1/h² with alternating sign
                k = (h - 1) // 2
                sign = 1 if (k % 2 == 0) else -1
                self.a_h_q15[h - 1] = int(sign * Q15_ONE / (h * h))
        self.H_active = min(H, len(self.a_h_q15))


# =============================================================================
# MODE B STATE (WSE-STD-210)
# =============================================================================

@dataclass
class ModeBState:
    """
    Per-voice Mode B oscillator state.

    Maintains phase accumulator and modulation parameters.
    """
    # Phase accumulator (Q32)
    phase: int = 0
    phase_inc: int = 0
    target_phase_inc: int = 0
    phase_slew_rate: int = 0

    # Current note (for basis frame selection)
    note: int = 60
    cents: int = 0

    # Harmonic configuration
    H_active: int = 16
    H_bandlimit: int = 64  # Computed from pitch

    # Harmonic weights (set by waveform type or PWM)
    weights: HarmonicWeights = field(default_factory=HarmonicWeights)

    # PWM state
    pwm_width_q15: int = 16384  # 50% duty cycle
    pwm_enabled: bool = False

    # FM/PM state
    fm_depth_q15: int = 0
    fm_mod_buffer: Optional[np.ndarray] = None

    # Sync state
    sync_enabled: bool = False
    sync_master_phase: int = 0
    sync_master_inc: int = 0

    # Unison state
    unison_voices: int = 1
    unison_spread_cents: int = 0

    def reset(self, note: int, phase_inc: int) -> None:
        """Reset state for a new note."""
        self.note = note
        self.phase_inc = phase_inc
        self.target_phase_inc = phase_inc
        self.phase_slew_rate = 0
        self.phase = 0

        # Compute bandlimit based on pitch
        freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
        nyquist = SAMPLE_RATE // 2
        self.H_bandlimit = max(1, int(0.9 * nyquist / max(1, freq_hz)))

        # Default to sine wave weights
        self.weights.set_sine()

    def advance_block(self) -> None:
        """Advance phase by one block."""
        mode_b_advance_phase(self)

    def set_pwm_weights(self, width_q15: int) -> None:
        """Set PWM weights for given duty cycle."""
        self.pwm_width_q15 = width_q15
        new_weights = compute_pwm_weights(width_q15, len(self.weights.a_h_q15))
        self.weights.dc_q15 = new_weights.dc_q15
        self.weights.a_h_q15[:] = new_weights.a_h_q15[:]
        self.weights.b_h_q15[:] = new_weights.b_h_q15[:]
        self.weights.H_active = new_weights.H_active


# =============================================================================
# BASIS FRAME GENERATION (WSE-STD-210 Section 6.2)
# =============================================================================

def generate_basis_frames(
    anchor_note: int,
    H_max: int = DEFAULT_H_MAX,
    sample_rate: int = SAMPLE_RATE,
) -> HarmonicBasisFrames:
    """
    Generate Mode B2 basis frames for a pitch anchor.

    Per WSE-STD-210 Section 6.2:
    - Generate time-domain reference blocks using Q32 phase stepping
    - Produce basis frames via FWPT

    CRITICAL: Match Mode A's LUT generation approach!
    Mode A generates each k-slot's block starting at phase = k * phase_inc.
    This means k-slot k reconstructs samples starting at time offset k.

    For Mode B to work correctly with the same phase model:
    - k-slot k's basis must represent the waveform starting at sample k
    - At runtime, we use the SAME phase for all k-slots (like Mode A)
    - The k-offset is baked into each k-slot's basis

    Args:
        anchor_note: MIDI note for this anchor
        H_max: Maximum harmonic count
        sample_rate: Sample rate in Hz

    Returns:
        HarmonicBasisFrames with DC_frame, Ucos_h, Usin_h
    """
    from ..wavelet.fast_lifting import fwpt_jit

    freq_hz = 440.0 * (2.0 ** ((anchor_note - 69) / 12.0))
    phase_inc = freq_to_phase_inc(freq_hz, sample_rate)

    # Allocate arrays
    DC_frame = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
    Ucos_h = np.zeros((H_max, NUM_LEAVES, VM_K), dtype=np.int16)
    Usin_h = np.zeros((H_max, NUM_LEAVES, VM_K), dtype=np.int16)

    # Generate basis frames for each k-slot (within-block offset)
    # MATCH MODE A: Each k-slot's block starts at phase = k * phase_inc
    for k in range(VM_K):
        # DC basis: constant signal (same for all k)
        dc_block = [Q15_ONE] * BLOCK_SIZE
        dc_coeffs = fwpt_jit(dc_block)
        DC_frame[:, k] = dc_coeffs

        # Starting phase for this k-slot (matches Mode A)
        k_phase_offset = (k * phase_inc) & PHASE_MASK

        # Harmonic bases - generate block starting at k_phase_offset
        for h in range(1, H_max + 1):
            cos_block = [0] * BLOCK_SIZE
            sin_block = [0] * BLOCK_SIZE

            phase = k_phase_offset
            for n in range(BLOCK_SIZE):
                h_phase = (phase * h) & PHASE_MASK
                sin_val, cos_val = sin_cos_q15(h_phase)
                cos_block[n] = cos_val
                sin_block[n] = sin_val
                phase = (phase + phase_inc) & PHASE_MASK

            # Transform to coefficients
            cos_coeffs = fwpt_jit(cos_block)
            sin_coeffs = fwpt_jit(sin_block)

            Ucos_h[h - 1, :, k] = cos_coeffs
            Usin_h[h - 1, :, k] = sin_coeffs

    return HarmonicBasisFrames(
        DC_frame=DC_frame,
        Ucos_h=Ucos_h,
        Usin_h=Usin_h,
        H_max=H_max,
        anchor_note=anchor_note,
        anchor_freq_hz=freq_hz,
    )


# =============================================================================
# BASIS FRAME ATLAS (WSE-STD-210 Section 5)
# =============================================================================

@dataclass
class BasisFrameAtlas:
    """
    Cache of Mode B2 basis frames keyed by (wave_type, anchor_id).

    Per WSE-STD-210 Section 5.2:
    - anchor_id is typically octave or coarse semitone bins
    - Each anchor provides basis frames for that pitch region
    """
    frames: Dict[int, HarmonicBasisFrames] = field(default_factory=dict)
    H_max: int = DEFAULT_H_MAX

    def get_frames(self, midi_note: int) -> HarmonicBasisFrames:
        """
        Get basis frames for a given pitch.

        Uses octave-based anchors (A in each octave: notes 9, 21, 33, 45, 57, 69, 81, 93, 105)

        Args:
            midi_note: MIDI note number

        Returns:
            HarmonicBasisFrames for the nearest anchor
        """
        # Compute anchor: A in each octave
        octave = midi_note // 12
        anchor_note = octave * 12 + 9  # A in this octave (A0=21, A1=33, ...)

        # Clamp to valid range
        anchor_note = max(21, min(105, anchor_note))

        if anchor_note not in self.frames:
            self.frames[anchor_note] = generate_basis_frames(anchor_note, self.H_max)

        return self.frames[anchor_note]


# Global atlas (lazy-loaded)
_BASIS_ATLAS: Optional[BasisFrameAtlas] = None


def get_basis_atlas() -> BasisFrameAtlas:
    """Get or create the global basis frame atlas."""
    global _BASIS_ATLAS
    if _BASIS_ATLAS is None:
        _BASIS_ATLAS = BasisFrameAtlas()
    return _BASIS_ATLAS


def clear_basis_atlas() -> None:
    """Clear the cached basis frames (useful after algorithm changes)."""
    global _BASIS_ATLAS
    _BASIS_ATLAS = None


def get_basis_for_note(midi_note: int) -> HarmonicBasisFrames:
    """
    Get basis frames for a given MIDI note.

    Convenience function that uses the global atlas.

    Args:
        midi_note: MIDI note number (0-127)

    Returns:
        HarmonicBasisFrames for the nearest pitch anchor
    """
    return get_basis_atlas().get_frames(midi_note)


# =============================================================================
# COEFFICIENT SYNTHESIS (WSE-STD-210 Section 3.2.1)
# =============================================================================

def mode_b_generate_coeffs(
    state: ModeBState,
    basis: HarmonicBasisFrames,
    output_coeffs: np.ndarray,
) -> None:
    """
    Generate coefficient frame using Mode B2 harmonic bank.

    Per WSE-STD-210 Section 3.2.1, for each lane k:
    1. Compute lane phase: φ_k = phase + k*phase_inc
    2. Compute sin(hφ_k), cos(hφ_k) via SineCore
    3. Modulate basis frames with phase to get time-varying coefficients:
       coeff = a_h * (cos(hφ)*Usin + sin(hφ)*Ucos) + b_h * (cos(hφ)*Ucos - sin(hφ)*Usin)

    The identity: a*sin(hθ+hφ) = a*(sin(hθ)cos(hφ) + cos(hθ)sin(hφ))
    where θ is basis phase and φ is oscillator phase.

    Since basis Usin/Ucos reconstruct sin(hθ)/cos(hθ), we modulate with phase φ.

    Args:
        state: ModeBState with phase, weights, etc.
        basis: HarmonicBasisFrames for current pitch anchor
        output_coeffs: Output array [512] to fill with coefficients
    """
    weights = state.weights
    H = min(state.H_active, state.H_bandlimit, basis.H_max)

    # Convert weights to Python int to avoid overflow
    dc_q15 = int(weights.dc_q15)

    # Use same phase for all k-slots. The time-shift is encoded in the basis
    # frames via time-domain rolling. Each k-slot's basis Usin_k was generated
    # as FWPT(roll(sin_block, k)), so IDWPT(Usin_k) gives the rolled signal.
    # Phase modulation with phi shifts the entire signal by phi.
    phi = state.phase

    for k in range(VM_K):
        # Start with DC contribution
        for leaf in range(NUM_LEAVES):
            idx = leaf * VM_K + k
            dc_contrib = RSHIFT_RNE(dc_q15 * int(basis.DC_frame[leaf, k]), 15)
            output_coeffs[idx] = SAT16(dc_contrib)

        # Add harmonic contributions
        for h in range(1, H + 1):
            # Compute sin(hφ), cos(hφ) for phase modulation
            h_phase = (phi * h) & PHASE_MASK
            sin_phi, cos_phi = sin_cos_q15(h_phase)

            # Get harmonic weights (convert to Python int to avoid overflow)
            # a_h = sine coefficient (weight on sin(hθ))
            # b_h = cosine coefficient (weight on cos(hθ))
            a_h = int(weights.a_h_q15[h - 1])
            b_h = int(weights.b_h_q15[h - 1])

            # For a_h * sin(hθ + hφ):
            #   = a_h * (sin(hθ)cos(hφ) + cos(hθ)sin(hφ))
            #   = a_h*cos(hφ)*Usin + a_h*sin(hφ)*Ucos
            #
            # For b_h * cos(hθ + hφ):
            #   = b_h * (cos(hθ)cos(hφ) - sin(hθ)sin(hφ))
            #   = b_h*cos(hφ)*Ucos - b_h*sin(hφ)*Usin
            #
            # Combined coefficients for Usin and Ucos:
            #   Usin_weight = a_h*cos(hφ) - b_h*sin(hφ)
            #   Ucos_weight = a_h*sin(hφ) + b_h*cos(hφ)

            # Compute combined weights with proper scaling
            w_sin = RSHIFT_RNE(a_h * cos_phi - b_h * sin_phi, 15)
            w_cos = RSHIFT_RNE(a_h * sin_phi + b_h * cos_phi, 15)

            # Add basis contributions
            for leaf in range(NUM_LEAVES):
                idx = leaf * VM_K + k
                sin_contrib = RSHIFT_RNE(w_sin * int(basis.Usin_h[h - 1, leaf, k]), 15)
                cos_contrib = RSHIFT_RNE(w_cos * int(basis.Ucos_h[h - 1, leaf, k]), 15)
                output_coeffs[idx] = SAT16(int(output_coeffs[idx]) + sin_contrib + cos_contrib)


def mode_b_advance_phase(state: ModeBState) -> None:
    """
    Advance phase by one block.

    Per WSE-STD-210, phase advances by BLOCK_SIZE * phase_inc per block.
    """
    # Apply pitch slew if active
    if state.phase_slew_rate != 0:
        if state.phase_slew_rate > 0:
            state.phase_inc = min(state.target_phase_inc,
                                  state.phase_inc + state.phase_slew_rate)
        else:
            state.phase_inc = max(state.target_phase_inc,
                                  state.phase_inc + state.phase_slew_rate)
        if state.phase_inc == state.target_phase_inc:
            state.phase_slew_rate = 0

    # Advance phase by block
    state.phase = (state.phase + BLOCK_SIZE * state.phase_inc) & PHASE_MASK


# =============================================================================
# PWM WEIGHTS (WSE-STD-210 Section 8.1)
# =============================================================================

def compute_pwm_weights(width_q15: int, H_max: int = DEFAULT_H_MAX) -> HarmonicWeights:
    """
    Compute harmonic weights for PWM at given duty cycle.

    Per WSE-STD-210 Section 8.1, PWM weights are computed analytically:
    - dc = 2d - 1
    - a_h = (2/πh) × sin(πhd)  (sine component)
    - b_h = 0 for standard pulse

    For asymmetric pulses, both a_h and b_h may be non-zero.

    Args:
        width_q15: Duty cycle in Q15 (0 = 0%, 32767 = ~100%)
        H_max: Maximum harmonic count

    Returns:
        HarmonicWeights configured for the given PWM width
    """
    d = width_q15 / 32768.0  # Duty cycle 0-1

    weights = HarmonicWeights()
    weights.a_h_q15 = np.zeros(H_max, dtype=np.int16)
    weights.b_h_q15 = np.zeros(H_max, dtype=np.int16)

    # DC offset: dc = 2d - 1
    weights.dc_q15 = SAT16(int(round((2.0 * d - 1.0) * Q15_ONE)))

    # Harmonic weights
    for h in range(1, H_max + 1):
        # Pulse Fourier series:
        # a_h = (2/πh) × sin(πhd)
        scale = 2.0 / (np.pi * h)
        a_val = scale * np.sin(np.pi * h * d)
        weights.a_h_q15[h - 1] = SAT16(int(round(a_val * Q15_ONE)))

    weights.H_active = H_max
    return weights


# =============================================================================
# BANDLIMIT CALCULATION (WSE-STD-210 Section 7.1)
# =============================================================================

def compute_bandlimit_H(freq_hz: float, sample_rate: int = SAMPLE_RATE) -> int:
    """
    Compute maximum harmonic count to stay below Nyquist.

    Per WSE-STD-210 Section 7.1:
    H ≤ floor(fs/2 / f0)

    With 0.9× safety margin for PWM/FM.

    Args:
        freq_hz: Fundamental frequency in Hz
        sample_rate: Sample rate in Hz

    Returns:
        Maximum safe harmonic count
    """
    nyquist = sample_rate // 2
    return max(1, int(0.9 * nyquist / max(1, freq_hz)))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'VM_K',
    'NUM_LEAVES',
    'BLOCK_SIZE',
    'SAMPLE_RATE',
    'OSC_MODE_A',
    'OSC_MODE_B',
    'DEFAULT_H_MAX',
    # Data structures
    'HarmonicBasisFrames',
    'HarmonicWeights',
    'ModeBState',
    'BasisFrameAtlas',
    # Functions
    'generate_basis_frames',
    'get_basis_atlas',
    'get_basis_for_note',
    'mode_b_generate_coeffs',
    'mode_b_advance_phase',
    'compute_pwm_weights',
    'compute_bandlimit_H',
]
