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
from .tables import COEFF_LEN, VM_K

# =============================================================================
# CONSTANTS (WSE-STD-210 Section 1)
# =============================================================================

# Polyphase configuration imported from tables.py (VM_K = 4 or 8)
# K_STRIDE: sample offset per slot for basis frame generation
# Both K=4 and K=8 use 1-sample phase offsets for interleaved reconstruction
K_STRIDE = 1
VM_K_RECONSTRUCT = VM_K  # For API compatibility

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
        With proper normalization: a_h = (2/π) × ((-1)^(h+1) / h)
        """
        import math
        self.dc_q15 = 0
        self.a_h_q15.fill(0)
        self.b_h_q15.fill(0)
        for h in range(1, min(H + 1, len(self.a_h_q15) + 1)):
            # a_h = (2/π) × (-1)^(h+1) / h
            # Sign pattern: +, -, +, -, ... for h = 1, 2, 3, 4, ...
            sign = -1 if (h % 2 == 0) else 1
            amplitude = (2.0 / math.pi) * (1.0 / h)
            self.a_h_q15[h - 1] = SAT16(int(sign * amplitude * Q15_ONE))
        self.H_active = min(H, len(self.a_h_q15))

    def set_square(self, H: int = 32, sigma: bool = False) -> None:
        """
        Configure for square wave (odd harmonics only, normalized).

        Raw Fourier series: x(φ) = 4/π × Σ_odd (1/h) × sin(hφ)
        Normalized (a_1=1): a_h = 1/h for odd h

        Args:
            H: Maximum harmonic count
            sigma: Apply Lanczos sigma factor sinc(h/H) to reduce Gibbs ringing
        """
        import math
        self.dc_q15 = 0
        self.a_h_q15.fill(0)
        self.b_h_q15.fill(0)
        for h in range(1, min(H + 1, len(self.a_h_q15) + 1)):
            if h % 2 == 1:  # Odd harmonics only
                amplitude = 1.0 / h
                if sigma and h < H:
                    x = math.pi * h / H
                    amplitude *= math.sin(x) / x
                self.a_h_q15[h - 1] = SAT16(int(amplitude * Q15_ONE))
        self.H_active = min(H, len(self.a_h_q15))

    def set_triangle(self, H: int = 32) -> None:
        """
        Configure for triangle wave (odd harmonics, 1/h² rolloff).

        Fourier series: x(φ) = 8/π² × Σ_odd ((-1)^((h-1)/2) / h²) × sin(hφ)
        With proper normalization: a_h = (8/π²) × ((-1)^k / h²) for h = 2k+1
        """
        import math
        self.dc_q15 = 0
        self.a_h_q15.fill(0)
        self.b_h_q15.fill(0)
        for h in range(1, min(H + 1, len(self.a_h_q15) + 1)):
            if h % 2 == 1:  # Odd harmonics only
                # a_h = (8/π²) × (-1)^k / h² where h = 2k+1
                k = (h - 1) // 2
                sign = 1 if (k % 2 == 0) else -1
                amplitude = (8.0 / (math.pi * math.pi)) / (h * h)
                self.a_h_q15[h - 1] = SAT16(int(sign * amplitude * Q15_ONE))
        self.H_active = min(H, len(self.a_h_q15))

    def set_pwm(self, duty_q15: int = 16384, H: int = 32) -> None:
        """
        Configure for PWM/pulse wave with given duty cycle.

        Uses analytical Fourier coefficients for bandlimited PWM:
        - dc = 2d - 1
        - a_h = (2/πh) × sin(πhd)

        Args:
            duty_q15: Duty cycle in Q15 (16384 = 50%, 8192 = 25%)
            H: Maximum harmonic count
        """
        from .waveform_matrix import compute_pwm_harmonics
        dc, a_h, b_h = compute_pwm_harmonics(duty_q15, H)
        self.dc_q15 = dc
        self.a_h_q15[:len(a_h)] = a_h
        self.b_h_q15[:len(b_h)] = b_h
        self.H_active = min(H, len(self.a_h_q15))

    def set_custom(self, harmonic_amps_q15: list) -> None:
        """
        Configure with custom harmonic amplitudes.

        Sets sine-phase harmonics (a_h) directly from provided Q15 amplitudes.
        This enables arbitrary timbres like violin, brass, woodwinds, etc.

        The harmonics are pure sine-phase (a_h only, b_h=0), which produces
        a waveform that starts at zero crossing. For most musical sounds
        this is appropriate since phase relationships don't affect timbre
        perception significantly.

        Args:
            harmonic_amps_q15: List of Q15 amplitudes [H1, H2, H3, ...]
                              Index 0 = fundamental, index 1 = 2nd harmonic, etc.
        """
        self.dc_q15 = 0
        self.a_h_q15.fill(0)
        self.b_h_q15.fill(0)

        # Copy provided amplitudes to sine weights
        for i, amp in enumerate(harmonic_amps_q15):
            if i < len(self.a_h_q15):
                self.a_h_q15[i] = SAT16(int(amp))

        self.H_active = min(len(harmonic_amps_q15), len(self.a_h_q15))

    def set_from_waveform_type(self, wave_type: int, H: int = 32, duty_q15: int = 16384) -> None:
        """
        Configure from WaveformType constant.

        Provides unified interface to all waveforms:
        - SINE: Pure sine (H=1)
        - SQUARE: 50% duty square
        - PULSE: Variable duty pulse
        - SAW: Sawtooth
        - TRIANGLE: Triangle

        Args:
            wave_type: WaveformType constant (from waveform_matrix)
            H: Maximum harmonic count
            duty_q15: Duty cycle for PULSE type
        """
        from .waveform_matrix import WaveformType, get_theoretical_harmonics

        if wave_type == WaveformType.SINE:
            self.set_sine()
        elif wave_type == WaveformType.SQUARE:
            self.set_square(H)
        elif wave_type == WaveformType.PULSE:
            self.set_pwm(duty_q15, H)
        elif wave_type == WaveformType.SAW or wave_type == WaveformType.SAW_DOWN:
            self.set_saw(H)
        elif wave_type == WaveformType.TRIANGLE:
            self.set_triangle(H)
        else:
            # For unknown types, use DFT decomposition of c vector
            dc, a_h, b_h = get_theoretical_harmonics(wave_type, H, duty_q15)
            self.dc_q15 = dc
            self.a_h_q15[:len(a_h)] = a_h
            self.b_h_q15[:len(b_h)] = b_h
            self.H_active = min(H, len(self.a_h_q15))


# =============================================================================
# MODE B STATE (WSE-STD-210)
# =============================================================================

@dataclass
class VibratoState:
    """
    Vibrato LFO state for expressive pitch/amplitude/timbre modulation.

    Supports delayed onset with ramp-in for natural performance feel.
    All depths are in Q15 format (32767 = 100%).
    """
    # LFO configuration
    enabled: bool = False
    rate_hz: float = 5.5              # LFO frequency (typical: 4-7 Hz)
    phase: int = 0                     # LFO phase accumulator (Q32)
    phase_inc: int = 0                 # LFO phase increment per sample

    # Modulation depths (Q15: 0-32767)
    depth_pitch_cents: int = 0         # Pitch vibrato depth in cents (typ: 20-50)
    depth_amp_q15: int = 0             # Amplitude modulation depth
    depth_timbre_q15: int = 0          # Timbre shift depth (H2/H4 energy exchange)

    # Envelope (delayed onset with ramp)
    delay_samples: int = 0             # Delay before vibrato starts
    ramp_samples: int = 0              # Ramp-in duration after delay
    elapsed_samples: int = 0           # Samples since note-on

    # Runtime state
    current_depth_scale: int = 0       # Current envelope level (Q15, 0-32767)

    def configure(
        self,
        rate_hz: float = 5.5,
        depth_pitch_cents: int = 30,
        depth_amp_q15: int = 0,
        depth_timbre_q15: int = 0,
        delay_ms: float = 40.0,
        ramp_ms: float = 27.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """
        Configure vibrato parameters.

        Args:
            rate_hz: LFO frequency in Hz (typically 4-7 Hz)
            depth_pitch_cents: Pitch modulation depth in cents
            depth_amp_q15: Amplitude modulation depth (Q15)
            depth_timbre_q15: Timbre modulation depth (Q15)
            delay_ms: Delay before vibrato starts (ms)
            ramp_ms: Ramp-in time after delay (ms)
            sample_rate: Sample rate in Hz
        """
        self.enabled = True
        self.rate_hz = rate_hz
        self.depth_pitch_cents = depth_pitch_cents
        self.depth_amp_q15 = depth_amp_q15
        self.depth_timbre_q15 = depth_timbre_q15

        # Compute LFO phase increment
        # phase_inc = rate_hz * 2^32 / sample_rate
        self.phase_inc = int((rate_hz * (1 << 32)) / sample_rate)

        # Compute envelope timing
        self.delay_samples = int(delay_ms * sample_rate / 1000.0)
        self.ramp_samples = max(1, int(ramp_ms * sample_rate / 1000.0))

        # Reset runtime state
        self.phase = 0
        self.elapsed_samples = 0
        self.current_depth_scale = 0

    def reset(self) -> None:
        """Reset vibrato state for new note."""
        self.phase = 0
        self.elapsed_samples = 0
        self.current_depth_scale = 0

    def disable(self) -> None:
        """Disable vibrato."""
        self.enabled = False
        self.current_depth_scale = 0

    def advance_block(self, block_samples: int = BLOCK_SIZE) -> None:
        """
        Advance vibrato LFO by one block.

        Updates phase accumulator and envelope level.
        """
        if not self.enabled:
            return

        # Advance elapsed time
        self.elapsed_samples += block_samples

        # Update envelope (delayed onset with ramp)
        if self.elapsed_samples <= self.delay_samples:
            # Still in delay period
            self.current_depth_scale = 0
        elif self.elapsed_samples <= self.delay_samples + self.ramp_samples:
            # Ramping in
            ramp_progress = self.elapsed_samples - self.delay_samples
            self.current_depth_scale = (ramp_progress * Q15_ONE) // self.ramp_samples
        else:
            # Fully ramped in
            self.current_depth_scale = Q15_ONE

        # Advance LFO phase (per block, using center sample)
        self.phase = (self.phase + self.phase_inc * block_samples) & PHASE_MASK

    def get_lfo_value(self) -> int:
        """
        Get current LFO value scaled by envelope.

        Returns:
            Q15 value in range [-32767, 32767]
        """
        if not self.enabled or self.current_depth_scale == 0:
            return 0

        # Get raw LFO sine value
        sin_val, _ = sin_cos_q15(self.phase)

        # Scale by envelope
        return RSHIFT_RNE(sin_val * self.current_depth_scale, 15)

    def get_pitch_mod_phase_inc_delta(self, base_phase_inc: int) -> int:
        """
        Get phase increment delta for pitch vibrato.

        Pitch modulation in cents: cents = depth * sin(lfo)
        Phase inc delta = base_phase_inc * (2^(cents/1200) - 1)

        For small cents, approximation: delta ≈ base * cents * ln(2) / 1200

        Args:
            base_phase_inc: Base phase increment for the note

        Returns:
            Delta to add to phase_inc for this block
        """
        if not self.enabled or self.depth_pitch_cents == 0:
            return 0

        lfo = self.get_lfo_value()

        # cents_now = depth_pitch_cents * lfo / 32767
        # For ±50 cents at full depth: multiplier range is [0.9715, 1.0293]
        #
        # Using fixed-point approximation:
        # delta = base_phase_inc * cents * ln(2) / 1200 * lfo / 32767
        # ln(2) ≈ 0.693 ≈ 22713/32768
        #
        # Simplified: delta = base * depth_cents * lfo * 22713 / (1200 * 32767 * 32768)
        #           = base * depth_cents * lfo / 56883507
        #           ≈ base * depth_cents * lfo >> 25.76
        #
        # For accuracy, compute: base * depth * lfo / (1200 * 32767) * ln2_q15
        LN2_Q15 = 22713  # ln(2) in Q15

        # Step by step to avoid overflow:
        # 1. cents_scaled = depth_cents * lfo (max ~50 * 32767 = 1.6M)
        # 2. ratio = cents_scaled * LN2_Q15 / (1200 * 32767) = cents_scaled * 22713 / 39320400
        #          ≈ cents_scaled / 1731
        # 3. delta = base * ratio / 32768

        cents_scaled = self.depth_pitch_cents * lfo  # max ±1.6M
        ratio = cents_scaled // 1731  # Approximation for cents_scaled * ln2 / 1200
        delta = RSHIFT_RNE(base_phase_inc * ratio, 15)

        return delta

    def get_amp_mod_q15(self) -> int:
        """
        Get amplitude modulation factor.

        Returns:
            Q15 multiplier (32767 = 1.0, range varies with depth)
        """
        if not self.enabled or self.depth_amp_q15 == 0:
            return Q15_ONE

        lfo = self.get_lfo_value()

        # amp_mod = 1.0 + depth * lfo / 32767
        # In Q15: result = 32767 + depth * lfo / 32767
        mod = RSHIFT_RNE(self.depth_amp_q15 * lfo, 15)
        return Q15_ONE + mod

    def get_timbre_shift_q15(self) -> int:
        """
        Get timbre shift value for harmonic energy redistribution.

        Returns:
            Q15 shift value for H2/H4 energy exchange
        """
        if not self.enabled or self.depth_timbre_q15 == 0:
            return 0

        lfo = self.get_lfo_value()
        return RSHIFT_RNE(self.depth_timbre_q15 * lfo, 15)


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

    # Vibrato state
    vibrato: VibratoState = field(default_factory=VibratoState)

    def reset(self, note: int, phase_inc: int) -> None:
        """Reset state for a new note (hard reset - zeros phase)."""
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

    def set_pitch_legato(self, note: int, phase_inc: int) -> None:
        """
        Change pitch without resetting phase (legato mode).

        This preserves phase continuity for smooth pitch transitions
        without clicking artifacts. The phase accumulator continues
        from its current value, only the increment changes.

        Args:
            note: New MIDI note number
            phase_inc: New phase increment for the target pitch
        """
        self.note = note
        self.phase_inc = phase_inc
        self.target_phase_inc = phase_inc
        # Note: We do NOT reset self.phase - this is the key to legato!

        # Update bandlimit for new pitch
        freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
        nyquist = SAMPLE_RATE // 2
        self.H_bandlimit = max(1, int(0.9 * nyquist / max(1, freq_hz)))
        # Note: We do NOT reset weights - preserve current waveform

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
    k_stride: int = K_STRIDE,
) -> HarmonicBasisFrames:
    """
    Generate Mode B2 basis frames for a pitch anchor.

    Per WSE-STD-210 Section 6.2:
    - Generate time-domain reference blocks using Q32 phase stepping
    - Produce basis frames via FWPT

    K=4 Polyphase Strategy (Consecutive Blocks):
    --------------------------------------------
    Each slot k handles samples [k*16 : (k+1)*16] of the output block.
    The basis for slot k is generated starting at sample position k*16,
    which corresponds to a phase offset of k * 16 * phase_inc.

    This matches the IDWPT polyphase reconstruction that takes:
        out[0:16]  = idwpt(slot_0)[0:16]
        out[16:32] = idwpt(slot_1)[16:32]
        out[32:48] = idwpt(slot_2)[32:48]
        out[48:64] = idwpt(slot_3)[48:64]

    Args:
        anchor_note: MIDI note for this anchor
        H_max: Maximum harmonic count
        sample_rate: Sample rate in Hz
        k_stride: Sample offset per slot (16 for K=4 consecutive blocks)

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

    # Generate basis frames for each k-slot
    # Slot k handles samples [k*16 : (k+1)*16], so basis starts at sample k*16
    #
    # NOTE: headroom_bits=2 is critical for Mode B accuracy.
    # The int16 quantization in FWPT causes the trigonometric modulation identity
    # to break down for harmonics with non-integer cycles per block.
    # With headroom_bits=1, H5 at 440Hz has RMS error ~1429 and -16 dB 50Hz artifact.
    # With headroom_bits=2, error drops to ~22 and artifact drops to -47 dB.
    headroom = 2

    for k in range(VM_K):
        # DC basis: constant signal (same for all k)
        dc_block = [Q15_ONE] * BLOCK_SIZE
        dc_coeffs = fwpt_jit(dc_block, headroom_bits=headroom)
        DC_frame[:, k] = dc_coeffs

        # Starting phase for this k-slot
        # Phase offset = k * k_stride * phase_inc (k_stride=16 for K=4)
        k_phase_offset = (k * k_stride * phase_inc) & PHASE_MASK

        # Harmonic bases - generate full 64-sample block starting at k_phase_offset
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
            cos_coeffs = fwpt_jit(cos_block, headroom_bits=headroom)
            sin_coeffs = fwpt_jit(sin_block, headroom_bits=headroom)

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
    Cache of Mode B2 basis frames keyed by MIDI note.

    Each note gets its own basis frames with frequency baked in.
    This mirrors Mode A's per-note LUT approach for correctness.
    """
    frames: Dict[int, HarmonicBasisFrames] = field(default_factory=dict)
    H_max: int = DEFAULT_H_MAX

    def get_frames(self, midi_note: int) -> HarmonicBasisFrames:
        """
        Get basis frames for a given pitch.

        IMPORTANT: Basis frames must match the exact target pitch.
        The basis frames encode sin/cos at a specific frequency's phase progression.
        Using an anchor that differs from the target pitch produces wrong output.

        This mirrors Mode A's per-note LUT approach (get_lut_for_note).

        Args:
            midi_note: MIDI note number

        Returns:
            HarmonicBasisFrames for the exact target pitch
        """
        # Clamp to valid range
        midi_note = max(21, min(108, midi_note))

        # Generate per-note (like Mode A's get_lut_for_note)
        if midi_note not in self.frames:
            self.frames[midi_note] = generate_basis_frames(midi_note, self.H_max)

        return self.frames[midi_note]


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
# FM6 BASIS FRAME SUPPORT (Arbitrary Ratio Basis Frames)
# =============================================================================

@dataclass
class RatioBasisFrame:
    """
    Basis frame for a single frequency ratio.

    Used by FM6 to generate coefficient-native output for each operator.
    Each operator has a ratio (e.g., 1.0, 2.0, 1.5) relative to the fundamental.

    Unlike HarmonicBasisFrames (which stores frames for harmonics 1..H),
    this stores frames for a SINGLE ratio.
    """
    Ucos: np.ndarray  # [64, K] Q15 - cos basis
    Usin: np.ndarray  # [64, K] Q15 - sin basis
    ratio_q16: int    # Q16 ratio (65536 = 1.0)
    base_phase_inc: int  # Base phase increment this was generated for


# Cache for ratio basis frames: key = (base_phase_inc, ratio_q16)
_RATIO_BASIS_CACHE: Dict[Tuple[int, int], RatioBasisFrame] = {}


def generate_ratio_basis_frame(
    base_phase_inc: int,
    ratio_q16: int,
) -> RatioBasisFrame:
    """
    Generate basis frames for a single frequency ratio.

    This creates sin/cos basis frames for FM6 operators, allowing
    coefficient-native FM synthesis (no FWPT at runtime).

    The basis frames are generated at the effective frequency:
        effective_phase_inc = (base_phase_inc * ratio_q16) >> 16

    K Polyphase Strategy (Interleaved):
    -----------------------------------
    Each slot k represents a 1-sample phase offset for interleaved reconstruction.
    The IDWPT polyphase uses: out[k::K] = idwpt(slot_k)[0::K]

    With K_STRIDE=1, the basis for slot k starts at sample position k.
    This matches the interleaved polyphase reconstruction pattern.

    Args:
        base_phase_inc: Q32 phase increment for fundamental frequency
        ratio_q16: Q16 frequency ratio (65536 = 1.0x)

    Returns:
        RatioBasisFrame with Ucos and Usin basis arrays
    """
    from ..wavelet.fast_lifting import fwpt_jit

    # Compute effective phase increment for this ratio
    # ratio_q16 = 65536 means 1.0x, so 131072 means 2.0x
    eff_phase_inc = (base_phase_inc * ratio_q16) >> 16

    # Allocate arrays
    Ucos = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
    Usin = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)

    # headroom_bits=2 critical for Mode B accuracy (see generate_basis_frames)
    headroom = 2

    # Generate basis frames for each k-slot using interleaved strategy
    # K_STRIDE=1 matches idwpt_polyphase_block() reconstruction pattern

    for k in range(VM_K):
        # Starting phase for this k-slot (K_STRIDE=1 for interleaved)
        k_phase_offset = (k * K_STRIDE * eff_phase_inc) & PHASE_MASK

        # Generate sin/cos blocks at this ratio's frequency
        cos_block = [0] * BLOCK_SIZE
        sin_block = [0] * BLOCK_SIZE

        phase = k_phase_offset
        for n in range(BLOCK_SIZE):
            sin_val, cos_val = sin_cos_q15(phase)
            cos_block[n] = cos_val
            sin_block[n] = sin_val
            phase = (phase + eff_phase_inc) & PHASE_MASK

        # Transform to coefficients
        cos_coeffs = fwpt_jit(cos_block, headroom_bits=headroom)
        sin_coeffs = fwpt_jit(sin_block, headroom_bits=headroom)

        Ucos[:, k] = cos_coeffs
        Usin[:, k] = sin_coeffs

    return RatioBasisFrame(
        Ucos=Ucos,
        Usin=Usin,
        ratio_q16=ratio_q16,
        base_phase_inc=base_phase_inc,
    )


def get_basis_for_ratio(base_phase_inc: int, ratio_q16: int) -> RatioBasisFrame:
    """
    Get or create basis frames for a frequency ratio.

    This is the main entry point for FM6 coefficient-native synthesis.
    Caches basis frames by (base_phase_inc, ratio_q16) key.

    Args:
        base_phase_inc: Q32 phase increment for fundamental frequency
        ratio_q16: Q16 frequency ratio (65536 = 1.0x)

    Returns:
        RatioBasisFrame for the given ratio
    """
    key = (base_phase_inc, ratio_q16)
    if key not in _RATIO_BASIS_CACHE:
        _RATIO_BASIS_CACHE[key] = generate_ratio_basis_frame(base_phase_inc, ratio_q16)
    return _RATIO_BASIS_CACHE[key]


def clear_ratio_basis_cache() -> None:
    """Clear the ratio basis frame cache."""
    global _RATIO_BASIS_CACHE
    _RATIO_BASIS_CACHE.clear()


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
    1. Compute lane phase: φ_k = phase + k*phase_inc + FM_mod
    2. Compute sin(hφ_k), cos(hφ_k) via SineCore
    3. Modulate basis frames with phase to get time-varying coefficients:
       coeff = a_h * (cos(hφ)*Usin + sin(hφ)*Ucos) + b_h * (cos(hφ)*Ucos - sin(hφ)*Usin)

    FM/PM Support (WSE-STD-210 Section 8.3):
    - fm_mod_buffer[k] provides audio-rate modulator samples per k-slot
    - fm_depth_q15 controls modulation index
    - Phase modulation: φ_mod = fm_depth × modulator_sample

    The identity: a*sin(hθ+hφ) = a*(sin(hθ)cos(hφ) + cos(hθ)sin(hφ))
    where θ is basis phase and φ is oscillator phase.

    Since basis Usin/Ucos reconstruct sin(hθ)/cos(hθ), we modulate with phase φ.

    Args:
        state: ModeBState with phase, weights, fm_depth_q15, fm_mod_buffer
        basis: HarmonicBasisFrames for current pitch anchor
        output_coeffs: Output array [512] to fill with coefficients
    """
    weights = state.weights
    H = min(state.H_active, state.H_bandlimit, basis.H_max)

    # Convert weights to Python int to avoid overflow
    dc_q15 = int(weights.dc_q15)

    # Compute total harmonic energy for normalization
    # This prevents clipping when summing multiple strong harmonics
    total_weight = 0
    for h in range(1, H + 1):
        a_h = abs(int(weights.a_h_q15[h - 1]))
        b_h = abs(int(weights.b_h_q15[h - 1]))
        total_weight += max(a_h, b_h)

    # Scale factor to prevent overflow (sum of harmonics shouldn't exceed ~32767)
    # Use headroom of 2x to be safe
    if total_weight > 32767:
        scale_shift = 0
        while (total_weight >> scale_shift) > 32767:
            scale_shift += 1
    else:
        scale_shift = 0

    # Base phase for this block
    base_phi = state.phase

    # FM/PM modulation: get modulator samples if available
    fm_active = state.fm_depth_q15 != 0 and state.fm_mod_buffer is not None

    for k in range(VM_K):
        # Compute phase for this k-slot with optional FM modulation
        # Per WSE-STD-210 Section 8.3: φ_k = phase + FM_mod
        # Lane-rate update: sample modulator at first output of each lane
        phi = base_phi

        if fm_active:
            # Get modulator sample for this k-slot
            # fm_mod_buffer is [K] samples, one per k-slot (lane-rate)
            mod_sample = int(state.fm_mod_buffer[k]) if k < len(state.fm_mod_buffer) else 0

            # Phase modulation: scale modulator by depth
            # fm_depth_q15 = modulation index in Q15 (32767 ≈ 1.0 radian max)
            # mod_sample = modulator signal (Q15: ±32767 = ±1.0)
            #
            # FM synthesis: phase_mod = β × mod, where β is modulation index
            # In Q32 phase: 1 radian = 2^32 / (2π) ≈ 683,565,275
            #
            # Calculation:
            # depth × mod = Q30 (max ~1.07×10^9)
            # We want depth=32767, mod=32767 → ~683M (1 radian)
            # So: Q30 >> 1 ≈ 535M, close enough for moderate FM
            #
            # For typical FM synth (DX-style), modulation index 0.5-3 is common.
            # fm_depth_q15=16384 (50%) → ~0.5 radians = classic FM bell territory
            phase_mod = RSHIFT_RNE(state.fm_depth_q15 * mod_sample, 1)  # Q30 >> 1 ≈ Q29
            phi = (phi + phase_mod) & PHASE_MASK

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
            # Apply scale_shift to prevent overflow when summing harmonics
            a_h = int(weights.a_h_q15[h - 1]) >> scale_shift
            b_h = int(weights.b_h_q15[h - 1]) >> scale_shift

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


def mode_b_generate_coeffs_matrix(
    state: ModeBState,
    waveform_type: int,
    output_coeffs: np.ndarray,
) -> None:
    """
    Generate coefficient frame using matrix method with precomputed coefficient LUT.

    For piecewise-linear waveforms (SQUARE, SAW, TRIANGLE), this method:
    1. Uses precomputed coefficient LUT (phase → coefficients)
    2. No FWPT at runtime - coefficients already in wavelet domain
    3. Linear interpolation between phase bins for smooth playback

    The LUT stores FWPT(c-vector) for each phase position, computed once at init.

    Args:
        state: ModeBState with phase information
        waveform_type: WaveformType constant (from waveform_matrix)
        output_coeffs: Output array [512] to fill with coefficients
    """
    # Get precomputed coefficient LUT for this waveform type
    coeff_lut = get_waveform_coeff_lut(waveform_type, state.phase_inc)

    # Sample from LUT at phase for each k-slot
    # Each k-slot needs its own phase-shifted coefficients for smooth output
    for k in range(VM_K):
        # Compute phase for this k-slot: phase + k * phase_inc
        phase_k = (state.phase + k * state.phase_inc) & PHASE_MASK
        coeffs_k = coeff_lut.sample_coeffs(phase_k)
        for leaf in range(NUM_LEAVES):
            idx = leaf * VM_K + k
            output_coeffs[idx] = coeffs_k[leaf]


# =============================================================================
# WAVEFORM COEFFICIENT LUT (Precomputed FWPT of c-vectors)
# =============================================================================

@dataclass
class WaveformCoeffLUT:
    """
    Precomputed coefficient LUT for a waveform at a specific frequency.

    Stores FWPT(c-vector samples) at P phase positions for zero-runtime-FWPT synthesis.

    Layout: data[P, 64] where P is number of phase bins, 64 is coefficient count.
    """
    waveform_type: int
    phase_inc: int  # Frequency-specific
    P: int = 256  # Phase resolution (power of 2)
    data: np.ndarray = field(default_factory=lambda: np.zeros((256, 64), dtype=np.int16))

    def sample_coeffs(self, phase: int) -> np.ndarray:
        """
        Sample coefficients at given phase with linear interpolation.

        Args:
            phase: Q32 phase

        Returns:
            64-element coefficient array
        """
        # Convert Q32 phase to bin index and fraction
        shift = 32 - int(np.log2(self.P))
        phase_idx = (phase >> shift) & (self.P - 1)
        phase_idx_next = (phase_idx + 1) & (self.P - 1)

        # Fractional part for interpolation
        frac_mask = (1 << shift) - 1
        frac = (phase & frac_mask) >> (shift - 15)  # Q15 fraction

        # Linear interpolation
        v0 = self.data[phase_idx].astype(np.int32)
        v1 = self.data[phase_idx_next].astype(np.int32)
        diff = v1 - v0
        result = v0 + ((diff * frac + (1 << 14)) >> 15)

        return np.clip(result, -32768, 32767).astype(np.int16)

    @staticmethod
    def generate(waveform_type: int, phase_inc: int, P: int = 256) -> 'WaveformCoeffLUT':
        """
        Generate coefficient LUT for a waveform at specific frequency.

        Args:
            waveform_type: WaveformType constant
            phase_inc: Phase increment per sample (defines frequency)
            P: Number of phase bins

        Returns:
            WaveformCoeffLUT with precomputed coefficients
        """
        from ..wavelet.fast_lifting import fwpt_jit
        from .waveform_matrix import waveform_sample_q15

        data = np.zeros((P, NUM_LEAVES), dtype=np.int16)

        for p in range(P):
            # Phase for this bin
            phase_p = (p << (32 - int(np.log2(P)))) & PHASE_MASK

            # Generate 64 time-domain samples
            block = []
            phase = phase_p
            for _ in range(BLOCK_SIZE):
                sample = waveform_sample_q15(phase, waveform_type)
                block.append(sample)
                phase = (phase + phase_inc) & PHASE_MASK

            # FWPT to coefficients
            coeffs = fwpt_jit(block, headroom_bits=1)
            data[p, :] = coeffs

        lut = WaveformCoeffLUT(
            waveform_type=waveform_type,
            phase_inc=phase_inc,
            P=P,
            data=data,
        )
        return lut


# Global cache for waveform coefficient LUTs
_WAVEFORM_COEFF_LUT_CACHE: Dict[Tuple[int, int], WaveformCoeffLUT] = {}


def get_waveform_coeff_lut(waveform_type: int, phase_inc: int) -> WaveformCoeffLUT:
    """
    Get or create precomputed coefficient LUT for waveform.

    Args:
        waveform_type: WaveformType constant
        phase_inc: Phase increment per sample

    Returns:
        WaveformCoeffLUT for this waveform/frequency combination
    """
    key = (waveform_type, phase_inc)
    if key not in _WAVEFORM_COEFF_LUT_CACHE:
        _WAVEFORM_COEFF_LUT_CACHE[key] = WaveformCoeffLUT.generate(
            waveform_type, phase_inc, P=256
        )
    return _WAVEFORM_COEFF_LUT_CACHE[key]


def clear_waveform_coeff_lut_cache() -> None:
    """Clear the waveform coefficient LUT cache."""
    global _WAVEFORM_COEFF_LUT_CACHE
    _WAVEFORM_COEFF_LUT_CACHE.clear()


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
# FM MODULATOR EXTRACTION (Phase 3)
# =============================================================================

def extract_modulator_signal(
    coeff_state: np.ndarray,
    method: int = 0
) -> np.ndarray:
    """
    Extract lane-rate modulator signal from coefficient state.

    This function downsamples the coefficient state to K=8 modulator samples
    for use as FM feedback in the next block.

    Per WSE-STD-210 Section 8.3:
    - z⁻¹ block delay: modulator extracted after operators, used next block
    - Lane-rate: one modulator sample per k-slot

    Args:
        coeff_state: [512] Q15 coefficients (64 leaves × 8 k-slots)
        method: Extraction method:
            0 = Average first 8 leaves (low-frequency content)
            1 = DC bin only (leaf 0)
            2 = Energy proxy (sum of squares)

    Returns:
        fm_mod_buffer: [8] Q15 modulator samples (one per k-slot)
    """
    fm_mod = np.zeros(VM_K, dtype=np.int16)

    if method == 0:
        # Average first 8 leaves (low-frequency content)
        # This captures the fundamental and first few harmonics
        for k in range(VM_K):
            acc = 0
            for leaf in range(8):
                acc += int(coeff_state[leaf * VM_K + k])
            # Divide by 8 with saturation
            fm_mod[k] = SAT16(acc >> 3)

    elif method == 1:
        # DC bin only (leaf 0) — simplest, fastest
        for k in range(VM_K):
            fm_mod[k] = coeff_state[k]

    elif method == 2:
        # Energy proxy (sum of squares, normalized)
        # Useful for envelope-following modulation
        for k in range(VM_K):
            energy = 0
            for leaf in range(NUM_LEAVES):
                val = int(coeff_state[leaf * VM_K + k])
                energy += (val * val) >> 15
            # Clamp to Q15 range
            fm_mod[k] = SAT16(min(32767, energy >> 6))

    return fm_mod


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'VM_K',
    'VM_K_RECONSTRUCT',
    'K_STRIDE',
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
    'WaveformCoeffLUT',
    'RatioBasisFrame',
    # Functions
    'generate_basis_frames',
    'get_basis_atlas',
    'get_basis_for_note',
    'mode_b_generate_coeffs',
    'mode_b_generate_coeffs_matrix',
    'mode_b_advance_phase',
    'compute_pwm_weights',
    'compute_bandlimit_H',
    'get_waveform_coeff_lut',
    'clear_waveform_coeff_lut_cache',
    # FM modulator extraction (Phase 3)
    'extract_modulator_signal',
    # FM6 ratio basis frames
    'get_basis_for_ratio',
    'generate_ratio_basis_frame',
    'clear_ratio_basis_cache',
]
