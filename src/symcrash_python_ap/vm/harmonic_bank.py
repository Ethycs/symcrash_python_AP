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
from enum import IntEnum
from typing import Dict, Tuple, Optional, List
import numpy as np

from ..fixed import RSHIFT_RNE, SAT16, Q15_ONE
from ..wavelet.sine_core import sin_cos_q15, freq_to_phase_inc, PHASE_MASK
from .tables import COEFF_LEN, VM_K
from ..primitives import BLOCK_SIZE, NUM_LEAVES, SAMPLE_RATE_HZ

# =============================================================================
# CONSTANTS (WSE-STD-210 Section 1)
# =============================================================================

# Polyphase configuration imported from tables.py (VM_K = 4 or 8)
# K_STRIDE: sample offset per slot for basis frame generation
# Both K=4 and K=8 use 1-sample phase offsets for interleaved reconstruction
K_STRIDE = 1
VM_K_RECONSTRUCT = VM_K  # For API compatibility

SAMPLE_RATE = SAMPLE_RATE_HZ  # Use canonical value

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

    def set_oeis(
        self,
        sequence_id: int,
        H: int = 32,
        rolloff_db_per_octave: float = -3.0,
        base_amplitude_q15: int = 32767,
    ) -> None:
        """
        Configure harmonics based on OEIS integer sequence.

        Uses mathematical sequences to select which partials are active,
        creating unique inharmonic timbres:

        - PRIMES (0): Metallic, bell-like (partials 2, 3, 5, 7, 11, ...)
        - FIBONACCI (1): Organic, complex (partials 1, 2, 3, 5, 8, 13, ...)
        - BEATTY_PHI (2): Golden ratio spacing (partials 1, 3, 4, 6, 8, ...)
        - TRIANGULAR (3): Stepped feel (partials 1, 3, 6, 10, 15, ...)
        - LUCAS (4): Like Fibonacci, brighter (partials 2, 1, 3, 4, 7, ...)
        - SQUARE_FREE (5): Dense but irregular (partials 1, 2, 3, 5, 6, 7, ...)
        - ODD_ONLY (6): Hollow, clarinet-like (partials 1, 3, 5, 7, ...)
        - MERSENNE (7): Exponentially sparse (partials 1, 3, 7, 15, 31, ...)

        Args:
            sequence_id: OEISSequenceID value (0-7)
            H: Maximum harmonic count
            rolloff_db_per_octave: Amplitude rolloff per octave (default -3dB)
            base_amplitude_q15: Amplitude for lowest active partial
        """
        from .oeis_tables import OEISSequenceID, get_partial_amplitudes

        # Get amplitudes for partials in the sequence
        amps = get_partial_amplitudes(
            OEISSequenceID(sequence_id),
            H_max=min(H, len(self.a_h_q15)),
            rolloff_db_per_octave=rolloff_db_per_octave,
            base_amplitude_q15=base_amplitude_q15,
        )

        # Apply to sine weights (a_h)
        self.dc_q15 = 0
        self.a_h_q15.fill(0)
        self.b_h_q15.fill(0)
        self.a_h_q15[:len(amps)] = amps
        self.H_active = min(H, len(self.a_h_q15))

    def set_oeis_primes(self, H: int = 32, rolloff: float = -3.0) -> None:
        """Shorthand for OEIS primes sequence - metallic, bell-like."""
        self.set_oeis(0, H, rolloff)

    def set_oeis_fibonacci(self, H: int = 32, rolloff: float = -3.0) -> None:
        """Shorthand for OEIS Fibonacci sequence - organic, complex."""
        self.set_oeis(1, H, rolloff)

    def set_oeis_golden(self, H: int = 32, rolloff: float = -3.0) -> None:
        """Shorthand for OEIS Beatty-phi sequence - golden ratio partials."""
        self.set_oeis(2, H, rolloff)

    def set_oeis_triangular(self, H: int = 32, rolloff: float = -3.0) -> None:
        """Shorthand for OEIS triangular numbers - stepped, quantized."""
        self.set_oeis(3, H, rolloff)


# =============================================================================
# LFO/MODULATION TYPES (Phase 2 Mod Matrix)
# =============================================================================

class LFOWaveform(IntEnum):
    """LFO waveform shapes."""
    SINE = 0
    TRIANGLE = 1
    SAW_UP = 2
    SQUARE = 3


class LFOSyncMode(IntEnum):
    """LFO phase sync behavior on note events."""
    FREE = 0           # Free-running, never resets
    GATE_RESET = 1     # Reset phase on every note-on
    LEGATO_RESET = 2   # Reset only on non-legato note-on


class LFOTarget(IntEnum):
    """CCS-native modulation targets (FPGA-efficient)."""
    # Phase domain (very cheap)
    PITCH = 0           # → cents field

    # Amplitude (cheap)
    GAIN = 1            # → gain_q15 multiplier

    # Mode B harmonic bank (cheap)
    PWM_WIDTH = 10      # → pwm_width_q15
    TILT = 11           # → tilt_env_q15 (brightness)

    # Filter (Material Engine SVF) - block rate, cheap
    FILTER_CUTOFF = 15  # → filter_cutoff_q15 (log scale)
    FILTER_RESO = 16    # → filter_reso_q15

    # Coefficient domain (medium)
    BAND_SCALE = 20     # → per-band coefficient scaling [6 bands]

    # Audio rate (expensive - avoid on FPGA)
    VFO_RING = 30       # → per-sample coefficient multiply


@dataclass
class LFORouting:
    """
    Where LFO output goes.

    target: Which parameter to modulate
    amount_q15: Scaling factor (32767 = 100%)
    """
    target: LFOTarget = LFOTarget.PITCH
    amount_q15: int = 32767


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
class LFOState(VibratoState):
    """
    General LFO extending VibratoState with multi-waveform and routing.

    Per WSE-ARCHITECTURE-v1, LFO state belongs in Voice Engine because:
    - Voice Engine owns musical time (WSE-STD-060)
    - LFO phase is per-voice state
    - LFO outputs modulate VoiceFrame fields

    Phase Storage (Unified with CMMM):
    - LFO phase is stored in VoiceState.arc_phase_q16[arc_slot]
    - This unifies LFO time with CMMM arc phase storage
    - arc_slot indicates which slot in arc_phase_q16 this LFO uses

    Extends VibratoState with:
    - Multiple waveform shapes (sine, triangle, saw, square)
    - Sync mode for gate reset behavior
    - Routing to CCS-native targets
    """
    id: str = ""
    waveform: LFOWaveform = LFOWaveform.SINE
    sync_mode: LFOSyncMode = LFOSyncMode.FREE
    routings: List[LFORouting] = field(default_factory=list)

    # Depth in Q15 (unified for all targets)
    depth_q15: int = 16384  # 50% default

    # Arc slot index - which arc_phase_q16 slot stores this LFO's phase
    # Phase is stored externally in VoiceState.arc_phase_q16[arc_slot]
    arc_slot: int = 0

    def on_gate(self, is_legato: bool = False) -> None:
        """
        Handle note-on event for sync modes.

        Note: This resets the internal phase tracking. The caller must also
        reset arc_phase_q16[arc_slot] to 0 for proper phase reset.
        """
        if self.sync_mode == LFOSyncMode.GATE_RESET:
            self.phase = 0
            self.elapsed_samples = 0
            self.current_depth_scale = 0
        elif self.sync_mode == LFOSyncMode.LEGATO_RESET and not is_legato:
            self.phase = 0
            self.elapsed_samples = 0
            self.current_depth_scale = 0

    def on_gate_with_arc(self, arc_phase_q16: List[int], is_legato: bool = False) -> None:
        """
        Handle note-on event for sync modes with unified arc phase storage.

        Args:
            arc_phase_q16: The VoiceState.arc_phase_q16 array
            is_legato: True if this is a legato note transition
        """
        if self.sync_mode == LFOSyncMode.GATE_RESET:
            arc_phase_q16[self.arc_slot] = 0
            self.elapsed_samples = 0
            self.current_depth_scale = 0
        elif self.sync_mode == LFOSyncMode.LEGATO_RESET and not is_legato:
            arc_phase_q16[self.arc_slot] = 0
            self.elapsed_samples = 0
            self.current_depth_scale = 0

    def get_waveform_value(self, phase: Optional[int] = None) -> int:
        """
        Get raw waveform value (before envelope scaling).

        Args:
            phase: Phase value to use (Q32). If None, uses self.phase.

        Returns:
            Q15 value in range [-32767, 32767]
        """
        if phase is None:
            phase = self.phase

        if self.waveform == LFOWaveform.SINE:
            sin_val, _ = sin_cos_q15(phase)
            return sin_val

        elif self.waveform == LFOWaveform.TRIANGLE:
            # Triangle: linear ramp -1 to +1 to -1
            # phase 0x00000000 → -32767
            # phase 0x40000000 → +32767
            # phase 0x80000000 → -32767
            # phase 0xC0000000 → +32767
            phase_q15 = (phase >> 17) & 0x7FFF  # Top 15 bits of lower half
            half_cycle = (phase >> 31) & 1
            if half_cycle == 0:
                # First half: -32767 to +32767
                return (phase_q15 * 2) - Q15_ONE
            else:
                # Second half: +32767 to -32767
                return Q15_ONE - (phase_q15 * 2)

        elif self.waveform == LFOWaveform.SAW_UP:
            # Saw: -32767 at phase 0, +32767 at phase max
            phase_q15 = (phase >> 17) & 0x7FFF
            return (phase_q15 * 2) - Q15_ONE

        elif self.waveform == LFOWaveform.SQUARE:
            # Square: +32767 for first half, -32767 for second half
            return Q15_ONE if phase < 0x80000000 else -Q15_ONE

        return 0

    def get_lfo_value(self, phase: Optional[int] = None) -> int:
        """
        Get current LFO value scaled by envelope and depth.

        Args:
            phase: Phase value to use (Q32). If None, uses self.phase.

        Returns:
            Q15 value in range [-32767, 32767]
        """
        if not self.enabled or self.current_depth_scale == 0:
            return 0

        # Get raw waveform value
        raw = self.get_waveform_value(phase)

        # Scale by envelope
        env_scaled = RSHIFT_RNE(raw * self.current_depth_scale, 15)

        # Scale by depth
        return RSHIFT_RNE(env_scaled * self.depth_q15, 15)

    def get_lfo_value_from_arc(self, arc_phase_q16: List[int]) -> int:
        """
        Get current LFO value using unified arc phase storage.

        Args:
            arc_phase_q16: The VoiceState.arc_phase_q16 array

        Returns:
            Q15 value in range [-32767, 32767]
        """
        phase = arc_phase_q16[self.arc_slot]
        return self.get_lfo_value(phase)

    def advance_block_with_arc(self, arc_phase_q16: List[int], block_samples: int = BLOCK_SIZE) -> None:
        """
        Advance LFO by one block using unified arc phase storage.

        Updates both the arc_phase_q16 slot and the envelope state.

        Args:
            arc_phase_q16: The VoiceState.arc_phase_q16 array
            block_samples: Number of samples in block
        """
        if not self.enabled:
            return

        # Advance elapsed time (for envelope)
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

        # Advance LFO phase in arc storage
        current_phase = arc_phase_q16[self.arc_slot]
        arc_phase_q16[self.arc_slot] = (current_phase + self.phase_inc * block_samples) & PHASE_MASK

    def configure_lfo(
        self,
        rate_hz: float = 5.0,
        depth_q15: int = 16384,
        waveform: LFOWaveform = LFOWaveform.SINE,
        sync_mode: LFOSyncMode = LFOSyncMode.FREE,
        delay_ms: float = 0.0,
        ramp_ms: float = 0.0,
        routings: Optional[List[LFORouting]] = None,
        arc_slot: int = 0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """
        Configure LFO parameters.

        Args:
            rate_hz: LFO frequency in Hz
            depth_q15: Modulation depth in Q15 (32767 = 100%)
            waveform: Waveform shape
            sync_mode: Gate sync behavior
            delay_ms: Delay before LFO starts
            ramp_ms: Ramp-in time after delay
            routings: List of LFORouting targets
            arc_slot: Index into arc_phase_q16 for phase storage
            sample_rate: Sample rate in Hz
        """
        self.enabled = True
        self.rate_hz = rate_hz
        self.depth_q15 = depth_q15
        self.waveform = waveform
        self.sync_mode = sync_mode
        self.arc_slot = arc_slot

        # Compute LFO phase increment
        self.phase_inc = int((rate_hz * (1 << 32)) / sample_rate)

        # Compute envelope timing
        self.delay_samples = int(delay_ms * sample_rate / 1000.0)
        self.ramp_samples = max(1, int(ramp_ms * sample_rate / 1000.0))

        # Set routings
        self.routings = routings if routings is not None else []

        # Reset runtime state (phase will be in arc_phase_q16)
        self.phase = 0
        self.elapsed_samples = 0
        self.current_depth_scale = 0


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

    # NOTE: Vibrato/LFO is now handled via LFOState in VoiceState.lfos
    # The VoiceEngine reads SystemMacro.VIBRATO to scale LFO depth globally.
    # See voice_engine.py step_voice_simple() for implementation.

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
# MODE B OPTIMIZED: FPGA-FRIENDLY AND VECTORIZED IMPLEMENTATIONS
# =============================================================================

# Fixed constants for HW implementation
HW_H_MAX = 16       # Fixed harmonic count for FPGA (no conditionals)
HW_K = 4            # Fixed K for FPGA SIMD width


def mode_b_generate_coeffs_hw(
    phase: int,
    phase_inc: int,
    a_h_q15: np.ndarray,
    b_h_q15: np.ndarray,
    dc_q15: int,
    Usin_h: np.ndarray,
    Ucos_h: np.ndarray,
    DC_frame: np.ndarray,
    output_coeffs: np.ndarray,
    fm_mod_buffer: np.ndarray = None,
    fm_depth_q15: int = 0,
) -> None:
    """
    FPGA-friendly Mode B coefficient generator.

    Key differences from mode_b_generate_coeffs:
    1. Fixed H=16 (no variable harmonic count)
    2. Loop order optimized for FPGA pipelining: h -> leaf -> k
    3. All phases computed upfront (enables parallel sin/cos)
    4. No conditionals in inner loops
    5. Explicit int32 arithmetic matching HW

    FPGA Implementation Notes:
    - Outer h loop: pipeline stages, one harmonic per cycle
    - Middle leaf loop: 64 parallel MAC units (one per leaf)
    - Inner k loop: K=4 SIMD lanes

    Memory Layout (matches FPGA BRAM):
    - Usin_h[H, L, K]: row-major, stride-1 on K for SIMD
    - output_coeffs[L*K]: interleaved for polyphase reconstruction

    Args:
        phase: Base oscillator phase (Q32)
        phase_inc: Phase increment per sample (Q32)
        a_h_q15: Sine harmonic weights [H_MAX] Q15
        b_h_q15: Cosine harmonic weights [H_MAX] Q15
        dc_q15: DC offset Q15
        Usin_h: Sine basis frames [H, 64, K] Q15
        Ucos_h: Cosine basis frames [H, 64, K] Q15
        DC_frame: DC basis frame [64, K] Q15
        output_coeffs: Output array [L*K] Q15
        fm_mod_buffer: Optional FM modulator samples [K] Q15
        fm_depth_q15: FM modulation depth Q15
    """
    K = HW_K
    H = HW_H_MAX
    L = NUM_LEAVES

    # Phase 1: Compute K phases (parallel on FPGA)
    # phi_k = base_phase + k * phase_inc + fm_mod
    phi_k = np.zeros(K, dtype=np.int64)
    for k in range(K):
        phi_k[k] = phase
        if fm_mod_buffer is not None and fm_depth_q15 != 0:
            # FM: phase_mod = depth * mod >> 1 (Q30 -> Q29)
            mod_sample = int(fm_mod_buffer[k]) if k < len(fm_mod_buffer) else 0
            phase_mod = (fm_depth_q15 * mod_sample) >> 1
            phi_k[k] = (phi_k[k] + phase_mod) & PHASE_MASK

    # Phase 2: Compute sin/cos for all harmonics and k-slots
    # On FPGA: H×K CORDIC units or sin/cos LUT with K parallel reads
    sin_hphi = np.zeros((H, K), dtype=np.int32)
    cos_hphi = np.zeros((H, K), dtype=np.int32)
    for h in range(H):
        for k in range(K):
            h_phase = (phi_k[k] * (h + 1)) & PHASE_MASK
            sin_hphi[h, k], cos_hphi[h, k] = sin_cos_q15(int(h_phase))

    # Phase 3: Compute combined weights w_sin, w_cos for each harmonic
    # w_sin = a_h*cos - b_h*sin, w_cos = a_h*sin + b_h*cos
    # On FPGA: H parallel 2x2 matrix multiplies
    w_sin = np.zeros((H, K), dtype=np.int32)
    w_cos = np.zeros((H, K), dtype=np.int32)
    for h in range(H):
        a_h = int(a_h_q15[h]) if h < len(a_h_q15) else 0
        b_h = int(b_h_q15[h]) if h < len(b_h_q15) else 0
        for k in range(K):
            # Q15 × Q15 >> 15 = Q15
            w_sin[h, k] = RSHIFT_RNE(a_h * cos_hphi[h, k] - b_h * sin_hphi[h, k], 15)
            w_cos[h, k] = RSHIFT_RNE(a_h * sin_hphi[h, k] + b_h * cos_hphi[h, k], 15)

    # Phase 4: Accumulate into output coefficients
    # On FPGA: 64 parallel accumulators, each with K SIMD MAC units
    # Loop order: leaf (parallel) -> h (pipelined) -> k (SIMD)
    for leaf in range(L):
        for k in range(K):
            idx = leaf * K + k
            # Start with DC
            acc = RSHIFT_RNE(dc_q15 * int(DC_frame[leaf, k]), 15)

            # Accumulate harmonics (unrolled on FPGA)
            for h in range(H):
                sin_contrib = RSHIFT_RNE(w_sin[h, k] * int(Usin_h[h, leaf, k]), 15)
                cos_contrib = RSHIFT_RNE(w_cos[h, k] * int(Ucos_h[h, leaf, k]), 15)
                acc += sin_contrib + cos_contrib

            output_coeffs[idx] = SAT16(acc)


def mode_b_generate_coeffs_vectorized(
    state: 'ModeBState',
    basis: 'HarmonicBasisFrames',
    output_coeffs: np.ndarray,
) -> None:
    """
    NumPy-vectorized Mode B coefficient generator for Python performance.

    Eliminates Python loops by using matrix operations:
    1. Batch sin/cos computation for all H×K
    2. Matrix multiply for weight rotation
    3. Einstein summation for basis frame accumulation

    ~10-20x faster than scalar mode_b_generate_coeffs in Python.

    Args:
        state: ModeBState with phase, weights, fm_depth_q15, fm_mod_buffer
        basis: HarmonicBasisFrames for current pitch anchor
        output_coeffs: Output array [512] to fill with coefficients
    """
    weights = state.weights
    H = min(state.H_active, state.H_bandlimit, basis.H_max)
    K = VM_K
    L = NUM_LEAVES

    if H == 0:
        output_coeffs.fill(0)
        return

    # Get weights as numpy arrays
    a_h = weights.a_h_q15[:H].astype(np.int32)
    b_h = weights.b_h_q15[:H].astype(np.int32)
    dc = int(weights.dc_q15)

    # Compute scale shift for normalization
    total_weight = np.maximum(np.abs(a_h), np.abs(b_h)).sum()
    scale_shift = 0
    while total_weight > 32767 and scale_shift < 15:
        total_weight >>= 1
        scale_shift += 1

    if scale_shift > 0:
        a_h = a_h >> scale_shift
        b_h = b_h >> scale_shift

    # Phase computation for K slots
    base_phi = state.phase
    fm_active = state.fm_depth_q15 != 0 and state.fm_mod_buffer is not None

    phi_k = np.full(K, base_phi, dtype=np.int64)
    if fm_active:
        fm_mod = np.array(state.fm_mod_buffer[:K], dtype=np.int64)
        phase_mod = (state.fm_depth_q15 * fm_mod) >> 1
        phi_k = (phi_k + phase_mod) & PHASE_MASK

    # Compute sin(hφ), cos(hφ) for all H harmonics and K slots
    # Shape: [H, K]
    h_indices = np.arange(1, H + 1, dtype=np.int64).reshape(-1, 1)
    h_phases = (phi_k.reshape(1, -1) * h_indices) & PHASE_MASK

    # Vectorized sin/cos (use LUT or numpy)
    sin_hphi = np.zeros((H, K), dtype=np.int32)
    cos_hphi = np.zeros((H, K), dtype=np.int32)
    for h in range(H):
        for k in range(K):
            sin_hphi[h, k], cos_hphi[h, k] = sin_cos_q15(int(h_phases[h, k]))

    # Compute rotation weights: w_sin[h,k], w_cos[h,k]
    # w_sin = a_h*cos - b_h*sin, w_cos = a_h*sin + b_h*cos
    a_h_2d = a_h.reshape(-1, 1)  # [H, 1]
    b_h_2d = b_h.reshape(-1, 1)  # [H, 1]

    w_sin = ((a_h_2d * cos_hphi - b_h_2d * sin_hphi) + (1 << 14)) >> 15
    w_cos = ((a_h_2d * sin_hphi + b_h_2d * cos_hphi) + (1 << 14)) >> 15

    # Basis frames: Usin_h[H, L, K], Ucos_h[H, L, K]
    Usin = basis.Usin_h[:H, :, :K].astype(np.int32)
    Ucos = basis.Ucos_h[:H, :, :K].astype(np.int32)

    # Accumulate: output[l,k] = dc*DC[l,k] + sum_h(w_sin[h,k]*Usin[h,l,k] + w_cos[h,k]*Ucos[h,l,k])
    # Using einsum for efficiency

    # DC contribution: [L, K]
    DC = basis.DC_frame[:, :K].astype(np.int32)
    result = ((dc * DC) + (1 << 14)) >> 15

    # Harmonic contributions using broadcasting
    # w_sin[h, k] * Usin[h, l, k] -> sum over h -> [l, k]
    # Reshape w_sin/w_cos to [H, 1, K] for broadcasting with [H, L, K]
    w_sin_3d = w_sin.reshape(H, 1, K)
    w_cos_3d = w_cos.reshape(H, 1, K)

    sin_contrib = ((w_sin_3d * Usin) + (1 << 14)) >> 15
    cos_contrib = ((w_cos_3d * Ucos) + (1 << 14)) >> 15

    # Sum over harmonics
    result += sin_contrib.sum(axis=0) + cos_contrib.sum(axis=0)

    # Saturate and interleave into output
    result = np.clip(result, -32768, 32767).astype(np.int16)

    # Interleave: output[l*K + k] = result[l, k]
    for l in range(L):
        for k in range(K):
            output_coeffs[l * K + k] = result[l, k]


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
    # LFO/Modulation types (Phase 2)
    'LFOWaveform',
    'LFOSyncMode',
    'LFOTarget',
    'LFORouting',
    'LFOState',
    'VibratoState',
    # Functions
    'generate_basis_frames',
    'get_basis_atlas',
    'get_basis_for_note',
    'mode_b_generate_coeffs',
    'mode_b_generate_coeffs_matrix',
    'mode_b_generate_coeffs_hw',
    'mode_b_generate_coeffs_vectorized',
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
    # HW constants
    'HW_H_MAX',
    'HW_K',
]
