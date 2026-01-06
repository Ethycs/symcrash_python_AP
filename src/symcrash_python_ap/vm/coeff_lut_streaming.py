"""
WSE-STD-121 — Streaming CoeffLUT (True Subband Time-Series)
===========================================================

Implements the CORRECT coefficient-domain carrier architecture per WSE-STD-120.

Key insight from CCS-ARCHITECTURE-GAP.md:
    Wavelet coefficients are NOT spectral snapshots.
    They are TIME-DOMAIN SUBBAND SIGNALS.

Structure:
    CoeffLUT[waveform w][leaf l][time k][phase p] -> q15

Where:
    - w: waveform ID (sine=0, saw=1, square=2, etc.)
    - l: leaf/subband index [0..L-1], L=64 for 64-sample blocks
    - k: time samples WITHIN the subband [0..K-1], K=8 default
    - p: phase step [0..P-1], P=64 default (fractional lerp gives more)

The K dimension is critical: each leaf has K time samples that represent
a STREAMING subband signal, not a single coefficient value.

Offline Generation:
    1. Generate long bandlimited waveform (C samples, e.g. 8192)
    2. Run analysis filterbank (polyphase FWPT) to get subband streams
    3. Record subband signals as phase-indexed tables
    4. Apply normalization per WSE-STD-122

Runtime:
    1. Maintain phase accumulator (continuous)
    2. Compute (p, alpha) from phase
    3. Fill c_base[l][k] = lerp(LUT[w][l][k][p], LUT[w][l][k][p+1], alpha)
    4. Let CMMM operate on persistent c_state[l][k]
    5. Pass to streaming IDWPT for synthesis
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from ..wavelet.fast_lifting import fwpt_jit, idwpt_jit
from ..wavelet.sine_core import sine_q15

logger = logging.getLogger(__name__)

# Constants
BLOCK_SIZE = 64         # B = 64 audio samples per block
NUM_LEAVES = 64         # L = 64 leaves (full WPT at depth 6)
DEFAULT_K = 8           # K = 8 time samples per subband per block
DEFAULT_P = 1024        # P = 1024 phase steps per cycle (higher = fewer spurs)
CYCLE_LENGTH = 8192     # C = 8192 samples for one complete waveform cycle
Q15_ONE = 32767
TARGET_RMS_Q15 = 16384  # Target RMS (-6 dBFS)
SAMPLE_RATE = 48000

# =============================================================================
# WAVEFORM IDs
# =============================================================================
# Each waveform is band-limited via additive synthesis to avoid aliasing.
# Waveforms are generated at Q15 amplitude and normalized for consistent levels.

WAVE_SINE = 0       # Pure sine wave (fundamental only)
WAVE_SAW = 1        # Sawtooth - all harmonics: 1/h amplitude series
WAVE_SQUARE = 2     # Square wave - odd harmonics only: 1/h amplitude series
WAVE_TRIANGLE = 3   # Triangle wave - odd harmonics only: 1/h² amplitude series
WAVE_PULSE = 4      # Pulse wave - all harmonics with alternating signs

# Maximum harmonics to include (prevents aliasing at high notes)
MAX_HARMONICS = 32


def midi_to_hz(note: int, cents: int = 0) -> float:
    """Convert MIDI note to Hz."""
    semitones = note - 69 + (cents / 100.0)
    return 440.0 * (2.0 ** (semitones / 12.0))


@dataclass
class StreamingCoeffLUT:
    """
    Streaming coefficient lookup table with full K dimension.

    Structure: data[leaf L][time K][phase P] -> int16

    This represents SUBBAND TIME SERIES, not spectral snapshots.
    Each leaf has K time samples that evolve continuously.

    NOTE: This is a per-note table. The target_freq_hz bakes in the
    frequency so that IDWPT reconstruction produces the correct pitch.
    """
    wave_id: int
    name: str

    # Core data: [L leaves, K time samples, P phases]
    data: np.ndarray  # shape: [64, K, P], dtype: int16

    # Parameters
    num_leaves: int = NUM_LEAVES  # L = 64
    K: int = DEFAULT_K            # K = 8 time samples per subband
    P: int = DEFAULT_P            # P = 64 phase steps

    # Target frequency this table is generated for
    target_freq_hz: float = 261.63  # Default to C4


    def get_subband_samples(
        self,
        phase_32: int,
        output: np.ndarray,
    ) -> None:
        """Fill `output` with per-leaf subband samples for the given phase.

        This runs in the real-time path. It MUST be deterministic and fixed-point
        only, so the phase interpolation is done with integer math.

        Args:
            phase_32: 32-bit phase accumulator value
            output: np.ndarray[num_leaves, K] int16 (modified in place)
        """
        # Compute phase index based on actual P of this LUT
        # For P=256: shift by 24, mask with 0xFF
        # For P=64:  shift by 26, mask with 0x3F
        # General: shift by (32 - log2(P)), mask with (P-1)
        P = self.P
        if P == 256:
            shift = 24
            mask = 0xFF
        elif P == 64:
            shift = 26
            mask = 0x3F
        elif P == 512:
            shift = 23
            mask = 0x1FF
        elif P == 1024:
            shift = 22
            mask = 0x3FF
        else:
            # Fallback for arbitrary P (slower)
            import math
            shift = 32 - int(math.log2(P))
            mask = P - 1

        phase_idx = (phase_32 >> shift) & mask
        phase_idx_next = (phase_idx + 1) & mask

        # Fractional bits for interpolation
        frac_bits = shift
        frac = phase_32 & ((1 << frac_bits) - 1)

        # Fixed-point lerp:
        #   val = v0 + (v1 - v0) * frac / 2^frac_bits
        # with rounding to nearest.
        #
        # IMPORTANT: Use int64 for multiplication to avoid overflow.
        # diff can be up to ~350, frac can be up to 2^24-1 = 16777215
        # Product can be up to 350 * 16777215 = 5.8 billion > int32 max
        v0 = self.data[:, :, phase_idx].astype(np.int64)
        v1 = self.data[:, :, phase_idx_next].astype(np.int64)
        diff = v1 - v0
        half = np.int64(1) << (frac_bits - 1)
        frac_64 = np.int64(frac)
        val = v0 + ((diff * frac_64 + half) >> frac_bits)

        np.clip(val, -32768, 32767, out=val)
        output[:, :] = val.astype(np.int16)

    @staticmethod
    def generate_for_frequency(
        wave_id: int,
        name: str,
        target_freq_hz: float,
        generate_sample_fn,  # Callable[[int], int] - phase -> sample (Q15)
        K: int = DEFAULT_K,
        P: int = DEFAULT_P,
        normalize: bool = True,
        S: int = 8,  # Unused, kept for API compatibility
    ) -> 'StreamingCoeffLUT':
        """Generate StreamingCoeffLUT for a specific frequency.

        Q32 PHASE-ALIGNED LUT GENERATION:
        ==================================
        Uses the SAME phase model as runtime to eliminate phase mismatch.

        For each phase bin p and polyphase slot k:
            phase_p = p * 2^32 / P  (exact phase for bin p)
            block[n] = waveform(phase_p + (k + n) * phase_inc)
            coeff[:, k, p] = FWPT(block)

        This ensures perfect alignment with runtime's Q32 phase accumulator
        and coefficient-space interpolation between bins.

        No cycle_len or round() - phase is continuous and matches runtime exactly.
        """
        logger.info(
            f"Generating StreamingCoeffLUT for {name} at {target_freq_hz:.2f} Hz (K={K}, P={P})"
        )

        L = NUM_LEAVES

        # Phase increment per sample - SAME as runtime uses
        phase_inc = int((target_freq_hz / SAMPLE_RATE) * (1 << 32))

        logger.debug(f"  phase_inc={phase_inc} (Q32)")

        # Build LUT using Q32 phase model
        # No cycle_len, no round() - uses exact same phase math as runtime
        data = np.zeros((L, K, P), dtype=np.int16)

        # Compute shift for this P (must be power of 2)
        # P=64  -> shift=26, P=256 -> shift=24, P=512 -> shift=23, P=1024 -> shift=22
        import math
        shift = 32 - int(math.log2(P))

        for p in range(P):
            # Exact Q32 phase for bin p - matches runtime's phase_idx extraction
            # Runtime: phase_idx = (phase_32 >> shift) & (P-1)
            # So bin p corresponds to phase_32 in [p << shift, (p+1) << shift)
            #
            # BUILD AT BIN EDGES (not centers):
            # When runtime phase=0, it queries bin 0 with frac=0
            # LUT[0] should contain coefficients for phase=0
            # When runtime phase approaches bin 1, frac approaches 1
            # Linear interpolation between LUT[0] and LUT[1] gives smooth transition
            phase_p = p << shift  # Edge of phase bin p

            for k in range(K):
                # Generate 64-sample block starting at phase_p + k*phase_inc
                block = np.empty(BLOCK_SIZE, dtype=np.int16)
                phase = (phase_p + k * phase_inc) & 0xFFFFFFFF

                for n in range(BLOCK_SIZE):
                    sample = generate_sample_fn(phase)
                    if normalize:
                        sample = (sample * TARGET_RMS_Q15) // Q15_ONE
                    block[n] = np.clip(sample, -32768, 32767)
                    phase = (phase + phase_inc) & 0xFFFFFFFF

                # Run FWPT to get coefficients
                coeffs = fwpt_jit(block.tolist(), headroom_bits=1)
                data[:, k, p] = np.array(coeffs, dtype=np.int16)

        return StreamingCoeffLUT(
            wave_id=wave_id,
            name=name,
            data=data,
            num_leaves=L,
            K=K,
            P=P,
            target_freq_hz=target_freq_hz,
        )

    @staticmethod
    def generate_from_waveform(
        wave_id: int,
        name: str,
        generate_sample_fn,  # Callable[[int], int] - phase -> sample (Q15)
        K: int = DEFAULT_K,
        P: int = DEFAULT_P,
        normalize: bool = True,
    ) -> 'StreamingCoeffLUT':
        """Generate StreamingCoeffLUT at default frequency (C4)."""
        return StreamingCoeffLUT.generate_for_frequency(
            wave_id=wave_id,
            name=name,
            target_freq_hz=261.63,
            generate_sample_fn=generate_sample_fn,
            K=K,
            P=P,
            normalize=normalize,
        )

    @staticmethod
    def generate_sine(K: int = DEFAULT_K, P: int = DEFAULT_P) -> 'StreamingCoeffLUT':
        """Generate a sine StreamingCoeffLUT."""
        return StreamingCoeffLUT.generate_from_waveform(
            wave_id=0,
            name='sine',
            generate_sample_fn=sine_q15,
            K=K,
            P=P,
        )

    @staticmethod
    def generate_saw(K: int = DEFAULT_K, P: int = DEFAULT_P) -> 'StreamingCoeffLUT':
        """Generate a bandlimited saw StreamingCoeffLUT."""
        def saw_sample(phase: int) -> int:
            val = 0
            for h in range(1, 17):
                h_phase = (phase * h) & 0xFFFFFFFF
                val += sine_q15(h_phase) // h
            return int(np.clip(val // 2, -32768, 32767))

        return StreamingCoeffLUT.generate_from_waveform(
            wave_id=1,
            name='saw',
            generate_sample_fn=saw_sample,
            K=K,
            P=P,
        )

    @staticmethod
    def generate_square(K: int = DEFAULT_K, P: int = DEFAULT_P) -> 'StreamingCoeffLUT':
        """Generate a bandlimited square StreamingCoeffLUT."""
        def square_sample(phase: int) -> int:
            val = 0
            for h in [1, 3, 5, 7, 9, 11, 13, 15]:
                h_phase = (phase * h) & 0xFFFFFFFF
                val += sine_q15(h_phase) // h
            return int(np.clip(val, -32768, 32767))

        return StreamingCoeffLUT.generate_from_waveform(
            wave_id=2,
            name='square',
            generate_sample_fn=square_sample,
            K=K,
            P=P,
        )


# =============================================================================
# Persistent Subband State (coeff_state[l][k])
# =============================================================================

@dataclass
class SubbandState:
    """
    Persistent subband time-series state per voice.

    This is the core of WSE-STD-120 compliant synthesis:
    - L leaves (subbands)
    - K time samples per subband
    - Continuous, never reset on note events
    - Evolves via CMMM operators
    """
    # Subband state: [L leaves, K time samples]
    state: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_LEAVES, DEFAULT_K), dtype=np.int16)
    )

    # Ring buffer position for each leaf
    # Each leaf advances independently at its own rate
    k_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LEAVES, dtype=np.int32)
    )

    # Phase accumulator for carrier
    phase: int = 0

    # Phase increment (from MIDI note)
    phase_inc: int = 0

    def inject_carrier(
        self,
        lut: StreamingCoeffLUT,
        base_mix_q15: int = Q15_ONE,
    ) -> None:
        """
        Inject carrier coefficients into state.

        c_state = lerp(c_state, c_base, base_mix)

        Args:
            lut: StreamingCoeffLUT to read from
            base_mix_q15: Blend factor (0 = keep state, 32767 = replace with carrier)
        """
        # Get carrier coefficients for current phase
        c_base = np.zeros((NUM_LEAVES, lut.K), dtype=np.int16)
        lut.get_subband_samples(self.phase, c_base)

        if base_mix_q15 >= Q15_ONE:
            # Full replacement
            np.copyto(self.state, c_base)
        elif base_mix_q15 > 0:
            # Blend
            for l in range(NUM_LEAVES):
                for k in range(lut.K):
                    c_s = int(self.state[l, k])
                    c_b = int(c_base[l, k])
                    # c_out = c_state + (c_base - c_state) * mix / 32768
                    diff = c_b - c_s
                    blended = c_s + (diff * base_mix_q15) // 32768
                    self.state[l, k] = int(np.clip(blended, -32768, 32767))

    def advance_phase(self, samples: int = BLOCK_SIZE) -> None:
        """Advance phase accumulator by given number of samples."""
        self.phase = (self.phase + self.phase_inc * samples) & 0xFFFFFFFF


# =============================================================================
# Streaming IDWPT (Coefficient-Domain Synthesis)
# =============================================================================

def streaming_idwpt_block(
    state: SubbandState,
    output: np.ndarray,
    headroom_bits: int = 1,
) -> None:
    """
    TRUE POLYPHASE IDWPT synthesis from subband state.

    Per WSE-STD-120 section 3.1: The wavelet engine is a tree-structured
    polyphase synthesis filterbank.

    The K dimension represents K phase positions within the oscillator cycle.
    To produce smooth audio that evolves through these positions:

    1. Run IDWPT K times, once per k-slot
    2. Each IDWPT produces 64 samples at that phase position
    3. Interleave results: output[n] comes from IDWPT run (n % K)

    This creates a smooth transition through the K phase positions,
    eliminating block-boundary discontinuities.

    Args:
        state: SubbandState with current subband time-series [64, K]
        output: np.ndarray[BLOCK_SIZE] int16 output buffer
        headroom_bits: Headroom scaling
    """
    K = state.state.shape[1]

    # TRUE POLYPHASE: Run IDWPT K times, interleave results
    # This produces smooth audio that transitions through the K phase positions

    # Storage for K IDWPT outputs
    idwpt_outputs = []

    for k in range(K):
        # Extract coefficients for this k-slot
        coeffs_k = [int(state.state[l, k]) for l in range(NUM_LEAVES)]

        # Run IDWPT for this phase position
        audio_k = idwpt_jit(coeffs_k, headroom_bits=headroom_bits)
        idwpt_outputs.append(audio_k)

    # Interleave: output sample n uses IDWPT result from k = n % K
    # This creates smooth evolution through the K phase positions
    #
    # For K=8 and BLOCK_SIZE=64:
    # - Samples 0,8,16,24,32,40,48,56 come from k=0
    # - Samples 1,9,17,25,33,41,49,57 come from k=1
    # - etc.
    #
    # Each k contributes 64/K = 8 samples to the output

    for n in range(BLOCK_SIZE):
        k = n % K  # Which k-slot
        sample_idx = n  # Which sample from that IDWPT output
        output[n] = idwpt_outputs[k][sample_idx]


# =============================================================================
# Global LUT Library
# =============================================================================

_STREAMING_LUT_LIBRARY: Optional[Dict[int, StreamingCoeffLUT]] = None


def _init_streaming_library() -> Dict[int, StreamingCoeffLUT]:
    """Initialize the streaming CoeffLUT library."""
    logger.info("Initializing StreamingCoeffLUT library...")
    return {
        0: StreamingCoeffLUT.generate_sine(),
        1: StreamingCoeffLUT.generate_saw(),
        2: StreamingCoeffLUT.generate_square(),
    }


def get_streaming_lut(wave_id: int) -> StreamingCoeffLUT:
    """Get StreamingCoeffLUT by wave ID (lazy initialization)."""
    global _STREAMING_LUT_LIBRARY
    if _STREAMING_LUT_LIBRARY is None:
        _STREAMING_LUT_LIBRARY = _init_streaming_library()
    return _STREAMING_LUT_LIBRARY.get(wave_id, _STREAMING_LUT_LIBRARY[0])


# =============================================================================
# Band-Limited Waveform Generators (Q15)
# =============================================================================
# These functions generate band-limited waveforms using additive synthesis.
# Each takes a 32-bit phase (0 = 0°, 2^32 = 360°) and returns Q15 sample.
# Normalization ensures consistent output levels across waveforms.


def _saw_q15(phase: int, max_harmonics: int = MAX_HARMONICS) -> int:
    """
    Band-limited sawtooth wave via additive synthesis.

    Fourier series: saw(t) = (2/π) * Σ(n=1 to ∞) [(-1)^(n+1) * sin(2πnt) / n]

    Uses all harmonics with 1/h amplitude falloff.
    """
    val = 0
    for h in range(1, max_harmonics + 1):
        h_phase = (phase * h) & 0xFFFFFFFF
        sign = 1 if (h % 2 == 1) else -1  # Alternating sign for proper saw shape
        val += sign * sine_q15(h_phase) // h
    # Normalize: theoretical peak is ~1.28 of sine, scale to ~0.8 for headroom
    return int(np.clip(val * 2 // 3, -32768, 32767))


def _square_q15(phase: int, max_harmonics: int = MAX_HARMONICS) -> int:
    """
    Band-limited square wave via additive synthesis.

    Fourier series: square(t) = (4/π) * Σ(n=1,3,5,...) [sin(2πnt) / n]

    Uses odd harmonics only with 1/h amplitude falloff.
    """
    val = 0
    for h in range(1, max_harmonics + 1, 2):  # Odd harmonics: 1, 3, 5, 7...
        h_phase = (phase * h) & 0xFFFFFFFF
        val += sine_q15(h_phase) // h
    # Normalize: theoretical peak is ~1.27 of sine, scale appropriately
    return int(np.clip(val * 3 // 4, -32768, 32767))


def _triangle_q15(phase: int, max_harmonics: int = MAX_HARMONICS) -> int:
    """
    Band-limited triangle wave via additive synthesis.

    Fourier series: tri(t) = (8/π²) * Σ(n=1,3,5,...) [(-1)^((n-1)/2) * sin(2πnt) / n²]

    Uses odd harmonics with 1/h² amplitude falloff and alternating signs.
    Triangle has faster harmonic rolloff, so it's smoother than square.
    """
    # Accumulate in higher precision to avoid truncation errors
    val = 0
    sign = 1
    for h in range(1, max_harmonics + 1, 2):  # Odd harmonics: 1, 3, 5, 7...
        h_phase = (phase * h) & 0xFFFFFFFF
        # Scale each harmonic: sine_q15 / h² but keep precision
        val += sign * sine_q15(h_phase) // (h * h)
        sign = -sign  # Alternate sign for proper triangle shape

    # Raw peak is ~39918 (all odd harmonics add at 90 degrees)
    # Scale down by 32767/39918 ≈ 0.821 ≈ 13/16 = 0.8125
    return int(np.clip(val * 13 // 16, -32767, 32767))


def _pulse_q15(phase: int, max_harmonics: int = MAX_HARMONICS, duty: float = 0.25) -> int:
    """
    Band-limited pulse wave via additive synthesis.

    A pulse wave is like a square wave but with variable duty cycle.
    Duty=0.5 is a square wave, duty=0.25 is a 25% pulse (common in subtractive synths).

    Fourier series: pulse(t) = d + (2/π) * Σ(n=1 to ∞) [sin(πnd)/n * cos(2πnt - πnd)]

    This implementation uses a simpler approach: sum of harmonics with amplitude
    modulated by sin(π*n*duty) / n.
    """
    val = 0
    duty_factor = duty * np.pi
    for h in range(1, max_harmonics + 1):
        # Amplitude modulated by sin(π*n*duty) - this gives the pulse spectrum
        amp_mod = np.sin(h * duty_factor)
        if abs(amp_mod) > 0.01:  # Skip near-zero harmonics
            h_phase = (phase * h) & 0xFFFFFFFF
            val += int(amp_mod * sine_q15(h_phase) / h)
    # Normalize similar to saw
    return int(np.clip(val * 2 // 3, -32768, 32767))


def get_waveform_generator(wave_id: int):
    """
    Get the appropriate waveform generator function for a wave ID.

    Args:
        wave_id: Waveform ID (WAVE_SINE, WAVE_SAW, etc.)

    Returns:
        Tuple of (generator_function, name_string)
    """
    if wave_id == WAVE_SINE:
        return sine_q15, "sine"
    elif wave_id == WAVE_SAW:
        return _saw_q15, "saw"
    elif wave_id == WAVE_SQUARE:
        return _square_q15, "square"
    elif wave_id == WAVE_TRIANGLE:
        return _triangle_q15, "triangle"
    elif wave_id == WAVE_PULSE:
        return _pulse_q15, "pulse"
    else:
        # Default to sine for unknown wave IDs
        logger.warning(f"Unknown wave_id {wave_id}, defaulting to sine")
        return sine_q15, "sine"


# =============================================================================
# Per-Note LUT Cache (for frequency-specific tables)
# =============================================================================

_NOTE_LUT_CACHE: Dict[Tuple[int, int], StreamingCoeffLUT] = {}


def get_lut_for_note(wave_id: int, midi_note: int) -> StreamingCoeffLUT:
    """
    Get StreamingCoeffLUT for a specific MIDI note.

    Tables are generated and cached on first access. Each note gets its own
    table because the frequency is baked into the FWPT coefficients.

    Args:
        wave_id: Waveform ID (WAVE_SINE=0, WAVE_SAW=1, WAVE_SQUARE=2, WAVE_TRIANGLE=3, WAVE_PULSE=4)
        midi_note: MIDI note number (0-127)

    Returns:
        StreamingCoeffLUT for this note/waveform combination
    """
    cache_key = (wave_id, midi_note)
    if cache_key not in _NOTE_LUT_CACHE:
        freq_hz = midi_to_hz(midi_note)
        logger.debug(f"Generating LUT for wave={wave_id}, note={midi_note} ({freq_hz:.1f} Hz)")

        # Get waveform generator from centralized function
        gen_fn, name = get_waveform_generator(wave_id)

        _NOTE_LUT_CACHE[cache_key] = StreamingCoeffLUT.generate_for_frequency(
            wave_id=wave_id,
            name=f"{name}_{midi_note}",
            target_freq_hz=freq_hz,
            generate_sample_fn=gen_fn,
        )

    return _NOTE_LUT_CACHE[cache_key]


# =============================================================================
# VM-Compatible Oscillator State
# =============================================================================

@dataclass
class CCSState:
    """
    VM-compatible coefficient oscillator state.

    This replaces the old CoeffOscState with the new streaming architecture.
    Simpler: just phase accumulator + target for slew.
    """
    # Phase accumulator (32-bit)
    phase: int = 0

    # Current phase increment (from MIDI note)
    phase_inc: int = 0

    # Target phase increment (for pitch slew)
    target_phase_inc: int = 0

    # Phase slew rate per sample
    phase_slew_rate: int = 0

    # Current MIDI note (for LUT cache lookup)
    note: int = 60

    # Current waveform ID
    wave_id: int = 0

    def reset(self, note: int = 60, phase_inc: int = 0, wave_id: int = 0) -> None:
        """Reset oscillator state for new note."""
        self.phase = 0
        self.note = note
        self.wave_id = wave_id
        self.phase_inc = phase_inc
        self.target_phase_inc = phase_inc
        self.phase_slew_rate = 0

    def set_pitch(self, phase_inc: int, slew_blocks: int = 8) -> None:
        """Set target pitch with optional slew."""
        if slew_blocks <= 0:
            self.phase_inc = phase_inc
            self.target_phase_inc = phase_inc
            self.phase_slew_rate = 0
        else:
            self.target_phase_inc = phase_inc
            diff = phase_inc - self.phase_inc
            self.phase_slew_rate = diff // (slew_blocks * BLOCK_SIZE)

    def advance_block(self) -> None:
        """Advance phase by one block, applying slew if active."""
        # Apply slew
        if self.phase_slew_rate != 0:
            if self.phase_slew_rate > 0:
                self.phase_inc = min(self.target_phase_inc,
                                     self.phase_inc + self.phase_slew_rate * BLOCK_SIZE)
            else:
                self.phase_inc = max(self.target_phase_inc,
                                     self.phase_inc + self.phase_slew_rate * BLOCK_SIZE)
            if self.phase_inc == self.target_phase_inc:
                self.phase_slew_rate = 0

        # Advance phase
        self.phase = (self.phase + self.phase_inc * BLOCK_SIZE) & 0xFFFFFFFF


def ccs_generate_coeffs(
    ccs_state: CCSState,
    output_coeffs: np.ndarray,
    base_mix_q15: int = Q15_ONE,
) -> None:
    """
    Generate 64 leaf coefficients from CCS state.

    This is the main interface for the VM to get carrier coefficients.
    Uses per-note LUT cache for correct frequency.

    Args:
        ccs_state: CCS oscillator state
        output_coeffs: np.ndarray[64] int16 output buffer (modified in place)
        base_mix_q15: Blend factor (not used here, handled by caller)
    """
    # Get LUT for this note
    lut = get_lut_for_note(ccs_state.wave_id, ccs_state.note)

    # Get subband samples for current phase (just k=0 for stepping stone)
    subband_output = np.zeros((NUM_LEAVES, lut.K), dtype=np.int16)
    lut.get_subband_samples(ccs_state.phase, subband_output)

    # Extract k=0 samples as the 64 coefficients
    for l in range(NUM_LEAVES):
        output_coeffs[l] = subband_output[l, 0]


# VM-side K dimension - MUST match LUT DEFAULT_K for correct polyphase synthesis
# coeff_state has 64 leaves × K time slots = 64×8 = 512 elements
VM_K = DEFAULT_K  # K=8


def ccs_generate_coeffs_polyphase(
    ccs_state: CCSState,
    output_coeffs: np.ndarray,
) -> None:
    """
    Generate polyphase coefficients: 64 leaves × K time slots = 512 values.

    TRUE POLYPHASE INJECTION per WSE-STD-120:
    - Each k-slot represents a polyphase sample offset within the block
    - coeff_state[leaf * K + k] = LUT[wave][leaf][k][phase_idx]

    VM_K equals LUT K - direct 1:1 mapping, no downsampling.

    Args:
        ccs_state: CCS oscillator state
        output_coeffs: np.ndarray[512] int16 output buffer (modified in place)
                       Layout: [leaf0_k0..k7, leaf1_k0..k7, ..., leaf63_k0..k7]
    """
    # Get LUT for this note
    lut = get_lut_for_note(ccs_state.wave_id, ccs_state.note)

    # Get all K subband samples for current phase
    subband_output = np.zeros((NUM_LEAVES, lut.K), dtype=np.int16)
    lut.get_subband_samples(ccs_state.phase, subband_output)

    # Direct 1:1 mapping - VM_K == LUT K == 8
    # Inject into 512-element state: [64 leaves × 8 k-slots]
    for l in range(NUM_LEAVES):
        for k in range(VM_K):
            output_coeffs[l * VM_K + k] = subband_output[l, k]


__all__ = [
    # Core classes
    'StreamingCoeffLUT',
    'SubbandState',
    'CCSState',
    # Waveform constants
    'WAVE_SINE',
    'WAVE_SAW',
    'WAVE_SQUARE',
    'WAVE_TRIANGLE',
    'WAVE_PULSE',
    'MAX_HARMONICS',
    # LUT access
    'get_streaming_lut',
    'get_lut_for_note',
    'get_waveform_generator',
    # Synthesis functions
    'streaming_idwpt_block',
    'ccs_generate_coeffs',
    'ccs_generate_coeffs_polyphase',
    # Constants
    'VM_K',
    'DEFAULT_K',
    'DEFAULT_P',
    'BLOCK_SIZE',
    'NUM_LEAVES',
    'midi_to_hz',
]
