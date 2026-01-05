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
DEFAULT_P = 64          # P = 64 phase steps per cycle
CYCLE_LENGTH = 8192     # C = 8192 samples for one complete waveform cycle
Q15_ONE = 32767
TARGET_RMS_Q15 = 16384  # Target RMS (-6 dBFS)
SAMPLE_RATE = 48000


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
        """
        Get K time samples per leaf for a given 32-bit phase.

        Uses linear interpolation between phase steps.

        Args:
            phase_32: 32-bit phase accumulator value
            output: np.ndarray[L, K] int16 output buffer (modified in place)
        """
        # Convert 32-bit phase to phase index + fraction
        # Top 6 bits give phase index (0-63)
        # Remaining bits give interpolation fraction
        phase_idx = (phase_32 >> 26) & 0x3F  # 0..63
        phase_idx_next = (phase_idx + 1) & 0x3F

        # Interpolation fraction from lower bits
        # 26-bit fraction scaled to [0, 1)
        frac_26bit = phase_32 & 0x3FFFFFF
        alpha = frac_26bit / float(1 << 26)  # [0.0, 1.0)

        # Linear interpolation for each leaf and time sample
        for l in range(self.num_leaves):
            for k in range(self.K):
                v0 = float(self.data[l, k, phase_idx])
                v1 = float(self.data[l, k, phase_idx_next])
                val = v0 * (1.0 - alpha) + v1 * alpha
                output[l, k] = int(np.clip(val, -32768, 32767))

    @staticmethod
    def generate_for_frequency(
        wave_id: int,
        name: str,
        target_freq_hz: float,
        generate_sample_fn,  # Callable[[int], int] - phase -> sample (Q15)
        K: int = DEFAULT_K,
        P: int = DEFAULT_P,
        normalize: bool = True,
        S: int = 8,  # Oversampling factor for smooth subband extraction
    ) -> 'StreamingCoeffLUT':
        """
        Generate StreamingCoeffLUT for a SPECIFIC FREQUENCY.

        Correct implementation per WSE-STD-121:
        1. Generate high-resolution periodic waveform (M = P * K * S samples)
        2. Run streaming analysis filterbank to get subband time-series
        3. Map (p, k) -> n(p,k) to extract table entries

        The key insight: K is polyphase time slots within the block.
        Each k produces N = 64/K output samples when fed to streaming IDWPT.

        Args:
            wave_id: Waveform ID
            name: Waveform name
            target_freq_hz: Target frequency in Hz (e.g., 261.63 for C4)
            generate_sample_fn: Function(phase_32) -> sample (Q15)
            K: Polyphase time slots per block (default 8)
            P: Phase steps per cycle (default 64)
            normalize: Apply normalization
            S: Oversampling factor (default 8)

        Returns:
            StreamingCoeffLUT with populated data
        """
        logger.info(f"Generating StreamingCoeffLUT for {name} at {target_freq_hz:.2f} Hz (K={K}, P={P})")

        L = NUM_LEAVES

        # The table encodes subband time-series for a periodic oscillator.
        # P phase steps represent ONE complete oscillator cycle.
        # K polyphase slots represent time within a block.
        #
        # KEY INSIGHT: Generate exactly ONE oscillator cycle of audio,
        # analyze it with FWPT, then map P*K table entries uniformly
        # across the resulting subband stream.

        # Phase increment for target frequency (samples at 48kHz)
        phase_inc = int((target_freq_hz / SAMPLE_RATE) * (1 << 32))

        # Step 0: Calculate how many samples make one oscillator cycle
        # Round up to nearest multiple of BLOCK_SIZE for clean FWPT blocking
        samples_per_osc_cycle = SAMPLE_RATE / target_freq_hz
        num_blocks_per_cycle = max(1, int(np.ceil(samples_per_osc_cycle / BLOCK_SIZE)))
        M = num_blocks_per_cycle * BLOCK_SIZE  # Actual samples to generate

        logger.debug(f"  samples_per_osc_cycle={samples_per_osc_cycle:.1f}")
        logger.debug(f"  num_blocks_per_cycle={num_blocks_per_cycle}, M={M}")

        # Generate exactly enough samples to cover one oscillator cycle
        # (might be slightly more than one cycle due to rounding up)
        waveform = np.zeros(M, dtype=np.float32)
        phase = 0
        for n in range(M):
            waveform[n] = generate_sample_fn(phase)
            phase = (phase + phase_inc) & 0xFFFFFFFF

        # Step 1: Run streaming analysis filterbank over the waveform
        # Process M/BLOCK_SIZE blocks, get L coefficients each
        num_analysis_blocks = M // BLOCK_SIZE
        subband_streams = np.zeros((L, num_analysis_blocks), dtype=np.float32)

        for block_idx in range(num_analysis_blocks):
            start = block_idx * BLOCK_SIZE
            audio_block = waveform[start:start + BLOCK_SIZE].astype(np.int16).tolist()
            coeffs = fwpt_jit(audio_block, headroom_bits=1)
            for l in range(L):
                subband_streams[l, block_idx] = coeffs[l]

        logger.debug(f"  num_analysis_blocks={num_analysis_blocks}")

        # Step 2: Map (p, k) -> subband sample index
        # P phase steps should span the entire subband stream (one cycle)
        # K polyphase slots subdivide each phase step

        data = np.zeros((L, K, P), dtype=np.float32)

        for p in range(P):
            for k in range(K):
                # (p, k) maps to a fraction of the cycle
                # frac = (p*K + k) / (P*K) spans [0, 1)
                frac_in_cycle = (p * K + k) / (P * K)

                # Map uniformly across all analysis blocks
                block_idx_float = frac_in_cycle * num_analysis_blocks

                # Linear interpolation between adjacent subband samples
                idx0 = int(block_idx_float) % num_analysis_blocks
                idx1 = (idx0 + 1) % num_analysis_blocks
                alpha = block_idx_float - int(block_idx_float)

                for l in range(L):
                    v0 = subband_streams[l, idx0]
                    v1 = subband_streams[l, idx1]
                    data[l, k, p] = v0 * (1.0 - alpha) + v1 * alpha

        # Normalize if requested
        if normalize:
            # DC removal across all phases
            dc = data.mean()
            data -= dc

            # RMS normalization to target level
            rms = np.sqrt(np.mean(data ** 2))
            if rms > 1e-6:
                scale = TARGET_RMS_Q15 / rms
                data *= scale

            # Peak safety clamp
            peak = np.abs(data).max()
            max_allowed = 0.95 * Q15_ONE
            if peak > max_allowed:
                data *= max_allowed / peak

        # Quantize to int16
        data_int = np.clip(data, -32768, 32767).astype(np.int16)

        return StreamingCoeffLUT(
            wave_id=wave_id,
            name=name,
            data=data_int,
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
        """
        Generate StreamingCoeffLUT at default frequency (C4 = 261.63 Hz).

        For per-note tables, use generate_for_frequency() instead.
        """
        return StreamingCoeffLUT.generate_for_frequency(
            wave_id=wave_id,
            name=name,
            target_freq_hz=261.63,  # C4
            generate_sample_fn=generate_sample_fn,
            K=K,
            P=P,
            normalize=normalize,
        )

    @staticmethod
    def generate_sine(K: int = DEFAULT_K, P: int = DEFAULT_P) -> 'StreamingCoeffLUT':
        """Generate sine wave StreamingCoeffLUT."""
        return StreamingCoeffLUT.generate_from_waveform(
            wave_id=0,
            name="sine",
            generate_sample_fn=sine_q15,
            K=K,
            P=P,
        )

    @staticmethod
    def generate_saw(K: int = DEFAULT_K, P: int = DEFAULT_P) -> 'StreamingCoeffLUT':
        """Generate bandlimited sawtooth wave."""
        def saw_sample(phase: int) -> int:
            # Bandlimited sawtooth using first 16 harmonics
            val = 0
            for h in range(1, 17):
                h_phase = (phase * h) & 0xFFFFFFFF
                val += sine_q15(h_phase) // h
            return int(np.clip(val // 2, -32768, 32767))

        return StreamingCoeffLUT.generate_from_waveform(
            wave_id=1,
            name="saw",
            generate_sample_fn=saw_sample,
            K=K,
            P=P,
        )

    @staticmethod
    def generate_square(K: int = DEFAULT_K, P: int = DEFAULT_P) -> 'StreamingCoeffLUT':
        """Generate bandlimited square wave."""
        def square_sample(phase: int) -> int:
            # Square wave: odd harmonics only
            val = 0
            for h in [1, 3, 5, 7, 9, 11, 13, 15]:
                h_phase = (phase * h) & 0xFFFFFFFF
                val += sine_q15(h_phase) // h
            return int(np.clip(val, -32768, 32767))

        return StreamingCoeffLUT.generate_from_waveform(
            wave_id=2,
            name="square",
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
# Streaming IDWPT (Polyphase Synthesis)
# =============================================================================

def streaming_idwpt_block(
    state: SubbandState,
    output: np.ndarray,
    headroom_bits: int = 1,
) -> None:
    """
    Streaming IDWPT synthesis from subband state.

    STEPPING STONE IMPLEMENTATION:
    This is NOT the full polyphase streaming IDWPT. Instead:
    1. Average the K time samples per leaf to get 64 coefficients
    2. Run standard block IDWPT
    3. This provides correct structure while we build the real polyphase core

    True polyphase implementation (future) would:
    - Maintain filter state across blocks
    - Process K samples per leaf in a specific interleaved pattern
    - Produce 64 audio samples from the polyphase combination

    Args:
        state: SubbandState with current subband time-series
        output: np.ndarray[BLOCK_SIZE] int16 output buffer
        headroom_bits: Headroom scaling
    """
    K = state.state.shape[1]

    # STEPPING STONE: Use first time sample (k=0) from each leaf
    # The K dimension represents time samples within the block, not independent measurements
    # For a sinusoidal subband, the K samples form a time series that would average to ~0
    # So we pick one representative sample per leaf instead of averaging
    #
    # True polyphase would process all K samples with proper filter state
    coeffs_list = [int(state.state[l, 0]) for l in range(NUM_LEAVES)]

    # Run standard block IDWPT
    audio = idwpt_jit(coeffs_list, headroom_bits=headroom_bits)

    # Copy to output
    for i in range(min(BLOCK_SIZE, len(audio))):
        output[i] = audio[i]


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
# Per-Note LUT Cache (for frequency-specific tables)
# =============================================================================

_NOTE_LUT_CACHE: Dict[Tuple[int, int], StreamingCoeffLUT] = {}


def get_lut_for_note(wave_id: int, midi_note: int) -> StreamingCoeffLUT:
    """
    Get StreamingCoeffLUT for a specific MIDI note.

    Tables are generated and cached on first access. Each note gets its own
    table because the frequency is baked into the FWPT coefficients.

    Args:
        wave_id: Waveform ID (0=sine, 1=saw, 2=square)
        midi_note: MIDI note number (0-127)

    Returns:
        StreamingCoeffLUT for this note/waveform combination
    """
    cache_key = (wave_id, midi_note)
    if cache_key not in _NOTE_LUT_CACHE:
        freq_hz = midi_to_hz(midi_note)
        logger.debug(f"Generating LUT for wave={wave_id}, note={midi_note} ({freq_hz:.1f} Hz)")

        # Get waveform generator
        if wave_id == 0:
            gen_fn = sine_q15
            name = "sine"
        elif wave_id == 1:
            def gen_fn(phase):
                val = 0
                for h in range(1, 17):
                    h_phase = (phase * h) & 0xFFFFFFFF
                    val += sine_q15(h_phase) // h
                return int(np.clip(val // 2, -32768, 32767))
            name = "saw"
        elif wave_id == 2:
            def gen_fn(phase):
                val = 0
                for h in [1, 3, 5, 7, 9, 11, 13, 15]:
                    h_phase = (phase * h) & 0xFFFFFFFF
                    val += sine_q15(h_phase) // h
                return int(np.clip(val, -32768, 32767))
            name = "square"
        else:
            gen_fn = sine_q15
            name = "sine"

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


__all__ = [
    'StreamingCoeffLUT',
    'SubbandState',
    'CCSState',
    'get_streaming_lut',
    'get_lut_for_note',
    'streaming_idwpt_block',
    'ccs_generate_coeffs',
    'DEFAULT_K',
    'DEFAULT_P',
    'BLOCK_SIZE',
    'NUM_LEAVES',
    'midi_to_hz',
]
