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
        """Fill `output` with per-leaf subband samples for the given phase.

        This runs in the real-time path. It MUST be deterministic and fixed-point
        only, so the phase interpolation is done with integer math.

        Args:
            phase_32: 32-bit phase accumulator value
            output: np.ndarray[num_leaves, K] int16 (modified in place)
        """
        # P is 64 by default; top 6 bits select phase index.
        phase_idx = (phase_32 >> 26) & 0x3F
        phase_idx_next = (phase_idx + 1) & 0x3F

        # 26-bit fraction for interpolation in [0, 2^26).
        frac = phase_32 & 0x3FFFFFF

        # Fixed-point lerp:
        #   val = v0 + (v1 - v0) * frac / 2^26
        # with rounding to nearest.
        v0 = self.data[:, :, phase_idx].astype(np.int32)
        v1 = self.data[:, :, phase_idx_next].astype(np.int32)
        diff = v1 - v0
        val = v0 + ((diff * frac + (1 << 25)) >> 26)

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
        S: int = 8,  # Oversampling factor for smooth subband extraction
    ) -> 'StreamingCoeffLUT':
        """Generate StreamingCoeffLUT for a specific frequency.

        Note: This is an offline table build routine; it may use floating point.
        The runtime lookup path (`get_subband_samples`) is fixed-point.
        """
        logger.info(
            f"Generating StreamingCoeffLUT for {name} at {target_freq_hz:.2f} Hz (K={K}, P={P})"
        )

        L = NUM_LEAVES

        # Phase increment for target frequency (samples at SAMPLE_RATE)
        phase_inc = int((target_freq_hz / SAMPLE_RATE) * (1 << 32))

        # Generate enough samples to cover one oscillator cycle (rounded up to whole blocks)
        samples_per_osc_cycle = SAMPLE_RATE / target_freq_hz
        num_blocks_per_cycle = max(1, int(np.ceil(samples_per_osc_cycle / BLOCK_SIZE)))
        M = num_blocks_per_cycle * BLOCK_SIZE

        waveform = np.zeros(M, dtype=np.float32)
        phase = 0
        for n in range(M):
            waveform[n] = generate_sample_fn(phase)
            phase = (phase + phase_inc) & 0xFFFFFFFF

        # Run analysis (FWPT) per block; collect 64 leaf streams over the cycle
        num_analysis_blocks = M // BLOCK_SIZE
        subband_streams = np.zeros((L, num_analysis_blocks), dtype=np.float32)

        for block_idx in range(num_analysis_blocks):
            start = block_idx * BLOCK_SIZE
            audio_block = waveform[start:start + BLOCK_SIZE].astype(np.int16).tolist()
            coeffs = fwpt_jit(audio_block, headroom_bits=1)
            for l in range(L):
                subband_streams[l, block_idx] = coeffs[l]

        # Map (p, k) to subband stream samples uniformly across the cycle
        data = np.zeros((L, K, P), dtype=np.float32)

        for p in range(P):
            for k in range(K):
                frac_in_cycle = (p * K + k) / (P * K)
                block_idx_float = frac_in_cycle * num_analysis_blocks

                idx0 = int(block_idx_float) % num_analysis_blocks
                idx1 = (idx0 + 1) % num_analysis_blocks
                alpha = block_idx_float - int(block_idx_float)

                v0 = subband_streams[:, idx0]
                v1 = subband_streams[:, idx1]
                data[:, k, p] = v0 * (1.0 - alpha) + v1 * alpha

        if normalize:
            dc = data.mean()
            data -= dc

            rms = np.sqrt(np.mean(data ** 2))
            if rms > 1e-6:
                scale = TARGET_RMS_Q15 / rms
                data *= scale

            peak = float(np.abs(data).max())
            max_allowed = 0.95 * Q15_ONE
            if peak > max_allowed:
                data *= max_allowed / peak

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


# VM-side K dimension (matches 256 coeff_state = 64 leaves × 4 time slots)
VM_K = 4


def ccs_generate_coeffs_polyphase(
    ccs_state: CCSState,
    output_coeffs: np.ndarray,
) -> None:
    """
    Generate polyphase coefficients: 64 leaves × 4 time slots = 256 values.

    TRUE POLYPHASE INJECTION per WSE-STD-120:
    - Each k-slot gets a different sub-phase position
    - coeff_state[leaf][k] = LUT[wave][leaf][k_mapped][phase_idx]

    The LUT has K=8 time samples, we map to VM's K=4:
    - VM k=0 → LUT k=0
    - VM k=1 → LUT k=2
    - VM k=2 → LUT k=4
    - VM k=3 → LUT k=6

    Args:
        ccs_state: CCS oscillator state
        output_coeffs: np.ndarray[256] int16 output buffer (modified in place)
                       Layout: [leaf0_k0, leaf0_k1, leaf0_k2, leaf0_k3,
                                leaf1_k0, leaf1_k1, ...]
    """
    # Get LUT for this note
    lut = get_lut_for_note(ccs_state.wave_id, ccs_state.note)

    # Get all K subband samples for current phase
    subband_output = np.zeros((NUM_LEAVES, lut.K), dtype=np.int16)
    lut.get_subband_samples(ccs_state.phase, subband_output)

    # Map LUT K=8 to VM K=4 (downsample by 2)
    # This preserves the temporal structure while fitting the 256-state budget
    lut_k_map = [0, 2, 4, 6]  # LUT indices for VM k=0,1,2,3

    # Inject into 256-element state: [64 leaves × 4 k-slots]
    for l in range(NUM_LEAVES):
        for k in range(VM_K):
            lut_k = lut_k_map[k]
            if lut_k < lut.K:
                output_coeffs[l * VM_K + k] = subband_output[l, lut_k]
            else:
                # Fallback if LUT has fewer K samples
                output_coeffs[l * VM_K + k] = subband_output[l, 0]


__all__ = [
    'StreamingCoeffLUT',
    'SubbandState',
    'CCSState',
    'get_streaming_lut',
    'get_lut_for_note',
    'streaming_idwpt_block',
    'ccs_generate_coeffs',
    'ccs_generate_coeffs_polyphase',
    'VM_K',
    'DEFAULT_K',
    'DEFAULT_P',
    'BLOCK_SIZE',
    'NUM_LEAVES',
    'midi_to_hz',
]
