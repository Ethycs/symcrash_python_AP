"""
Composite A+B Oscillator (WSE-STD-211)
======================================

Combines Mode A (LUT-based) and Mode B (Harmonic Bank) for optimal synthesis:

- Mode A: Fast LUT lookup for basic waveforms (sine, saw, square, triangle)
  - Quarter-phase symmetry storage: 4× memory reduction
  - 8-bit coefficient option: additional 2× reduction
  - Best for: static waveforms, polyphony under memory pressure

- Mode B: Runtime harmonic synthesis
  - PWM modulation
  - Audio-rate FM/PM
  - Dynamic waveform morphing
  - Best for: expressive timbres, modulation-heavy patches

Composite Blending:
    c_mix = blend_q15 * c_B + (32767 - blend_q15) * c_A >> 15

Benefits:
    - Mode A provides stable reference (no accumulated errors)
    - Mode B provides modulation capability
    - Blend allows smooth transitions and hybrid timbres
    - If Mode A cache misses deadline, pure B fallback

Memory Budget (25F with 128KB BRAM):
    - Mode A (4 waves, quarter-phase, 8-bit): ~8 KB total
    - Mode B basis atlas (8 anchors): ~32 KB
    - Per-voice coeff caches (32 voices × 1KB): ~32 KB
    - Remaining: ~56 KB for buffers, state, sine LUT
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from ..fixed import RSHIFT_RNE, SAT16, Q15_ONE
from .coeff_lut_streaming import (
    StreamingCoeffLUT, CCSState, get_lut_for_note,
    WAVE_SINE, WAVE_SAW, WAVE_SQUARE, WAVE_TRIANGLE, WAVE_PULSE,
    NUM_LEAVES, DEFAULT_K, INTERP_LINEAR, INTERP_CUBIC,
)
from .harmonic_bank import (
    ModeBState, HarmonicBasisFrames, BasisFrameAtlas,
    mode_b_generate_coeffs, DEFAULT_H_MAX,
)

# =============================================================================
# CONSTANTS
# =============================================================================

COEFF_LEN_POLYPHASE = NUM_LEAVES * DEFAULT_K  # 64 * 8 = 512

# Composite blend modes
BLEND_PURE_A = 0        # 100% Mode A
BLEND_PURE_B = 32767    # 100% Mode B
BLEND_EQUAL = 16384     # 50/50 mix

# Quarter-phase quadrants
QUADRANT_0 = 0  # 0-90°: use table directly
QUADRANT_1 = 1  # 90-180°: mirror index
QUADRANT_2 = 2  # 180-270°: mirror + negate
QUADRANT_3 = 3  # 270-360°: negate only


# =============================================================================
# HALF-PHASE STORAGE (2× memory reduction)
# =============================================================================
# Note: Wavelet coefficients don't have simple quarter-wave symmetry like
# time-domain signals. We use half-wave storage with sign inversion, which
# is mathematically correct for periodic waveforms.

@dataclass
class HalfPhaseLUT:
    """
    Half-phase coefficient LUT with sign reconstruction.

    For all periodic waveforms, the second half of one cycle is the
    sign-inverted version of the first half (for zero-mean waveforms).

    Storage: P/2 phases instead of P → 2× reduction
    With 8-bit coefficients: additional 2× → 4× total reduction
    """
    wave_id: int
    name: str

    # Half-phase data: [L leaves, K time samples, P/2 phases]
    data: np.ndarray  # dtype: int8 or int16

    # Parameters
    num_leaves: int = NUM_LEAVES
    K: int = DEFAULT_K
    half_P: int = 32  # P/2 - e.g., 32 for full P=64

    # 8-bit storage flag
    is_8bit: bool = False

    # Target frequency
    target_freq_hz: float = 261.63

    def get_subband_samples(
        self,
        phase_32: int,
        output: np.ndarray,
    ) -> None:
        """
        Reconstruct full-phase coefficients from half-phase storage.

        Second half of cycle uses sign inversion.
        """
        full_P = self.half_P * 2

        # Extract phase index for full P
        if full_P == 64:
            shift = 26
            mask = 0x3F
        elif full_P == 128:
            shift = 25
            mask = 0x7F
        elif full_P == 32:
            shift = 27
            mask = 0x1F
        else:
            import math
            shift = 32 - int(math.log2(full_P))
            mask = full_P - 1

        full_idx = (phase_32 >> shift) & mask

        # First half: use directly, second half: negate
        if full_idx < self.half_P:
            table_idx = full_idx
            sign = 1
        else:
            table_idx = full_idx - self.half_P
            sign = -1

        # Clamp to valid range
        table_idx = min(table_idx, self.half_P - 1)

        # Fetch coefficients
        coeffs = self.data[:, :, table_idx]

        # Apply sign and scale (8-bit → 16-bit if needed)
        if self.is_8bit:
            # Scale 8-bit to 16-bit: multiply by 256
            output[:, :] = (sign * coeffs.astype(np.int16)) << 8
        else:
            output[:, :] = sign * coeffs

    @staticmethod
    def from_full_lut(
        full_lut: StreamingCoeffLUT,
        use_8bit: bool = False,
    ) -> 'HalfPhaseLUT':
        """
        Create half-phase LUT from full StreamingCoeffLUT.

        Args:
            full_lut: Full-phase LUT to compress
            use_8bit: Store as 8-bit (additional 2× reduction)

        Returns:
            HalfPhaseLUT with half-phase storage
        """
        full_P = full_lut.P
        half_P = full_P // 2

        # Store first half
        half_data = full_lut.data[:, :, :half_P].copy()

        # Convert to 8-bit if requested
        if use_8bit:
            # Scale to 8-bit range: divide by 256, round to nearest
            half_data_8bit = ((half_data.astype(np.int32) + 128) >> 8).astype(np.int8)
            return HalfPhaseLUT(
                wave_id=full_lut.wave_id,
                name=full_lut.name + "_h8",
                data=half_data_8bit,
                num_leaves=full_lut.num_leaves,
                K=full_lut.K,
                half_P=half_P,
                is_8bit=True,
                target_freq_hz=full_lut.target_freq_hz,
            )
        else:
            return HalfPhaseLUT(
                wave_id=full_lut.wave_id,
                name=full_lut.name + "_h16",
                data=half_data,
                num_leaves=full_lut.num_leaves,
                K=full_lut.K,
                half_P=half_P,
                is_8bit=False,
                target_freq_hz=full_lut.target_freq_hz,
            )


# Legacy alias for backwards compatibility
QuarterPhaseLUT = HalfPhaseLUT


# =============================================================================
# COMPOSITE OSCILLATOR STATE
# =============================================================================

@dataclass
class CompositeOscState:
    """
    Per-voice composite oscillator state.

    Combines Mode A and Mode B state with blending control.
    """
    # Mode A state (LUT-based)
    mode_a: CCSState = field(default_factory=CCSState)

    # Mode B state (Harmonic Bank)
    mode_b: ModeBState = field(default_factory=ModeBState)

    # Blend control
    blend_q15: int = BLEND_PURE_A  # 0=pure A, 32767=pure B
    target_blend_q15: int = BLEND_PURE_A
    blend_slew_per_block: int = 0  # Slew rate

    # Mode selection (for pure mode operation)
    active_mode: int = 0  # 0=composite, 1=pure A, 2=pure B

    # Coefficient cache (avoid re-allocation)
    _coeff_a: np.ndarray = field(
        default_factory=lambda: np.zeros(COEFF_LEN_POLYPHASE, dtype=np.int16)
    )
    _coeff_b: np.ndarray = field(
        default_factory=lambda: np.zeros(COEFF_LEN_POLYPHASE, dtype=np.int16)
    )

    def reset(self, note: int, phase_inc: int, wave_id: int = WAVE_SAW) -> None:
        """Reset both modes for a new note."""
        self.mode_a.reset(note=note, phase_inc=phase_inc, wave_id=wave_id)
        self.mode_b.reset(note=note, phase_inc=phase_inc)

        # Sync phases
        self.mode_b.phase = self.mode_a.phase

    def set_blend(self, blend_q15: int, slew_blocks: int = 0) -> None:
        """Set blend target with optional slew."""
        self.target_blend_q15 = max(0, min(Q15_ONE, blend_q15))
        if slew_blocks <= 0:
            self.blend_q15 = self.target_blend_q15
            self.blend_slew_per_block = 0
        else:
            diff = self.target_blend_q15 - self.blend_q15
            self.blend_slew_per_block = diff // slew_blocks

    def advance_block(self) -> None:
        """Advance both oscillators by one block."""
        # Advance Mode A
        self.mode_a.advance_block()

        # Advance Mode B (keeps phase synced)
        self.mode_b.phase = self.mode_a.phase
        self.mode_b.phase_inc = self.mode_a.phase_inc

        # Apply blend slew
        if self.blend_slew_per_block != 0:
            self.blend_q15 += self.blend_slew_per_block
            if self.blend_slew_per_block > 0:
                if self.blend_q15 >= self.target_blend_q15:
                    self.blend_q15 = self.target_blend_q15
                    self.blend_slew_per_block = 0
            else:
                if self.blend_q15 <= self.target_blend_q15:
                    self.blend_q15 = self.target_blend_q15
                    self.blend_slew_per_block = 0


# =============================================================================
# COMPOSITE COEFFICIENT GENERATION
# =============================================================================

def composite_generate_coeffs(
    state: CompositeOscState,
    output_coeffs: np.ndarray,
    basis_atlas: Optional[BasisFrameAtlas] = None,
) -> None:
    """
    Generate blended coefficients from Mode A + Mode B.

    Per WSE-STD-211:
        c_out = ((Q15_ONE - blend) * c_A + blend * c_B) >> 15

    If blend_q15 == 0: pure Mode A (fast path)
    If blend_q15 == Q15_ONE: pure Mode B (fast path)

    Args:
        state: CompositeOscState with mode A/B states
        output_coeffs: Output buffer [512] int16
        basis_atlas: Mode B basis frame atlas (required if blend > 0)
    """
    blend = state.blend_q15

    # Fast path: pure Mode A
    if blend == 0 or state.active_mode == 1:
        from .coeff_lut_streaming import ccs_generate_coeffs_polyphase
        ccs_generate_coeffs_polyphase(state.mode_a, output_coeffs)
        return

    # Fast path: pure Mode B
    if blend >= Q15_ONE or state.active_mode == 2:
        if basis_atlas is None:
            # Fallback to Mode A if no atlas
            from .coeff_lut_streaming import ccs_generate_coeffs_polyphase
            ccs_generate_coeffs_polyphase(state.mode_a, output_coeffs)
            return
        basis = basis_atlas.get_frames(state.mode_b.note)
        mode_b_generate_coeffs(state.mode_b, basis, output_coeffs)
        return

    # Composite path: blend A + B
    from .coeff_lut_streaming import ccs_generate_coeffs_polyphase

    # Generate Mode A coefficients
    ccs_generate_coeffs_polyphase(state.mode_a, state._coeff_a)

    # Generate Mode B coefficients
    if basis_atlas is not None:
        basis = basis_atlas.get_frames(state.mode_b.note)
        mode_b_generate_coeffs(state.mode_b, basis, state._coeff_b)
    else:
        # No basis atlas - use A for B too
        state._coeff_b[:] = state._coeff_a

    # Blend: c_out = ((32767 - blend) * c_A + blend * c_B) >> 15
    inv_blend = Q15_ONE - blend
    for i in range(COEFF_LEN_POLYPHASE):
        a_val = int(state._coeff_a[i])
        b_val = int(state._coeff_b[i])
        blended = (inv_blend * a_val + blend * b_val + 16384) >> 15
        output_coeffs[i] = SAT16(blended)


def composite_generate_coeffs_with_fallback(
    state: CompositeOscState,
    output_coeffs: np.ndarray,
    basis_atlas: Optional[BasisFrameAtlas] = None,
    mode_a_available: bool = True,
) -> None:
    """
    Generate coefficients with automatic fallback.

    If Mode A cache misses its deadline (mode_a_available=False),
    automatically use pure Mode B to avoid audio dropout.

    Args:
        state: CompositeOscState
        output_coeffs: Output buffer [512]
        basis_atlas: Mode B basis atlas
        mode_a_available: False if Mode A cache missed deadline
    """
    if not mode_a_available and basis_atlas is not None:
        # Force pure Mode B
        basis = basis_atlas.get_frames(state.mode_b.note)
        mode_b_generate_coeffs(state.mode_b, basis, output_coeffs)
    else:
        # Normal composite generation
        composite_generate_coeffs(state, output_coeffs, basis_atlas)


# =============================================================================
# MEMORY SIZE CALCULATIONS
# =============================================================================

def calculate_mode_a_memory(
    num_waves: int = 4,
    P: int = 64,
    K: int = DEFAULT_K,
    L: int = NUM_LEAVES,
    use_half_phase: bool = True,
    use_8bit: bool = True,
) -> int:
    """
    Calculate Mode A table memory requirements.

    Args:
        num_waves: Number of waveforms (sine, saw, square, triangle)
        P: Phase resolution
        K: Time samples per subband
        L: Number of leaves
        use_half_phase: Use half-phase symmetry (2× reduction)
        use_8bit: Use 8-bit coefficients (2× reduction)

    Returns:
        Total memory in bytes
    """
    bytes_per_coeff = 1 if use_8bit else 2
    phase_divisor = 2 if use_half_phase else 1

    per_wave = L * K * (P // phase_divisor) * bytes_per_coeff
    return num_waves * per_wave


def print_memory_budget(
    num_waves: int = 4,
    num_voices: int = 32,
    num_anchors: int = 4,  # Reduced from 8 - use interpolation between anchors
    P: int = 64,
    H_max: int = 16,  # Reduced from 32 - sufficient for most timbres
):
    """Print detailed memory budget for 25F FPGA."""
    print("=" * 60)
    print("CCS Composite A+B Memory Budget (LFE5U-25F)")
    print("=" * 60)

    # Mode A variants
    full_a = calculate_mode_a_memory(num_waves, P, use_half_phase=False, use_8bit=False)
    half_a = calculate_mode_a_memory(num_waves, P, use_half_phase=True, use_8bit=False)
    compact_a = calculate_mode_a_memory(num_waves, P, use_half_phase=True, use_8bit=True)

    print(f"\nMode A (per {num_waves} waves, P={P}):")
    print(f"  Full tables (16-bit):         {full_a:>8,} bytes ({full_a/1024:.1f} KB)")
    print(f"  Half-phase (16-bit):          {half_a:>8,} bytes ({half_a/1024:.1f} KB)")
    print(f"  Half-phase + 8-bit:           {compact_a:>8,} bytes ({compact_a/1024:.1f} KB)")

    # Mode B basis atlas (optimized)
    per_anchor = (2 * H_max + 1) * NUM_LEAVES * DEFAULT_K * 2  # DC + cos + sin
    mode_b_atlas = num_anchors * per_anchor

    print(f"\nMode B (H_max={H_max}, {num_anchors} anchors):")
    print(f"  Per anchor:                   {per_anchor:>8,} bytes ({per_anchor/1024:.1f} KB)")
    print(f"  Full atlas:                   {mode_b_atlas:>8,} bytes ({mode_b_atlas/1024:.1f} KB)")

    # Per-voice state
    per_voice = COEFF_LEN_POLYPHASE * 2  # Coefficient cache
    voice_total = num_voices * per_voice

    print(f"\nPer-voice ({num_voices} voices):")
    print(f"  Coefficient cache each:       {per_voice:>8,} bytes")
    print(f"  Total voice caches:           {voice_total:>8,} bytes ({voice_total/1024:.1f} KB)")

    # Misc
    sine_lut = 2048  # 1K entries × 2 bytes
    misc_buffers = 4096  # Output buffers, control state

    print(f"\nMisc:")
    print(f"  Sine LUT:                     {sine_lut:>8,} bytes")
    print(f"  Output/control buffers:       {misc_buffers:>8,} bytes")

    # Totals
    total_compact = compact_a + mode_b_atlas + voice_total + sine_lut + misc_buffers
    bram_available = 128 * 1024  # 128 KB for 25F

    print(f"\n" + "-" * 40)
    print(f"Total (compact Mode A):         {total_compact:>8,} bytes ({total_compact/1024:.1f} KB)")
    print(f"BRAM available (25F):           {bram_available:>8,} bytes ({bram_available/1024:.1f} KB)")
    print(f"Headroom:                       {bram_available - total_compact:>8,} bytes ({(bram_available - total_compact)/1024:.1f} KB)")
    print(f"Utilization:                    {100*total_compact/bram_available:.1f}%")
    print("=" * 60)


def print_fpga_fit_config():
    """
    Print a memory configuration that fits the 25F FPGA.

    Key optimizations:
    1. Mode A only (no Mode B atlas) - Mode B can generate at runtime
    2. Reduced K=4 (smaller tables, still smooth)
    3. Half-phase + 8-bit storage
    4. Fewer voices if needed
    """
    print("=" * 60)
    print("FPGA-Fit Configuration (LFE5U-25F)")
    print("=" * 60)

    # Aggressive Mode A configuration
    P = 64
    K = 4  # Reduced from 8
    L = NUM_LEAVES

    # Mode A: 4 waves, half-phase, 8-bit
    per_wave = L * K * (P // 2) * 1  # 64 * 4 * 32 * 1 = 8 KB
    mode_a_total = 4 * per_wave

    print(f"\nMode A (P={P}, K={K}, half-phase, 8-bit):")
    print(f"  Per wave:                     {per_wave:>8,} bytes ({per_wave/1024:.1f} KB)")
    print(f"  4 waves total:                {mode_a_total:>8,} bytes ({mode_a_total/1024:.1f} KB)")

    # Mode B: runtime generation only (no atlas in BRAM)
    # Use Mode A's phase model, compute harmonics on-the-fly
    print(f"\nMode B: Runtime generation (no BRAM atlas)")
    print(f"  Sine LUT (shared):            {2048:>8,} bytes (2.0 KB)")

    # Per-voice: coefficient state only
    num_voices = 16  # Reduced for FPGA fit
    per_voice = L * K * 2  # 64 * 4 * 2 = 512 bytes
    voice_total = num_voices * per_voice

    print(f"\nPer-voice ({num_voices} voices, K={K}):")
    print(f"  Coefficient state each:       {per_voice:>8,} bytes")
    print(f"  Total voice state:            {voice_total:>8,} bytes ({voice_total/1024:.1f} KB)")

    # Output buffers and misc
    output_buffers = 64 * 2 * 2  # Stereo output double-buffer
    misc = 2048  # Control state, registers

    print(f"\nMisc:")
    print(f"  Output buffers:               {output_buffers:>8,} bytes")
    print(f"  Control/misc:                 {misc:>8,} bytes")

    # Total
    total = mode_a_total + 2048 + voice_total + output_buffers + misc
    bram = 128 * 1024

    print(f"\n" + "-" * 40)
    print(f"Total:                          {total:>8,} bytes ({total/1024:.1f} KB)")
    print(f"BRAM available:                 {bram:>8,} bytes ({bram/1024:.1f} KB)")
    print(f"Headroom:                       {bram - total:>8,} bytes ({(bram - total)/1024:.1f} KB)")
    print(f"Utilization:                    {100*total/bram:.1f}%")

    print(f"\n" + "=" * 60)
    print("Configuration Summary:")
    print("  - Mode A: 4 basic waves (sine, saw, square, tri)")
    print("  - Half-phase storage with 8-bit coefficients")
    print("  - K=4 polyphase lanes (vs 8 default)")
    print("  - 16 polyphonic voices")
    print("  - Mode B: runtime harmonic synthesis (no atlas)")
    print("  - Composite blending supported")
    print("=" * 60)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Half-phase LUT
    'HalfPhaseLUT',
    'QuarterPhaseLUT',  # Legacy alias
    'QUADRANT_0', 'QUADRANT_1', 'QUADRANT_2', 'QUADRANT_3',

    # Composite state
    'CompositeOscState',
    'BLEND_PURE_A', 'BLEND_PURE_B', 'BLEND_EQUAL',

    # Coefficient generation
    'composite_generate_coeffs',
    'composite_generate_coeffs_with_fallback',

    # Memory utilities
    'calculate_mode_a_memory',
    'print_memory_budget',
    'print_fpga_fit_config',

    # Constants
    'COEFF_LEN_POLYPHASE',
]
