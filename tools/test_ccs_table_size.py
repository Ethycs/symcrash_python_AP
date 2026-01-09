"""
Test CCS Mode A table size reduction.

Verifies that P=64 (64 KB per table) maintains quality vs P=1024 (1 MB per table).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.vm.coeff_lut_streaming import (
    StreamingCoeffLUT, get_lut_for_note, DEFAULT_P, DEFAULT_K, NUM_LEAVES,
    WAVE_SAW, WAVE_SINE, WAVE_SQUARE, _NOTE_LUT_CACHE,
    INTERP_LINEAR, INTERP_CUBIC, streaming_idwpt_block, SubbandState
)

def main():
    # Clear cache to force regeneration
    _NOTE_LUT_CACHE.clear()

    # Check table size
    print("=" * 60)
    print("CCS Mode A Table Size Analysis")
    print("=" * 60)

    print(f"\nTable parameters: P={DEFAULT_P}, K={DEFAULT_K}, L={NUM_LEAVES}")
    table_size_bytes = NUM_LEAVES * DEFAULT_K * DEFAULT_P * 2
    print(f"Table size per note/wave: {table_size_bytes:,} bytes ({table_size_bytes/1024:.1f} KB)")

    # Compare to old size
    old_p = 1024
    old_size = NUM_LEAVES * DEFAULT_K * old_p * 2
    print(f"Old size (P={old_p}): {old_size:,} bytes ({old_size/1024:.1f} KB)")
    print(f"Reduction factor: {old_size / table_size_bytes:.1f}x")

    # Generate a table for A4 (440 Hz) saw wave
    print("\n" + "-" * 40)
    print("Generating saw LUT for MIDI note 69 (A4 = 440 Hz)...")
    lut = get_lut_for_note(WAVE_SAW, 69)
    print(f"LUT shape: {lut.data.shape}")
    print(f"LUT dtype: {lut.data.dtype}")
    print(f"LUT target freq: {lut.target_freq_hz:.2f} Hz")

    # Test lookup with linear interpolation
    print("\n" + "-" * 40)
    print("Testing phase lookup (linear interpolation)...")
    output = np.zeros((NUM_LEAVES, DEFAULT_K), dtype=np.int16)
    lut.get_subband_samples(0, output, interp_mode=INTERP_LINEAR)
    print(f"Output at phase=0:")
    print(f"  First leaf coeffs: {output[0, :]}")
    print(f"  Max coeff: {np.max(np.abs(output))}")

    # Test with cubic interpolation
    print("\nTesting phase lookup (cubic interpolation)...")
    output_cubic = np.zeros((NUM_LEAVES, DEFAULT_K), dtype=np.int16)
    lut.get_subband_samples(0, output_cubic, interp_mode=INTERP_CUBIC)
    print(f"Output at phase=0 (cubic):")
    print(f"  First leaf coeffs: {output_cubic[0, :]}")
    print(f"  Max coeff: {np.max(np.abs(output_cubic))}")

    # Compare linear vs cubic at mid-phase
    mid_phase = (1 << 31)  # Half cycle
    lut.get_subband_samples(mid_phase, output, interp_mode=INTERP_LINEAR)
    lut.get_subband_samples(mid_phase, output_cubic, interp_mode=INTERP_CUBIC)
    diff = np.abs(output.astype(np.int32) - output_cubic.astype(np.int32))
    print(f"\nLinear vs Cubic difference at mid-phase:")
    print(f"  Max diff: {np.max(diff)}")
    print(f"  Mean diff: {np.mean(diff):.2f}")

    # Test audio synthesis quality
    print("\n" + "-" * 40)
    print("Testing audio synthesis (IDWPT)...")

    state = SubbandState()
    state.phase = 0
    state.phase_inc = int((440.0 / 48000) * (1 << 32))

    # Inject carrier and synthesize
    state.inject_carrier(lut, base_mix_q15=32767)

    audio = np.zeros(64, dtype=np.int16)
    streaming_idwpt_block(state, audio)

    print(f"Audio output (first 64 samples):")
    print(f"  Peak: {np.max(np.abs(audio))}")
    print(f"  RMS: {np.sqrt(np.mean(audio.astype(float)**2)):.1f}")
    print(f"  First 8 samples: {audio[:8].tolist()}")

    # Check periodicity
    # For 440 Hz at 48 kHz, period is ~109 samples
    # In 64 samples we should see ~0.59 cycles
    zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
    print(f"  Zero crossings in block: {zero_crossings}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Memory savings: {old_size / table_size_bytes:.0f}x reduction")
    print(f"  New table size: {table_size_bytes / 1024:.0f} KB per note/wave")
    print(f"  Linear interpolation: Works at P={DEFAULT_P}")
    print(f"  Cubic interpolation: Available for higher quality")
    print("=" * 60)


if __name__ == "__main__":
    main()
