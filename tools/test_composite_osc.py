"""
Test Composite A+B Oscillator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.vm.composite_osc import (
    CompositeOscState, HalfPhaseLUT,
    composite_generate_coeffs, print_memory_budget,
    BLEND_PURE_A, BLEND_PURE_B, BLEND_EQUAL,
    COEFF_LEN_POLYPHASE,
)
from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, WAVE_SAW, WAVE_SINE, _NOTE_LUT_CACHE,
    ccs_generate_coeffs_polyphase,
)
from symcrash_python_ap.vm.harmonic_bank import BasisFrameAtlas


def test_memory_budget():
    """Print memory budget for FPGA deployment."""
    print("\n")
    print_memory_budget(num_waves=4, num_voices=32, num_anchors=8)


def test_half_phase_lut():
    """Test half-phase compression."""
    print("\n" + "=" * 60)
    print("Half-Phase LUT Compression Test")
    print("=" * 60)

    # Clear cache
    _NOTE_LUT_CACHE.clear()

    # Get full LUT for sine
    full_lut = get_lut_for_note(WAVE_SINE, 69)
    full_size = full_lut.data.nbytes

    # Create half-phase version (16-bit)
    half_lut = HalfPhaseLUT.from_full_lut(full_lut, use_8bit=False)
    half_size = half_lut.data.nbytes

    # Create 8-bit half-phase version
    compact_lut = HalfPhaseLUT.from_full_lut(full_lut, use_8bit=True)
    compact_size = compact_lut.data.nbytes

    print(f"\nSine wave (A4 = 440 Hz):")
    print(f"  Full LUT:         {full_size:>8,} bytes")
    print(f"  Half-phase:       {half_size:>8,} bytes ({full_size/half_size:.1f}× reduction)")
    print(f"  Half + 8-bit:     {compact_size:>8,} bytes ({full_size/compact_size:.1f}× reduction)")

    # Test reconstruction accuracy
    print("\nReconstruction accuracy test...")
    output_full = np.zeros((64, 8), dtype=np.int16)
    output_half = np.zeros((64, 8), dtype=np.int16)
    output_compact = np.zeros((64, 8), dtype=np.int16)

    # Test multiple phase positions
    test_phases = [0, 1 << 30, 1 << 31, (3 << 30)]  # 0°, 90°, 180°, 270°
    phase_names = ["0°", "90°", "180°", "270°"]

    for phase, name in zip(test_phases, phase_names):
        full_lut.get_subband_samples(phase, output_full)
        half_lut.get_subband_samples(phase, output_half)
        compact_lut.get_subband_samples(phase, output_compact)

        diff_half = np.abs(output_full.astype(np.int32) - output_half.astype(np.int32))
        diff_compact = np.abs(output_full.astype(np.int32) - output_compact.astype(np.int32))

        print(f"  Phase {name}: half max diff = {np.max(diff_half)}, "
              f"compact max diff = {np.max(diff_compact)}")


def test_composite_blending():
    """Test A+B blending."""
    print("\n" + "=" * 60)
    print("Composite A+B Blending Test")
    print("=" * 60)

    # Create composite state
    state = CompositeOscState()
    phase_inc = int((440.0 / 48000) * (1 << 32))
    state.reset(note=69, phase_inc=phase_inc, wave_id=WAVE_SAW)

    # Create basis atlas for Mode B
    atlas = BasisFrameAtlas()

    # Output buffers
    output_a = np.zeros(COEFF_LEN_POLYPHASE, dtype=np.int16)
    output_b = np.zeros(COEFF_LEN_POLYPHASE, dtype=np.int16)
    output_blend = np.zeros(COEFF_LEN_POLYPHASE, dtype=np.int16)

    # Test pure Mode A
    state.set_blend(BLEND_PURE_A)
    composite_generate_coeffs(state, output_a, atlas)
    print(f"\nPure Mode A: max={np.max(np.abs(output_a))}, rms={np.sqrt(np.mean(output_a.astype(float)**2)):.1f}")

    # Test pure Mode B
    state.set_blend(BLEND_PURE_B)
    state.mode_b.weights.set_saw(H=16)
    composite_generate_coeffs(state, output_b, atlas)
    print(f"Pure Mode B: max={np.max(np.abs(output_b))}, rms={np.sqrt(np.mean(output_b.astype(float)**2)):.1f}")

    # Test 50/50 blend
    state.set_blend(BLEND_EQUAL)
    composite_generate_coeffs(state, output_blend, atlas)
    print(f"50/50 Blend: max={np.max(np.abs(output_blend))}, rms={np.sqrt(np.mean(output_blend.astype(float)**2)):.1f}")

    # Verify blend is between A and B
    # For each coefficient, blended should be approximately (A + B) / 2
    expected = (output_a.astype(np.int32) + output_b.astype(np.int32)) // 2
    diff = np.abs(output_blend.astype(np.int32) - expected)
    print(f"\nBlend accuracy (vs (A+B)/2): max diff = {np.max(diff)}, mean diff = {np.mean(diff):.2f}")


def test_blend_slew():
    """Test smooth blend transitions."""
    print("\n" + "=" * 60)
    print("Blend Slew Test")
    print("=" * 60)

    state = CompositeOscState()
    phase_inc = int((440.0 / 48000) * (1 << 32))
    state.reset(note=69, phase_inc=phase_inc, wave_id=WAVE_SAW)

    # Start at pure A, slew to pure B over 8 blocks
    state.set_blend(BLEND_PURE_A)
    state.set_blend(BLEND_PURE_B, slew_blocks=8)

    print(f"\nSlewing from A to B over 8 blocks:")
    for block in range(10):
        print(f"  Block {block}: blend = {state.blend_q15} ({100*state.blend_q15/32767:.1f}%)")
        state.advance_block()


def main():
    test_memory_budget()
    test_half_phase_lut()
    test_composite_blending()
    test_blend_slew()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
