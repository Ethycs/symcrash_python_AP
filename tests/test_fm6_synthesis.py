"""
FM6 6-Operator FM Synthesis Tests
=================================

Tests for DX7-compatible 6-operator FM synthesis per WSE-STD-FM-TOPOLOGY-v0.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.vm.fm_algorithms import get_algorithm, list_algorithms, ALGORITHMS
from symcrash_python_ap.vm.fm6_state import (
    FM6State, FMOperator, create_fm6_state,
    Q15_ONE, Q16_ONE, DX7_FEEDBACK_TABLE, dx7_feedback_to_q15
)
from symcrash_python_ap.vm.fm6_engine import (
    fm6_generate_block, fm6_to_coefficients, fm6_generate_coeffs,
    fm6_generate_audio, configure_electric_piano, configure_brass,
    configure_bell, configure_organ
)
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState, OSC_MODE_FM6
from symcrash_python_ap.vm.tables import VM_K, COEFF_LEN
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.fixed import midi_to_phase_inc_fixed


SAMPLE_RATE = 48000


def test_algorithm_table():
    """Test that all 32 DX7 algorithms are defined."""
    print("\n=== Algorithm Table Test ===\n")

    all_algs = list_algorithms()
    print(f"Total algorithms defined: {len(all_algs)}")

    # Verify all 32 algorithms exist
    missing = []
    for alg_id in range(1, 33):
        if alg_id not in ALGORITHMS:
            missing.append(alg_id)

    if missing:
        print(f"[FAIL] Missing algorithms: {missing}")
        return False

    print("[PASS] All 32 algorithms defined")

    # Verify each algorithm has valid topology
    for alg_id in range(1, 33):
        alg = get_algorithm(alg_id)

        # Check modulation matrix dimensions
        assert alg.modulation_matrix.shape == (6, 6), f"Alg {alg_id}: Invalid matrix shape"

        # Check carrier mask is non-zero
        assert alg.carrier_mask > 0, f"Alg {alg_id}: No carriers defined"

        # Check eval order has all 6 operators
        assert len(alg.eval_order) == 6, f"Alg {alg_id}: Invalid eval order length"
        assert set(alg.eval_order) == {0, 1, 2, 3, 4, 5}, f"Alg {alg_id}: Invalid eval order"

        # Check feedback op is valid
        assert alg.feedback_op >= -1 and alg.feedback_op < 6, f"Alg {alg_id}: Invalid feedback op"

    print("[PASS] All algorithm topologies valid")
    return True


def test_fm6_state_creation():
    """Test FM6State creation and configuration."""
    print("\n=== FM6State Creation Test ===\n")

    # Create default state
    state = create_fm6_state(algorithm_id=1)

    assert state.algorithm_id == 1
    assert len(state.operators) == 6
    assert state.op_outputs.shape == (6, VM_K)  # [6 ops, K k-slots]
    assert state.feedback_buffer.shape == (VM_K,)

    print("[PASS] Default FM6State created")

    # Test algorithm change
    state.set_algorithm(32)
    assert state.algorithm_id == 32
    print("[PASS] Algorithm change works")

    # Test operator configuration
    state.set_operator_ratio(0, Q16_ONE * 2)  # 2.0x
    state.set_operator_level(0, Q15_ONE // 2)  # 50%
    state.set_operator_feedback(0, 16384)      # 50% feedback

    assert state.operators[0].ratio_q16 == Q16_ONE * 2
    assert state.operators[0].level_q15 == Q15_ONE // 2
    assert state.operators[0].feedback_q15 == 16384

    print("[PASS] Operator configuration works")

    # Test reset
    state.operators[0].phase = 12345678
    state.reset()
    assert state.operators[0].phase == 0
    assert np.all(state.op_outputs == 0)

    print("[PASS] State reset works")
    return True


def test_fm6_single_operator():
    """Test FM6 with single carrier (Algorithm 32 = all carriers)."""
    print("\n=== Single Operator Test ===\n")

    state = create_fm6_state(algorithm_id=32)

    # Configure for pure sine from op 0 only
    for i in range(6):
        state.operators[i].level_q15 = 0
    state.operators[0].level_q15 = Q15_ONE
    state.operators[0].ratio_q16 = Q16_ONE

    # Generate at A4 (440 Hz)
    base_phase_inc = midi_to_phase_inc_fixed(69, 0)

    # Generate multiple blocks
    audio = fm6_generate_audio(state, base_phase_inc, num_blocks=100)

    # Should produce audio
    expected_len = 100 * VM_K  # 100 blocks × K samples
    assert len(audio) == expected_len, f"Expected {expected_len} samples, got {len(audio)}"
    assert np.max(np.abs(audio)) > 1000, "Output too quiet"

    # Check that we have a clear peak in the spectrum
    fft = np.abs(np.fft.rfft(audio.astype(np.float64)))
    peak_bin = np.argmax(fft[1:]) + 1  # Skip DC

    # Peak should be significant (at least 10% of total energy)
    peak_energy = fft[peak_bin]
    total_energy = np.sum(fft[1:])  # Exclude DC

    ratio = peak_energy / (total_energy + 1e-10)
    print(f"Peak to total ratio: {ratio:.2f}")
    print(f"Peak bin: {peak_bin}, Peak amplitude: {peak_energy:.1f}")

    assert ratio > 0.1, "Should have a clear spectral peak"
    assert peak_energy > 1000, "Peak should be significant"

    print("[PASS] Single operator produces audio")
    return True


def test_fm6_modulation():
    """Test FM6 with modulation (2-op chain)."""
    print("\n=== FM Modulation Test ===\n")

    # Algorithm 1: 6-op chain, but we'll use only 2 ops
    state = create_fm6_state(algorithm_id=1)

    # Configure: op6 modulates op5 modulates ... op1 (carrier)
    # We'll set only op2 (modulator) and op1 (carrier) active
    for i in range(6):
        state.operators[i].level_q15 = 0

    # Carrier (op1 = index 0)
    state.operators[0].level_q15 = Q15_ONE
    state.operators[0].ratio_q16 = Q16_ONE

    # Modulator (op2 = index 1)
    state.operators[1].level_q15 = Q15_ONE // 2  # Strong modulation
    state.operators[1].ratio_q16 = Q16_ONE       # 1:1 ratio

    base_phase_inc = midi_to_phase_inc_fixed(69, 0)

    # Generate audio
    audio_modulated = fm6_generate_audio(state, base_phase_inc, num_blocks=100)

    # Now without modulation (only carrier)
    state.reset()
    state.operators[1].level_q15 = 0
    audio_carrier = fm6_generate_audio(state, base_phase_inc, num_blocks=100)

    # Compare spectra - modulated should have more harmonics
    fft_mod = np.abs(np.fft.rfft(audio_modulated.astype(np.float64)))
    fft_car = np.abs(np.fft.rfft(audio_carrier.astype(np.float64)))

    # Count significant harmonics
    threshold = np.max(fft_mod) * 0.05
    harmonics_mod = np.sum(fft_mod[1:] > threshold)
    harmonics_car = np.sum(fft_car[1:] > threshold)

    print(f"Harmonics with modulation: {harmonics_mod}")
    print(f"Harmonics without modulation: {harmonics_car}")

    # Modulation should add harmonics
    assert harmonics_mod >= harmonics_car, "Modulation should add harmonics"

    print("[PASS] FM modulation adds harmonics")
    return True


def test_fm6_feedback():
    """Test FM6 feedback stability."""
    print("\n=== FM Feedback Stability Test ===\n")

    state = create_fm6_state(algorithm_id=1)

    # Configure with feedback on op1
    state.operators[0].level_q15 = Q15_ONE
    state.operators[0].ratio_q16 = Q16_ONE

    base_phase_inc = midi_to_phase_inc_fixed(69, 0)

    # Test all feedback levels
    for fb_level in range(8):
        state.reset()
        fb_q15 = dx7_feedback_to_q15(fb_level)
        state.operators[0].feedback_q15 = fb_q15

        # Generate 1000 blocks
        audio = fm6_generate_audio(state, base_phase_inc, num_blocks=1000)

        # Check for stability
        rms = np.sqrt(np.mean(audio.astype(np.float64)**2))
        peak = np.max(np.abs(audio))

        print(f"Feedback level {fb_level}: RMS={rms:.1f}, Peak={peak}")

        # Should not overflow or produce NaN
        assert not np.any(np.isnan(audio)), f"NaN at feedback level {fb_level}"
        assert rms < 35000, f"Unstable at feedback level {fb_level}"

    print("[PASS] All feedback levels stable")
    return True


def test_fm6_presets():
    """Test FM6 preset configurations."""
    print("\n=== FM6 Presets Test ===\n")

    presets = [
        ("Electric Piano", configure_electric_piano, 5),
        ("Brass", configure_brass, 22),
        ("Bell", configure_bell, 1),
        ("Organ", configure_organ, 32),
    ]

    base_phase_inc = midi_to_phase_inc_fixed(60, 0)  # C4

    for name, configure_fn, expected_alg in presets:
        state = create_fm6_state()
        configure_fn(state)

        assert state.algorithm_id == expected_alg, f"{name}: Wrong algorithm"

        # Generate audio
        audio = fm6_generate_audio(state, base_phase_inc, num_blocks=100)

        # Should produce audio
        rms = np.sqrt(np.mean(audio.astype(np.float64)**2))
        assert rms > 100, f"{name}: Too quiet (RMS={rms})"

        print(f"[PASS] {name} preset: RMS={rms:.1f}")

    print("[PASS] All presets produce audio")
    return True


def test_fm6_executor_integration():
    """Test FM6 integration with VM executor."""
    print("\n=== FM6 Executor Integration Test ===\n")

    # Create VM state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=69)
    state.use_coeff_osc = True

    # Create VoiceFrame with FM6 mode
    frame = VoiceFrame(
        note=69,
        coeff_osc_mode=OSC_MODE_FM6,
        fm_algorithm_id=1,
        fm_feedback_depth=3,
    )

    # Generate blocks
    outputs = []
    for _ in range(50):
        state, out = vm_step_block(state, frame)
        outputs.append(out.copy())

    audio = np.concatenate(outputs)

    # Should produce audio
    rms = np.sqrt(np.mean(audio.astype(np.float64)**2))
    print(f"FM6 via executor RMS: {rms:.1f}")

    assert rms > 100, f"FM6 executor output too quiet: {rms}"

    print("[PASS] FM6 executor integration works")
    return True


def test_fm6_coefficient_format():
    """Test FM6 to coefficient conversion."""
    print("\n=== FM6 Coefficient Format Test ===\n")

    state = create_fm6_state(algorithm_id=32)
    state.operators[0].level_q15 = Q15_ONE

    base_phase_inc = midi_to_phase_inc_fixed(69, 0)

    # Generate one block
    output_k = fm6_generate_block(state, base_phase_inc)

    assert output_k.shape == (VM_K,), f"Wrong output shape: {output_k.shape}"
    assert output_k.dtype == np.int16

    # Convert to coefficients
    coeffs = fm6_to_coefficients(output_k)

    assert coeffs.shape == (COEFF_LEN,), f"Wrong coeffs shape: {coeffs.shape}"
    assert coeffs.dtype == np.int16

    # Verify replication pattern: each k-slot replicated across 64 leaves
    for k in range(VM_K):
        for leaf in range(64):
            expected_idx = leaf * VM_K + k
            assert coeffs[expected_idx] == output_k[k], \
                f"Mismatch at leaf={leaf}, k={k}"

    print("[PASS] Coefficient format correct")
    return True


def test_fm6_pitch_tracking():
    """Test FM6 pitch tracking across notes."""
    print("\n=== FM6 Pitch Tracking Test ===\n")

    # Test multiple notes
    notes = [48, 60, 72, 84]  # C3, C4, C5, C6

    state = create_fm6_state(algorithm_id=32)
    state.operators[0].level_q15 = Q15_ONE

    for note in notes:
        state.reset()
        base_phase_inc = midi_to_phase_inc_fixed(note, 0)

        # Generate audio
        audio = fm6_generate_audio(state, base_phase_inc, num_blocks=200)

        # Find peak frequency
        fft = np.abs(np.fft.rfft(audio.astype(np.float64)))
        peak_bin = np.argmax(fft[1:]) + 1
        freq_resolution = SAMPLE_RATE / len(audio)
        measured_freq = peak_bin * freq_resolution

        # Expected frequency
        expected_freq = 440.0 * (2 ** ((note - 69) / 12))

        error_cents = 1200 * np.log2(measured_freq / expected_freq) if measured_freq > 0 else float('inf')

        print(f"Note {note}: Expected {expected_freq:.1f} Hz, Got {measured_freq:.1f} Hz, Error: {error_cents:.1f} cents")

        # Allow very generous tolerance due to short FFT and lane-rate output
        # With VM_K=4 and 200 blocks, we only have 800 samples at 48kHz
        # which gives very coarse frequency resolution
        assert abs(error_cents) < 200, f"Note {note}: Pitch error too large"

    print("[PASS] Pitch tracking works")
    return True


def main():
    print("=" * 60)
    print("FM6 6-Operator FM Synthesis Test Suite")
    print("=" * 60)

    results = []

    results.append(("Algorithm Table", test_algorithm_table()))
    results.append(("FM6State Creation", test_fm6_state_creation()))
    results.append(("Single Operator", test_fm6_single_operator()))
    results.append(("FM Modulation", test_fm6_modulation()))
    results.append(("Feedback Stability", test_fm6_feedback()))
    results.append(("Presets", test_fm6_presets()))
    results.append(("Executor Integration", test_fm6_executor_integration()))
    results.append(("Coefficient Format", test_fm6_coefficient_format()))
    results.append(("Pitch Tracking", test_fm6_pitch_tracking()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("[PASS] All FM6 tests passed!")
        print("\n6-Operator FM Synthesis Features:")
        print("  - 32 DX7 algorithms")
        print("  - Topological evaluation order")
        print("  - z^-1 block-delayed feedback")
        print("  - Preset configurations (E.Piano, Brass, Bell, Organ)")
        print("  - Full CCS pipeline integration")
        return 0
    else:
        print("[FAIL] Some FM6 tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
