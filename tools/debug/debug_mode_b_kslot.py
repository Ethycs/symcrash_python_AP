"""
Debug Mode B k-slot interleaving.

The 512-element carrier array is [leaf0_k0, leaf0_k1, ..., leaf0_k7, leaf1_k0, ...].
Let's verify IDWPT reconstruction handles this correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, ModeBState,
    BLOCK_SIZE, SAMPLE_RATE, VM_K, NUM_LEAVES
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

def test_kslot_reconstruction():
    """Test how to properly reconstruct from 512 coefficients."""
    print("=" * 60)
    print("TEST: K-slot Reconstruction Methods")
    print("=" * 60)

    note = 69
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    basis = generate_basis_frames(anchor_note=note, H_max=8)
    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_sine()
    mb_state.H_active = 1

    # Generate coefficients
    carrier_512 = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(mb_state, basis, carrier_512)

    print(f"\nGenerated 512 coefficients for 440 Hz sine")
    print(f"Layout: [leaf0_k0, leaf0_k1, ..., leaf0_k7, leaf1_k0, ...]")

    # Check layout - should be interleaved by k-slot
    print(f"\nFirst 16 coefficients (leaf 0-1, all k):")
    for i in range(16):
        leaf = i // VM_K
        k = i % VM_K
        print(f"  idx={i:2d} leaf={leaf} k={k}: {carrier_512[i]:6d}")

    # Method 1: Reconstruct each k-slot separately, then interleave
    print(f"\n=== Method 1: Per-k-slot IDWPT ===")
    method1_samples = []
    for k in range(VM_K):
        # Extract k-slot (every 8th starting at k)
        coeffs_64 = carrier_512[k::VM_K].tolist()
        samples_8 = idwpt_jit(coeffs_64)
        method1_samples.append(samples_8)

    # Interleave: k0[0], k1[0], ..., k7[0], k0[1], k1[1], ...
    # Wait - that gives 64*8 = 512 samples, but we want 64 samples per block
    # The k-slots should produce DIFFERENT time positions within the block

    # Actually, per WSE-STD, each k-slot produces 64/K = 8 samples
    # The 64 samples of the block come from: [k0[0..7], k1[0..7], ..., k7[0..7]]
    # So we should have 8 samples per k-slot? Let me check IDWPT output size
    print(f"IDWPT output size: {len(method1_samples[0])} samples")

    # IDWPT of 64 coeffs gives 64 samples. The k-slots represent POLYPHASE components.
    # The correct interpretation: each k-slot reconstructs a DIFFERENT PHASE of the block.
    # To get 64 samples, we take 8 from each k-slot: samples[k*8:(k+1)*8] from k-slot k

    block_samples = []
    for k in range(VM_K):
        coeffs_64 = carrier_512[k::VM_K].tolist()
        samples_64 = idwpt_jit(coeffs_64)
        # Take samples k*8 to (k+1)*8 from this k-slot's reconstruction
        block_samples.extend(samples_64[k*8:(k+1)*8])

    print(f"Block samples (64): {len(block_samples)}")

    # FFT
    samples_arr = np.array(block_samples, dtype=np.float64)
    # Pad to get better frequency resolution
    padded = np.zeros(1024)
    padded[:len(samples_arr)] = samples_arr
    spectrum = np.abs(rfft(padded))
    freqs = rfftfreq(len(padded), 1/SAMPLE_RATE)

    peak_idx = np.argmax(spectrum[1:100]) + 1
    peak_freq = freqs[peak_idx]
    print(f"Method 1 peak freq: {peak_freq:.1f} Hz")

    # Method 2: Original approach - just take k=0 reconstruction as representative
    print(f"\n=== Method 2: k=0 only ===")
    coeffs_k0 = carrier_512[0::VM_K].tolist()
    samples_k0 = idwpt_jit(coeffs_k0)

    samples_arr2 = np.array(samples_k0, dtype=np.float64)
    padded2 = np.zeros(1024)
    padded2[:len(samples_arr2)] = samples_arr2
    spectrum2 = np.abs(rfft(padded2))
    peak_idx2 = np.argmax(spectrum2[1:100]) + 1
    peak_freq2 = freqs[peak_idx2]
    print(f"Method 2 (k=0 only) peak freq: {peak_freq2:.1f} Hz")

    # Method 3: Interleave all k-slots sample by sample
    print(f"\n=== Method 3: Sample interleave ===")
    all_kslot_samples = []
    for k in range(VM_K):
        coeffs_k = carrier_512[k::VM_K].tolist()
        samples_k = idwpt_jit(coeffs_k)
        all_kslot_samples.append(samples_k)

    # Per WSE-STD-121, polyphase reconstruction:
    # Block sample n uses k-slot k = n mod K, sample index n // K
    interleaved = []
    for n in range(BLOCK_SIZE):
        k = n % VM_K
        idx = n // VM_K
        interleaved.append(all_kslot_samples[k][idx])

    samples_arr3 = np.array(interleaved, dtype=np.float64)
    padded3 = np.zeros(1024)
    padded3[:len(samples_arr3)] = samples_arr3
    spectrum3 = np.abs(rfft(padded3))
    peak_idx3 = np.argmax(spectrum3[1:100]) + 1
    peak_freq3 = freqs[peak_idx3]
    print(f"Method 3 (interleaved) peak freq: {peak_freq3:.1f} Hz")

    # Method 4: Generate multiple blocks, concatenate, analyze
    print(f"\n=== Method 4: Multi-block ===")
    mb_state2 = ModeBState()
    mb_state2.reset(note=note, phase_inc=phase_inc)
    mb_state2.weights.set_sine()
    mb_state2.H_active = 1

    all_samples = []
    for _ in range(50):
        carrier_512 = np.zeros(512, dtype=np.int16)
        mode_b_generate_coeffs(mb_state2, basis, carrier_512)

        # Use method 3 interleaving
        kslot_samples = []
        for k in range(VM_K):
            coeffs_k = carrier_512[k::VM_K].tolist()
            samples_k = idwpt_jit(coeffs_k)
            kslot_samples.append(samples_k)

        block = []
        for n in range(BLOCK_SIZE):
            k = n % VM_K
            idx = n // VM_K
            block.append(kslot_samples[k][idx])

        all_samples.extend(block)
        mb_state2.advance_block()

    samples_arr4 = np.array(all_samples, dtype=np.float64)
    spectrum4 = np.abs(rfft(samples_arr4))
    freqs4 = rfftfreq(len(samples_arr4), 1/SAMPLE_RATE)

    # Find peaks
    threshold = np.max(spectrum4) * 0.1
    peaks = []
    for i in range(1, len(spectrum4)-1):
        if spectrum4[i] > threshold and spectrum4[i] > spectrum4[i-1] and spectrum4[i] > spectrum4[i+1]:
            peaks.append((freqs4[i], spectrum4[i]))

    peaks.sort(key=lambda x: -x[1])
    print("Top frequencies (multi-block):")
    for freq, amp in peaks[:5]:
        print(f"  {freq:.1f} Hz (amp: {amp:.0f})")


def test_compare_with_lane_ccs():
    """Compare Mode B output with how lane_ccs_render does it."""
    print("\n" + "=" * 60)
    print("TEST: Compare with lane_ccs_render")
    print("=" * 60)

    # Look at how lane_ccs_render actually does IDWPT
    from symcrash_python_ap.vm.lanes import lane_ccs_render, lane_idwpt_render
    from symcrash_python_ap.vm.state import VMState
    from symcrash_python_ap.vm.decode import decode_frame, DecodedPlan
    from symcrash_python_ap.frames import VoiceFrame
    from symcrash_python_ap.vm.tables import DEFAULT_TABLES

    # Create a voice state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=69)

    # Create frame for pure sine
    frame = VoiceFrame(
        note=69,
        coeff_osc_mode=1,  # Mode B
        H_request=1,
        gain_q15=32767,
        beta_target_q15=0,
        waveform_type=0,  # Sine
    )

    # Decode frame
    cfg = decode_frame(frame, DEFAULT_TABLES, 0, 0)

    # Now manually generate Mode B coefficients
    from symcrash_python_ap.vm.harmonic_bank import get_basis_for_note, mode_b_generate_coeffs

    note = 69
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    # Initialize mode_b_state
    state.mode_b_state.reset(note=note, phase_inc=phase_inc)
    state.mode_b_state.weights.set_sine()
    state.mode_b_state.H_active = 1

    basis = get_basis_for_note(note)

    # Generate coefficients
    carrier_512 = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(state.mode_b_state, basis, carrier_512)

    # Copy to state.coeff_state
    state.coeff_state[:] = carrier_512

    # Now call lane_ccs_render
    cfg.eta_curr_q15 = 32767  # Full wet
    cfg.eta_prev_q15 = 32767
    cfg.idwpt_enable = True

    output = lane_ccs_render(state, cfg)

    print(f"Output from lane_ccs_render: {len(output)} samples")
    print(f"First 10 samples: {output[:10].tolist()}")

    # Analyze output
    samples_arr = np.array(output, dtype=np.float64)
    padded = np.zeros(1024)
    padded[:len(samples_arr)] = samples_arr
    spectrum = np.abs(rfft(padded))
    freqs = rfftfreq(len(padded), 1/SAMPLE_RATE)

    peak_idx = np.argmax(spectrum[1:100]) + 1
    peak_freq = freqs[peak_idx]
    print(f"lane_ccs_render peak freq: {peak_freq:.1f} Hz")


if __name__ == "__main__":
    test_kslot_reconstruction()
    test_compare_with_lane_ccs()
