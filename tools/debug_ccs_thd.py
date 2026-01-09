"""Debug CCS THD Analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.fft import rfft, rfftfreq
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame


def test_lut_directly():
    """Test LUT coefficients and IDWPT directly, bypassing VM."""
    print("\n=== Direct LUT + IDWPT Test ===\n")

    from symcrash_python_ap.vm.coeff_lut_streaming import (
        get_lut_for_note, WAVE_SINE, CCSState, ccs_generate_coeffs_polyphase,
        NUM_LEAVES, VM_K,
    )
    from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

    SAMPLE_RATE = 48000
    NUM_BLOCKS = 100
    expected_freq = 261.63  # C4

    # Create CCS state
    ccs_state = CCSState()
    ccs_state.note = 60
    ccs_state.wave_id = WAVE_SINE
    # Calculate phase increment PER SAMPLE (advance_block multiplies by 64)
    ccs_state.phase_inc = int((expected_freq / SAMPLE_RATE) * (1 << 32))

    print(f"CCS state:")
    print(f"  note: {ccs_state.note}")
    print(f"  wave_id: {ccs_state.wave_id} (SINE)")
    print(f"  phase_inc: {ccs_state.phase_inc} (per-sample)")

    # Generate audio using the SIMPLE path (K=1, single IDWPT)
    print("\n--- Simple path: single IDWPT per block ---")
    lut = get_lut_for_note(WAVE_SINE, 60)
    print(f"LUT info: K={lut.K}, P={lut.P}, leaves={lut.num_leaves}")

    all_audio_simple = []
    ccs_state.phase = 0
    for block in range(NUM_BLOCKS):
        # Get coefficients for K=0 only (single frame)
        subband = np.zeros((NUM_LEAVES, lut.K), dtype=np.int16)
        lut.get_subband_samples(ccs_state.phase, subband)

        # Just use k=0
        coeffs = subband[:, 0].tolist()

        # IDWPT
        audio = idwpt_jit(coeffs, headroom_bits=1)
        all_audio_simple.append(audio)

        # Advance phase (multiply by 64 for one block)
        ccs_state.phase = (ccs_state.phase + ccs_state.phase_inc * 64) & 0xFFFFFFFF

    audio_simple = np.concatenate(all_audio_simple).astype(np.float64)
    audio_simple = audio_simple - np.mean(audio_simple)  # Remove DC

    # Analyze
    spectrum = rfft(audio_simple)
    freqs = rfftfreq(len(audio_simple), 1.0/SAMPLE_RATE)
    mag_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

    # Find peak > 50 Hz
    mask = freqs > 50
    spec_masked = np.abs(spectrum).copy()
    spec_masked[~mask] = 0
    peak_idx = np.argmax(spec_masked)

    print(f"\nSimple path results:")
    print(f"  RMS: {np.sqrt(np.mean(audio_simple**2)):.1f}")
    print(f"  Peak freq: {freqs[peak_idx]:.1f} Hz (expected {expected_freq:.1f})")
    print(f"  Peak magnitude: {mag_db[peak_idx]:.1f} dB")

    # Now test POLYPHASE path
    print("\n--- Polyphase path: K IDWPTs interleaved ---")
    all_audio_poly = []
    ccs_state.phase = 0

    for block in range(NUM_BLOCKS):
        # Generate polyphase coefficients
        carrier_512 = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
        ccs_generate_coeffs_polyphase(ccs_state, carrier_512)

        # Reshape to [64 leaves x K k-slots]
        coeff_2d = carrier_512.reshape(NUM_LEAVES, VM_K)

        # Run K IDWPTs
        idwpt_outputs = []
        for k in range(VM_K):
            coeffs_k = coeff_2d[:, k].tolist()
            audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
            idwpt_outputs.append(audio_k)

        # Interleave - ORIGINAL buggy version
        output = np.zeros(64, dtype=np.int16)
        for k in range(VM_K):
            y_k = idwpt_outputs[k]
            for m in range(64 // VM_K):
                output[k + m * VM_K] = y_k[m * VM_K]

        all_audio_poly.append(output)

        # Advance phase
        ccs_state.advance_block()

    audio_poly = np.concatenate(all_audio_poly).astype(np.float64)
    audio_poly = audio_poly - np.mean(audio_poly)

    # Analyze
    spectrum = rfft(audio_poly)
    freqs = rfftfreq(len(audio_poly), 1.0/SAMPLE_RATE)
    mag_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

    mask = freqs > 50
    spec_masked = np.abs(spectrum).copy()
    spec_masked[~mask] = 0
    peak_idx = np.argmax(spec_masked)

    print(f"\nPolyphase path results:")
    print(f"  RMS: {np.sqrt(np.mean(audio_poly**2)):.1f}")
    print(f"  Peak freq: {freqs[peak_idx]:.1f} Hz (expected {expected_freq:.1f})")
    print(f"  Peak magnitude: {mag_db[peak_idx]:.1f} dB")

    # Now test alternative: just use k=0 IDWPT output directly
    print("\n--- Alternative: just use k=0 IDWPT output directly ---")
    all_audio_k0 = []
    ccs_state.phase = 0

    for block in range(NUM_BLOCKS):
        carrier_512 = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
        ccs_generate_coeffs_polyphase(ccs_state, carrier_512)

        coeff_2d = carrier_512.reshape(NUM_LEAVES, VM_K)

        # Just use k=0
        coeffs = coeff_2d[:, 0].tolist()
        audio = idwpt_jit(coeffs, headroom_bits=1)

        all_audio_k0.append(audio)
        ccs_state.advance_block()

    audio_k0 = np.concatenate(all_audio_k0).astype(np.float64)
    audio_k0 = audio_k0 - np.mean(audio_k0)

    spectrum = rfft(audio_k0)
    freqs = rfftfreq(len(audio_k0), 1.0/SAMPLE_RATE)
    mag_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

    mask = freqs > 50
    spec_masked = np.abs(spectrum).copy()
    spec_masked[~mask] = 0
    peak_idx = np.argmax(spec_masked)

    print(f"\nk=0 only results:")
    print(f"  RMS: {np.sqrt(np.mean(audio_k0**2)):.1f}")
    print(f"  Peak freq: {freqs[peak_idx]:.1f} Hz (expected {expected_freq:.1f})")
    print(f"  Peak magnitude: {mag_db[peak_idx]:.1f} dB")

    # THD analysis for k=0 path
    fundamental_idx = np.argmin(np.abs(freqs - expected_freq))
    fundamental_power = np.abs(spectrum[fundamental_idx])**2
    harmonic_power = 0
    for h in range(2, 11):
        h_freq = expected_freq * h
        if h_freq > SAMPLE_RATE / 2:
            break
        h_idx = np.argmin(np.abs(freqs - h_freq))
        harmonic_power += np.abs(spectrum[h_idx])**2

    thd = 100 * np.sqrt(harmonic_power / (fundamental_power + 1e-10))
    print(f"  THD: {thd:.2f}%")

    # Check block boundary discontinuities in k=0 path
    print(f"\n  Block boundary check:")
    jumps = []
    for i in range(1, NUM_BLOCKS):
        idx = i * 64
        jump = abs(audio_k0[idx] - audio_k0[idx - 1])
        jumps.append(jump)
    print(f"    Max jump: {max(jumps):.0f}")
    print(f"    Avg jump: {np.mean(jumps):.0f}")


def main():
    print("=== CCS THD Debug ===\n")

    SAMPLE_RATE = 48000
    NUM_BLOCKS = 100

    # Create state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60, cents=0)

    print(f"Initial state:")
    print(f"  use_coeff_osc: {state.use_coeff_osc}")
    print(f"  ccs_state.phase_inc: {state.ccs_state.phase_inc}")
    print(f"  ccs_state.note: {state.ccs_state.note}")
    print(f"  coeff_state sum: {state.coeff_state.sum()}")

    # Create frame
    frame = VoiceFrame(
        voice_id=0,
        note=60,  # Middle C = 261.63 Hz
        cents=0,
        gain_q15=32767,
        beta_target_q15=32767,
    )

    # Decode first frame to see what's happening
    from symcrash_python_ap.vm.decode import decode_frame
    cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=0, prev_gain_q15=32767)
    print(f"\nDecoded config:")
    print(f"  use_coeff_osc: {cfg.use_coeff_osc}")
    print(f"  coeff_osc_params.phase_inc: {cfg.coeff_osc_params.phase_inc}")
    print(f"  coeff_osc_params.base_mix_q15: {cfg.coeff_osc_params.base_mix_q15}")
    print(f"  backbone_instrs: {[(i.op_id, i.strength_q15) for i in cfg.backbone_instrs]}")
    print(f"  event_instrs: {[(i.op_id, i.strength_q15) for i in cfg.event_instrs]}")
    print(f"  tail_ops: {cfg.tail_ops}")

    # Test lane_ccs_render directly
    print("\n--- Testing lane_ccs_render directly ---")
    from symcrash_python_ap.vm.coeff_lut_streaming import (
        ccs_generate_coeffs_polyphase, CCSState, NUM_LEAVES, VM_K, WAVE_SINE,
    )
    from symcrash_python_ap.vm.lanes import lane_ccs_render

    # Create clean state
    test_state = VMState.create(voice_id=99)
    test_state.reset_for_note_on(note=60, cents=0)

    # Generate carrier coefficients manually
    carrier_512 = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
    test_state.ccs_state.note = 60
    test_state.ccs_state.wave_id = WAVE_SINE
    test_state.ccs_state.phase_inc = cfg.coeff_osc_params.phase_inc

    ccs_generate_coeffs_polyphase(test_state.ccs_state, carrier_512)

    # Copy to coeff_state (this is what executor does)
    test_state.coeff_state[:NUM_LEAVES * VM_K] = carrier_512

    print(f"  carrier_512 stats: sum={carrier_512.sum()}, max={carrier_512.max()}, min={carrier_512.min()}")
    print(f"  coeff_state[:8]: {test_state.coeff_state[:8]}")

    # Now run lane_ccs_render
    output = lane_ccs_render(test_state, cfg)
    print(f"  lane_ccs_render output stats: sum={output.sum()}, max={output.max()}, min={output.min()}")
    print(f"  output[:16]: {output[:16]}")

    # Debug: manually run IDWPT with k=0 coefficients
    print("\n  Manual k=0 IDWPT test:")
    from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
    coeff_2d = test_state.coeff_state[:NUM_LEAVES * VM_K].reshape(NUM_LEAVES, VM_K)
    coeffs_k0 = coeff_2d[:, 0].tolist()
    print(f"    coeffs_k0[:8]: {coeffs_k0[:8]}")
    manual_audio = idwpt_jit(coeffs_k0, headroom_bits=1)
    print(f"    manual_audio[:16]: {manual_audio[:16]}")

    # Generate audio BYPASSING vm_step_block to isolate the issue
    print("\n--- Direct CCS generation (bypassing VM executor) ---")
    from symcrash_python_ap.vm.coeff_lut_streaming import ccs_generate_coeffs_polyphase
    from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

    test_ccs = CCSState()
    test_ccs.note = 60
    test_ccs.wave_id = WAVE_SINE
    test_ccs.phase_inc = cfg.coeff_osc_params.phase_inc
    test_ccs.phase = 0

    direct_audio = []
    for i in range(NUM_BLOCKS):
        carrier = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
        ccs_generate_coeffs_polyphase(test_ccs, carrier)

        # Extract k=0 and IDWPT
        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
        coeffs = coeff_2d[:, 0].tolist()
        audio = idwpt_jit(coeffs, headroom_bits=1)
        direct_audio.append(np.array(audio, dtype=np.int16))

        test_ccs.advance_block()

    direct_audio = np.concatenate(direct_audio).astype(np.float64)
    direct_audio = direct_audio - np.mean(direct_audio)

    # Check boundary jumps
    jumps = []
    for i in range(1, NUM_BLOCKS):
        idx = i * 64
        jump = abs(direct_audio[idx] - direct_audio[idx - 1])
        jumps.append(jump)
    print(f"  Direct CCS: Max jump={max(jumps):.0f}, Avg={np.mean(jumps):.0f}")

    # Now test through VM with step-by-step debugging
    print("\n--- VM executor path (step-by-step debugging for block 0) ---")

    # Manually trace block 0
    from symcrash_python_ap.vm.decode import decode_frame
    from symcrash_python_ap.vm.lanes import lane_inject, lane_tail, lane_ccs_render
    from symcrash_python_ap.fixed import RSHIFT_RNE

    debug_state = VMState.create(voice_id=99)
    debug_state.reset_for_note_on(note=60, cents=0)

    debug_cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=0, prev_gain_q15=32767)

    # Step 0: Generate carrier coefficients
    carrier = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
    debug_state.ccs_state.note = debug_cfg.coeff_osc_params.note
    debug_state.ccs_state.wave_id = debug_cfg.coeff_osc_params.table_id
    debug_state.ccs_state.phase_inc = debug_cfg.coeff_osc_params.phase_inc
    ccs_generate_coeffs_polyphase(debug_state.ccs_state, carrier)

    print(f"  After ccs_generate_coeffs_polyphase:")
    print(f"    carrier[:8]: {carrier[:8]}")

    # Step 1: Blend with coeff_state
    base_mix = debug_cfg.coeff_osc_params.base_mix_q15
    print(f"  base_mix_q15: {base_mix}")
    print(f"  coeff_state before blend [:8]: {debug_state.coeff_state[:8]}")

    for idx in range(NUM_LEAVES * VM_K):
        carrier_val = int(carrier[idx])
        old_val = int(debug_state.coeff_state[idx])
        inv_mix = 32767 - base_mix
        blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)
        debug_state.coeff_state[idx] = np.clip(blended, -32768, 32767)

    print(f"  coeff_state after blend [:8]: {debug_state.coeff_state[:8]}")

    # Step 2: lane_inject (no events)
    lane_inject(debug_state, debug_cfg)
    print(f"  coeff_state after lane_inject [:8]: {debug_state.coeff_state[:8]}")

    # Step 3: backbone ops (none)
    # Step 4: lane_tail
    lane_tail(debug_state, debug_cfg)
    print(f"  coeff_state after lane_tail [:8]: {debug_state.coeff_state[:8]}")

    # Step 5: lane_ccs_render
    debug_output = lane_ccs_render(debug_state, debug_cfg)
    print(f"  lane_ccs_render output[:8]: {debug_output[:8]}")

    # Compare to direct
    coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
    direct_coeffs = coeff_2d[:, 0].tolist()
    direct_audio = idwpt_jit(direct_coeffs, headroom_bits=1)
    print(f"  Direct IDWPT (from carrier) output[:8]: {direct_audio[:8]}")

    # Now run full blocks with detailed tracing
    print("\n--- VM executor path (with coeff tracing) ---")

    # Reset state for full test
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60, cents=0)

    # Generate audio
    all_audio = []
    for i in range(NUM_BLOCKS):
        old_phase = state.ccs_state.phase
        old_coeff_k0 = state.coeff_state[:8].copy()

        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_audio.append(output.copy())  # Must copy - lane_ccs_render uses global buffer

        new_coeff_k0 = state.coeff_state[:8]

        if i < 5:
            phase_deg = (old_phase / (2**32)) * 360
            # Generate expected coefficients for this phase
            from symcrash_python_ap.vm.coeff_lut_streaming import CCSState as CCS2, ccs_generate_coeffs_polyphase as ccs_gen2
            expected_ccs = CCS2()
            expected_ccs.note = 60
            expected_ccs.wave_id = WAVE_SINE
            expected_ccs.phase_inc = cfg.coeff_osc_params.phase_inc
            expected_ccs.phase = old_phase
            expected_carrier = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
            ccs_gen2(expected_ccs, expected_carrier)

            print(f"Block {i}: phase={phase_deg:.1f}deg")
            print(f"  Expected coeff[:8]: {expected_carrier[:8]}")
            print(f"  Before coeff[:8]: {old_coeff_k0}")
            print(f"  After coeff[:8]: {new_coeff_k0}")
            print(f"  Output: mean={output.mean():.0f}, range=[{output.min()}, {output.max()}]")

    audio = np.concatenate(all_audio).astype(np.float64)

    print(f"\nAudio stats:")
    print(f"  Total samples: {len(audio)}")
    print(f"  Range: [{audio.min():.0f}, {audio.max():.0f}]")
    print(f"  Mean (DC): {audio.mean():.1f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.1f}")

    # FFT analysis - skip initial blocks (latency warmup)
    skip_samples = 64 * 10  # Skip first 10 blocks
    audio_analysis = audio[skip_samples:]

    # Remove DC for analysis
    audio_analysis = audio_analysis - np.mean(audio_analysis)

    print(f"\nAnalysis segment (after DC removal):")
    print(f"  Samples: {len(audio_analysis)}")
    print(f"  Mean: {audio_analysis.mean():.3f}")
    print(f"  RMS: {np.sqrt(np.mean(audio_analysis**2)):.1f}")

    # FFT
    spectrum = rfft(audio_analysis)
    freqs = rfftfreq(len(audio_analysis), 1.0/SAMPLE_RATE)
    mag_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

    # Find peak (excluding DC)
    dc_mask = freqs > 50
    spectrum_no_dc = np.abs(spectrum).copy()
    spectrum_no_dc[~dc_mask] = 0
    peak_idx = np.argmax(spectrum_no_dc)
    peak_freq = freqs[peak_idx]
    peak_db = mag_db[peak_idx]

    expected = 261.63

    print(f"\nFrequency Analysis:")
    print(f"  Expected: {expected:.2f} Hz")
    print(f"  Found peak: {peak_freq:.2f} Hz (error: {abs(peak_freq-expected):.2f} Hz)")
    print(f"  Peak magnitude: {peak_db:.1f} dB")

    # THD calculation
    fundamental_power = np.abs(spectrum[peak_idx])**2
    harmonic_power = 0

    print(f"\nHarmonics:")
    for h in range(1, 11):
        h_freq = expected * h
        if h_freq > SAMPLE_RATE / 2:
            break
        h_idx = np.argmin(np.abs(freqs - h_freq))
        h_db = mag_db[h_idx]
        rel_db = h_db - peak_db

        if h > 1:
            harmonic_power += np.abs(spectrum[h_idx])**2

        marker = " <- FUNDAMENTAL" if h == 1 else ""
        print(f"  H{h:2d} ({h_freq:7.1f} Hz): {h_db:6.1f} dB ({rel_db:+6.1f} rel){marker}")

    thd = 100 * np.sqrt(harmonic_power / (fundamental_power + 1e-10))
    print(f"\nTHD (raw FFT, includes spectral leakage): {thd:.2f}%")

    # Windowed THD for accurate measurement
    from scipy.signal import windows
    audio_windowed = audio_analysis * windows.hann(len(audio_analysis)) * 2
    spectrum_w = rfft(audio_windowed)
    freqs_w = rfftfreq(len(audio_windowed), 1.0/SAMPLE_RATE)

    # Find peak in windowed spectrum
    mask_w = freqs_w > 50
    spec_w_masked = np.abs(spectrum_w).copy()
    spec_w_masked[~mask_w] = 0
    peak_w_idx = np.argmax(spec_w_masked)

    fund_power_w = np.abs(spectrum_w[peak_w_idx])**2
    harm_power_w = 0
    for h in range(2, 11):
        h_freq = expected * h
        if h_freq > SAMPLE_RATE / 2:
            break
        h_idx = np.argmin(np.abs(freqs_w - h_freq))
        harm_power_w += np.abs(spectrum_w[h_idx])**2

    thd_windowed = 100 * np.sqrt(harm_power_w / (fund_power_w + 1e-10))
    print(f"THD (Hann windowed, accurate): {thd_windowed:.4f}%")

    # Check for 750 Hz comb spacing artifacts
    print(f"\nComb artifact check (750 Hz spacing):")
    for mult in range(1, 8):
        comb_freq = 750.0 * mult
        if comb_freq > 5000:
            break
        idx = np.argmin(np.abs(freqs - comb_freq))
        comb_db = mag_db[idx]
        rel = comb_db - peak_db
        status = "WARN" if rel > -40 else "OK"
        print(f"  {comb_freq:.0f} Hz: {comb_db:.1f} dB ({rel:+.1f} rel) [{status}]")

    # Check discontinuities
    print(f"\nBlock boundary check:")
    jumps = []
    for i in range(10, NUM_BLOCKS):  # Skip initial blocks
        idx = i * 64
        jump = abs(audio[idx] - audio[idx - 1])
        jumps.append(jump)

    print(f"  Max jump: {max(jumps):.0f}")
    print(f"  Avg jump: {np.mean(jumps):.0f}")

    if max(jumps) < 500:
        print("  [OK] Smooth boundaries")
    elif max(jumps) < 2000:
        print("  [ACCEPTABLE] Minor discontinuities")
    else:
        print("  [WARN] Large discontinuities!")


if __name__ == "__main__":
    test_lut_directly()
    print("\n" + "=" * 60)
    main()
