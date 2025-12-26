"""
Simple AntiClick Test - Isolated
================================
Test anticlick with a simple sine carrier to verify no discontinuities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
from symcrash_python_ap.safety.anti_click import AntiClickOperator, AntiClickConfig


def test_pure_dry():
    """Test pure dry mode (beta=0) - should pass carrier through directly."""
    print("\n[1] PURE DRY (beta=0)")
    print("-" * 40)

    cfg = AntiClickConfig(block_size=64, wet_latency_blocks=5, ola_K=16)
    anticlick = AntiClickOperator(cfg)
    state = anticlick.init_state()

    anticlick.note_on(state)
    anticlick.set_wet_target(state, 0.0)  # Pure dry

    samples = []
    phase = 0.0
    freq = 261.63
    sample_rate = 48000

    for block in range(50):
        # Generate sine carrier with continuous phase
        carrier = np.zeros(64, dtype=np.float32)
        for i in range(64):
            carrier[i] = np.sin(phase) * 0.5
            phase += 2 * np.pi * freq / sample_rate

        output = anticlick.step(state, carrier, None)
        samples.extend(output.tolist())

    arr = np.array(samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]
    print(f"Total samples: {len(arr)}")
    print(f"Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")
    print(f"Threshold (10x mean): {threshold:.4f}")
    print(f"Spikes: {len(spikes)}")

    # Check block boundaries
    for block in range(1, 15):
        idx = block * 64 - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.4f}")

    return len(spikes)


def test_wet_transition():
    """Test wet transition (beta 0->0.5) - should ramp smoothly."""
    print("\n[2] WET TRANSITION (beta 0->0.5)")
    print("-" * 40)

    cfg = AntiClickConfig(block_size=64, wet_latency_blocks=5, ola_K=16)
    anticlick = AntiClickOperator(cfg)
    state = anticlick.init_state()

    anticlick.note_on(state)
    anticlick.set_wet_target(state, 0.5)  # Transition to 50% wet

    samples = []
    phase = 0.0
    freq = 261.63
    sample_rate = 48000

    for block in range(50):
        # Generate sine carrier with continuous phase
        carrier = np.zeros(64, dtype=np.float32)
        for i in range(64):
            carrier[i] = np.sin(phase) * 0.5
            phase += 2 * np.pi * freq / sample_rate

        # Wet is the same as carrier (ideal case - should blend smoothly)
        wet = carrier.copy() if block >= 5 else None

        output = anticlick.step(state, carrier, wet)
        samples.extend(output.tolist())

        if block < 12:
            print(f"  Block {block}: beta={state.beta:.2f}, fifo_warmup={state.fifo_warmup}")

    arr = np.array(samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]
    print(f"\nTotal samples: {len(arr)}")
    print(f"Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")
    print(f"Threshold (10x mean): {threshold:.4f}")
    print(f"Spikes: {len(spikes)}")

    # Check block boundaries
    for block in range(1, 15):
        idx = block * 64 - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.4f}")

    return len(spikes)


def test_carrier_alone():
    """Test CarrierOscillator directly (no pipeline)."""
    print("\n[3] CARRIER OSCILLATOR ALONE")
    print("-" * 40)

    from symcrash_python_ap.wavelet.carrier import CarrierOscillator

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=60, reset_phase=True)
    carrier.set_envelope_instant(0)
    carrier.gate_on(0)

    samples = []
    for block in range(50):
        block_samples = carrier.render_block(0, 64)
        samples.extend(block_samples)

    arr = np.array(samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]
    print(f"Total samples: {len(arr)}")
    print(f"Max diff: {max_diff:.0f}, Mean diff: {mean_diff:.2f}")
    print(f"Threshold (10x mean): {threshold:.0f}")
    print(f"Spikes: {len(spikes)}")

    # Check block boundaries
    for block in range(1, 15):
        idx = block * 64 - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.0f}")

    return len(spikes)


def test_pipeline_dry():
    """Test CarrierLiftPipeline with pure dry (beta=0)."""
    print("\n[4] PIPELINE PURE DRY (beta=0)")
    print("-" * 40)

    from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
    from symcrash_python_ap.frames import VoiceFrame

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    carrier_samples = []
    output_samples = []

    for block in range(50):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=261.63,
            gain_q15=32767,
            beta_target_q15=0,  # Pure dry
        )

        # Capture carrier BEFORE anticlick
        x_carrier = pipeline.carrier.render_block(0, 64)
        carrier_samples.extend(x_carrier)

        # Now run anticlick
        ac_state = pipeline.ac_states[0]
        carrier_np = np.array(x_carrier, dtype=np.float32) / 32768.0
        pipeline.anticlick.set_wet_target(ac_state, 0.0)
        output_np = pipeline.anticlick.step(ac_state, carrier_np, None)

        # Convert back to Q15
        output = (output_np * 32768.0).astype(np.int32)
        output = np.clip(output, -32768, 32767).tolist()

        # Apply gain
        output = [(s * frame.gain_q15) >> 15 for s in output]
        output_samples.extend(output)

        if block < 8:
            # Debug: show the transition
            prev_carrier = carrier_samples[(block-1)*64 + 63] if block > 0 else 0
            curr_carrier = carrier_samples[block*64]
            prev_output = output_samples[(block-1)*64 + 63] if block > 0 else 0
            curr_output = output_samples[block*64]
            print(f"  Block {block}: carrier {prev_carrier:6d}->{curr_carrier:6d} (diff={abs(curr_carrier-prev_carrier):5d}), output {prev_output:6d}->{curr_output:6d} (diff={abs(curr_output-prev_output):5d})")

    arr = np.array(output_samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]
    print(f"\nTotal samples: {len(arr)}")
    print(f"Max diff: {max_diff:.0f}, Mean diff: {mean_diff:.2f}")
    print(f"Threshold (10x mean): {threshold:.0f}")
    print(f"Spikes: {len(spikes)}")

    # Check block boundaries
    for block in range(1, 15):
        idx = block * 64 - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.0f}")

    return len(spikes)


def test_pipeline_wet():
    """Test CarrierLiftPipeline with wet path (beta=0.5)."""
    print("\n[5] PIPELINE WET (beta=0.5)")
    print("-" * 40)

    from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
    from symcrash_python_ap.frames import VoiceFrame

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    samples = []
    for block in range(50):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=261.63,
            gain_q15=32767,
            beta_target_q15=16384,  # 50% wet
        )
        L, R = pipeline.render_block(frame)
        samples.extend(L)

    arr = np.array(samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]
    print(f"Total samples: {len(arr)}")
    print(f"Max diff: {max_diff:.0f}, Mean diff: {mean_diff:.2f}")
    print(f"Threshold (10x mean): {threshold:.0f}")
    print(f"Spikes: {len(spikes)}")

    # Check block boundaries
    for block in range(1, 15):
        idx = block * 64 - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.0f}")

    return len(spikes)


if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE ANTICLICK TEST")
    print("=" * 60)

    dry_spikes = test_pure_dry()
    wet_spikes = test_wet_transition()
    carrier_spikes = test_carrier_alone()
    pipeline_dry = test_pipeline_dry()
    pipeline_wet = test_pipeline_wet()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pure dry (direct anticlick): {dry_spikes} spikes")
    print(f"Wet transition (direct anticlick): {wet_spikes} spikes")
    print(f"Carrier alone: {carrier_spikes} spikes")
    print(f"Pipeline dry: {pipeline_dry} spikes")
    print(f"Pipeline wet: {pipeline_wet} spikes")

    if carrier_spikes == 0:
        print("\n[OK] Carrier oscillator is clean")
    else:
        print("\n[ISSUE] Carrier oscillator has discontinuities")

    if pipeline_dry == 0:
        print("[OK] Pipeline dry path is clean")
    else:
        print("[ISSUE] Pipeline dry path has discontinuities")

    if pipeline_wet == 0:
        print("[OK] Pipeline wet path is clean")
    else:
        print("[ISSUE] Pipeline wet path has discontinuities")
