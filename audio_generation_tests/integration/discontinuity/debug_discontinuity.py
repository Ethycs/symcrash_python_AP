"""
Debug Discontinuities - Trace carrier vs wet paths
===================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame


def check_carrier_only():
    """Test with pure carrier (beta=0) - should have no discontinuities."""
    print("\n[1] PURE CARRIER (beta=0) - RAW (no safety)")
    print("-" * 40)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    samples = []
    for i in range(50):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=261.63,
            gain_q15=16384,
            beta_target_q15=0,  # Pure dry
        )
        # Use render_voice directly to bypass SafetyEngine
        raw_samples = pipeline.render_voice(frame)
        samples.extend(raw_samples)

    arr = np.array(samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]
    print(f"Total samples: {len(arr)}")
    print(f"Max diff: {max_diff:.0f}, Mean diff: {mean_diff:.2f}")
    print(f"Spikes (>{threshold:.0f}): {len(spikes)}")

    # Check block boundaries
    block_size = 64
    boundary_spikes = 0
    for block in range(1, 20):
        idx = block * block_size - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.0f}")
                boundary_spikes += 1

    return len(spikes)


def check_wet_path():
    """Test with wet path (beta=0.5) - may have issues."""
    print("\n[2] WET PATH (beta=0.5)")
    print("-" * 40)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    samples = []
    for i in range(50):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=261.63,
            gain_q15=16384,
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
    print(f"Spikes (>{threshold:.0f}): {len(spikes)}")

    # Check block boundaries
    block_size = 64
    boundary_spikes = 0
    for block in range(1, 20):
        idx = block * block_size - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.0f}")
                boundary_spikes += 1

    return len(spikes)


def check_anticlick_raw():
    """Bypass unified_renderer, test anticlick directly."""
    print("\n[3] DIRECT ANTICLICK TEST")
    print("-" * 40)

    from symcrash_python_ap.safety.anti_click import AntiClickOperator, AntiClickConfig

    cfg = AntiClickConfig(block_size=64, ola_K=16)
    anticlick = AntiClickOperator(cfg)
    state = anticlick.init_state()

    anticlick.note_on(state)
    anticlick.set_wet_target(state, 0.5)

    samples = []
    for block in range(50):
        # Generate simple sine carrier
        t = np.arange(64) / 48000.0 + block * 64 / 48000.0
        carrier = (np.sin(2 * np.pi * 261.63 * t) * 0.5).astype(np.float32)

        # Generate same sine as wet (should be identical)
        wet = carrier.copy()

        output = anticlick.step(state, carrier, wet if block >= 5 else None)
        samples.extend(output.tolist())

    arr = np.array(samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]
    print(f"Total samples: {len(arr)}")
    print(f"Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")
    print(f"Spikes (>{threshold:.4f}): {len(spikes)}")

    # Check block boundaries
    block_size = 64
    for block in range(1, 15):
        idx = block * block_size - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.4f}")

    return len(spikes)


def check_carrier_oscillator():
    """Test carrier oscillator directly."""
    print("\n[4] CARRIER OSCILLATOR DIRECT")
    print("-" * 40)

    from symcrash_python_ap.wavelet.carrier import CarrierOscillator

    osc = CarrierOscillator()
    osc.configure_voice(0, note=60, reset_phase=True)
    osc.set_envelope_instant(0)
    osc.gate_on(0)

    samples = []
    for block in range(50):
        block_samples = osc.render_block(0, 64)
        samples.extend(block_samples)

    arr = np.array(samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]
    print(f"Total samples: {len(arr)}")
    print(f"Max diff: {max_diff:.0f}, Mean diff: {mean_diff:.2f}")
    print(f"Spikes (>{threshold:.0f}): {len(spikes)}")

    # Check block boundaries
    block_size = 64
    for block in range(1, 15):
        idx = block * block_size - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.0f}")

    return len(spikes)


def check_render_voice_detailed():
    """Trace render_voice step by step."""
    print("\n[5] DETAILED RENDER_VOICE TRACE")
    print("-" * 40)

    from symcrash_python_ap.safety.anti_click import AntiClickOperator, AntiClickConfig

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    carrier_samples = []
    delayed_samples = []
    anticlick_output = []
    final_output = []

    for block in range(20):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=261.63,
            gain_q15=32767,  # Full gain
            beta_target_q15=0,  # Pure dry
        )

        # Step 1: Get carrier directly
        x_carrier = pipeline.carrier.render_block(0, 64)
        carrier_samples.extend(x_carrier)

        # Step 2: Check FIFO state BEFORE anticlick.step
        ac_state = pipeline.ac_states[0]
        delayed_before = ac_state.carrier_fifo[0].copy()
        delayed_samples.extend((delayed_before * 32768).astype(int).tolist())

        # Step 3: Get anticlick output
        carrier_np = np.array(x_carrier, dtype=np.float32) / 32768.0
        output_np = pipeline.anticlick.step(ac_state, carrier_np, None)
        output = (output_np * 32768.0).astype(np.int32)
        output = np.clip(output, -32768, 32767).tolist()
        anticlick_output.extend(output)

        # Step 4: Apply gain (render_voice does this)
        output = [(s * frame.gain_q15) >> 15 for s in output]
        final_output.extend(output)

        # Debug: show first and last sample of each block
        if block < 10:
            prev_block_last = anticlick_output[(block-1)*64 + 63] if block > 0 else 0
            curr_block_first = anticlick_output[block*64]
            boundary_diff = abs(curr_block_first - prev_block_last)
            print(f"  Block {block}: carrier[0]={x_carrier[0]:6d}, delayed[0]={int(delayed_before[0]*32768):6d}, output[0]={curr_block_first:6d}, boundary_diff={boundary_diff:6d}")

    def analyze(name, samples):
        arr = np.array(samples, dtype=np.float64)
        diffs = np.abs(np.diff(arr))
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)
        threshold = mean_diff * 10 if mean_diff > 0 else 1000

        spikes = np.where(diffs > threshold)[0]
        boundary_spikes = []
        for block in range(1, 10):
            idx = block * 64 - 1
            if idx < len(diffs) and diffs[idx] > threshold:
                boundary_spikes.append((block, diffs[idx]))

        print(f"  {name}:")
        print(f"    Max diff: {max_diff:.0f}, Mean: {mean_diff:.2f}, Spikes: {len(spikes)}")
        if boundary_spikes:
            for b, d in boundary_spikes[:3]:
                print(f"    Block {b} boundary: {d:.0f}")

    print("\nAnalysis:")
    analyze("Carrier raw", carrier_samples)
    analyze("Delayed (from FIFO)", delayed_samples)
    analyze("AntiClick output", anticlick_output)
    analyze("Final (with gain)", final_output)


if __name__ == "__main__":
    print("=" * 60)
    print("DISCONTINUITY DEBUG")
    print("=" * 60)

    carrier_spikes = check_carrier_oscillator()
    dry_spikes = check_carrier_only()
    raw_spikes = check_anticlick_raw()
    wet_spikes = check_wet_path()
    check_render_voice_detailed()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Carrier oscillator spikes: {carrier_spikes}")
    print(f"Pure dry (beta=0) spikes: {dry_spikes}")
    print(f"Direct anticlick spikes: {raw_spikes}")
    print(f"Wet path (beta=0.5) spikes: {wet_spikes}")

    if carrier_spikes > 0:
        print("\n=> ISSUE IS IN CARRIER OSCILLATOR")
    elif dry_spikes > 0:
        print("\n=> ISSUE IS IN PIPELINE DRY PATH")
    elif raw_spikes > 0:
        print("\n=> ISSUE IS IN ANTICLICK OLA")
    elif wet_spikes > 0:
        print("\n=> ISSUE IS IN WET PATH (wavelet/effect_mode)")
    else:
        print("\n=> NO DISCONTINUITIES DETECTED")
