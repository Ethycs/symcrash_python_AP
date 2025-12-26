"""
Drone Isolation Diagnostic Tests
================================

Following the diagnostic approach:
1. Force beta=0 (carrier-only) - if drone goes away, problem is IDWPT/wet path
2. Force beta=1 (wet-only) - if wet-only drones, IDWPT is producing constant output
3. Force leaf input c[k]=0 - if output nonzero, IDWPT state isn't reset correctly
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline, RenderFrame
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.fixed import sat16

# =============================================================================
# SETUP
# =============================================================================

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)

Q15_ONE = 32767
Q15_HALF = 16384


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def save_audio(L: List[int], R: List[int], name: str) -> str:
    metadata = audio_io.write_audio(name, L, R, subdirectory="drone_debug", timestamp=False)
    return metadata.filepath


def analyze_spectrum(samples: List[int], name: str) -> dict:
    """Analyze frequency content of samples."""
    arr = np.array(samples, dtype=np.float64)

    # FFT
    fft = np.fft.rfft(arr)
    freqs = np.fft.rfftfreq(len(arr), 1/FS)
    magnitudes = np.abs(fft)

    # Find peak frequency
    peak_idx = np.argmax(magnitudes[1:]) + 1  # Skip DC
    peak_freq = freqs[peak_idx]
    peak_mag = magnitudes[peak_idx]

    # Energy in different bands
    dc_energy = magnitudes[0]
    low_mask = (freqs > 0) & (freqs < 200)
    mid_mask = (freqs >= 200) & (freqs < 2000)
    high_mask = freqs >= 2000

    low_energy = np.sum(magnitudes[low_mask])
    mid_energy = np.sum(magnitudes[mid_mask])
    high_energy = np.sum(magnitudes[high_mask])
    total_energy = np.sum(magnitudes[1:])

    # RMS
    rms = np.sqrt(np.mean(arr ** 2))

    result = {
        'name': name,
        'rms': rms,
        'peak_freq': peak_freq,
        'peak_mag': peak_mag,
        'dc_energy': dc_energy,
        'low_pct': 100 * low_energy / max(1, total_energy),
        'mid_pct': 100 * mid_energy / max(1, total_energy),
        'high_pct': 100 * high_energy / max(1, total_energy),
    }

    print(f"\n{name}:")
    print(f"  RMS: {rms:.1f}")
    print(f"  Peak freq: {peak_freq:.1f} Hz (mag={peak_mag:.1f})")
    print(f"  DC energy: {dc_energy:.1f}")
    print(f"  Low (<200Hz): {result['low_pct']:.1f}%")
    print(f"  Mid (200-2kHz): {result['mid_pct']:.1f}%")
    print(f"  High (>2kHz): {result['high_pct']:.1f}%")

    return result


# =============================================================================
# TEST 1: Force beta=0 (Carrier Only)
# =============================================================================

def test_carrier_only():
    """Force beta=0 to isolate carrier path."""
    print("\n" + "="*60)
    print("TEST 1: CARRIER ONLY (beta=0)")
    print("If drone goes away, problem is IDWPT/wet path")
    print("="*60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)  # C4 = 261.6 Hz

    num_blocks = int(3.0 * BLOCKS_PER_SEC)
    all_L = []
    all_R = []

    for _ in range(num_blocks):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,  # FORCE CARRIER ONLY
        )
        L, R = pipeline.render_block(frame)
        all_L.extend(L)
        all_R.extend(R)

    filepath = save_audio(all_L, all_R, "1_carrier_only_beta0")
    result = analyze_spectrum(all_L, "Carrier Only (beta=0)")
    print(f"Saved: {filepath}")

    pipeline.voice_off(0)
    return result


# =============================================================================
# TEST 2: Force beta=1 (Wet Only)
# =============================================================================

def test_wet_only():
    """Force beta=1 to isolate wet/IDWPT path."""
    print("\n" + "="*60)
    print("TEST 2: WET ONLY (beta=1)")
    print("If wet-only drones, IDWPT is producing constant output")
    print("="*60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    # Force wet ready immediately by bypassing latency counter
    pipeline.blend_states[0].wet_ready_counter = 0
    pipeline.blend_states[0].beta_q15 = Q15_ONE

    num_blocks = int(3.0 * BLOCKS_PER_SEC)
    all_L = []
    all_R = []

    for _ in range(num_blocks):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=Q15_ONE,  # FORCE WET ONLY
        )

        # Force beta to stay at 1
        pipeline.blend_states[0].beta_q15 = Q15_ONE

        L, R = pipeline.render_block(frame)
        all_L.extend(L)
        all_R.extend(R)

    filepath = save_audio(all_L, all_R, "2_wet_only_beta1")
    result = analyze_spectrum(all_L, "Wet Only (beta=1)")
    print(f"Saved: {filepath}")

    pipeline.voice_off(0)
    return result


# =============================================================================
# TEST 3: Zero Leaf Coefficients
# =============================================================================

def test_zero_leaf_input():
    """Force leaf input c[k]=0 to check for IDWPT DC leakage."""
    print("\n" + "="*60)
    print("TEST 3: ZERO LEAF INPUT (WET PATH ONLY)")
    print("If output nonzero, IDWPT state isn't being reset correctly")
    print("="*60)

    # Access WaveletBackend directly to test IDWPT in isolation
    from symcrash_python_ap.safety.unified_renderer import WaveletBackend, RenderFrame

    backend = WaveletBackend(block_size=BLOCK_SIZE)
    backend.engine.voice_allocate(0)
    backend.engine.reset_wet_state(0)

    num_blocks = int(3.0 * BLOCKS_PER_SEC)
    all_samples = []

    for _ in range(num_blocks):
        # Create frame with ALL ZERO coefficients
        frame = RenderFrame(
            voice_id=0,
            note=60,
            gain_q15=Q15_HALF,
            coeff_q15=[0] * 256,  # ALL ZERO COEFFICIENTS
        )

        samples = backend.render_block(frame, BLOCK_SIZE)
        all_samples.extend(samples)

    # Mono to stereo for saving
    filepath = save_audio(all_samples, all_samples, "3_zero_leaf_input_wet_only")
    result = analyze_spectrum(all_samples, "Zero Leaf Input (Wet Only)")
    print(f"Saved: {filepath}")

    # This should be near-silent if IDWPT is working correctly
    if result['rms'] > 100:
        print("\n  WARNING: Non-zero output with zero input!")
        print("     This indicates DC leakage or stale state in IDWPT")
    else:
        print("\n  OK: Output near-silent (expected)")

    return result


# =============================================================================
# TEST 4: Default Blend (Normal Operation)
# =============================================================================

def test_default_blend():
    """Normal operation for comparison."""
    print("\n" + "="*60)
    print("TEST 4: DEFAULT BLEND (Normal Operation)")
    print("Reference for comparison")
    print("="*60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    num_blocks = int(3.0 * BLOCKS_PER_SEC)
    all_L = []
    all_R = []

    for _ in range(num_blocks):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            # Default beta_target_q15 (~0.75)
        )
        L, R = pipeline.render_block(frame)
        all_L.extend(L)
        all_R.extend(R)

    filepath = save_audio(all_L, all_R, "4_default_blend")
    result = analyze_spectrum(all_L, "Default Blend (~0.75 wet)")
    print(f"Saved: {filepath}")

    pipeline.voice_off(0)
    return result


# =============================================================================
# TEST 5: Check Leaf 0 Energy (DC-ish leaf)
# =============================================================================

def test_leaf_0_suppression():
    """Check if suppressing leaf 0 reduces drone."""
    print("\n" + "="*60)
    print("TEST 5: SUPPRESS LEAF 0")
    print("Leaf 0 is near-DC. Suppressing it may eliminate drone.")
    print("="*60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    # Manually suppress leaf 0 in tuning
    original_gains = pipeline.tuning.get_current_gains()
    modified_gains = original_gains.copy()
    modified_gains[0] = 0  # Zero out leaf 0
    modified_gains[1] = modified_gains[1] // 2  # Reduce leaf 1

    num_blocks = int(3.0 * BLOCKS_PER_SEC)
    all_L = []
    all_R = []

    for _ in range(num_blocks):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
        )

        # Manually patch the tuning for this test
        pipeline.tuning._state.tuning_a.gains_q15 = modified_gains
        pipeline.tuning._state.tuning_b.gains_q15 = modified_gains

        L, R = pipeline.render_block(frame)
        all_L.extend(L)
        all_R.extend(R)

    filepath = save_audio(all_L, all_R, "5_suppress_leaf0")
    result = analyze_spectrum(all_L, "Suppress Leaf 0")
    print(f"Saved: {filepath}")

    pipeline.voice_off(0)
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("DRONE ISOLATION DIAGNOSTIC")
    print("="*60)
    print(f"\nExpected note: C4 = 261.6 Hz")
    print(f"Sample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")

    results = {}

    results['carrier_only'] = test_carrier_only()
    results['wet_only'] = test_wet_only()
    results['zero_input'] = test_zero_leaf_input()
    results['default'] = test_default_blend()

    try:
        results['suppress_leaf0'] = test_leaf_0_suppression()
    except Exception as e:
        print(f"\nTest 5 failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\n1. If carrier-only has no drone but default does:")
    print("   -> Problem is in IDWPT/wet path")
    print("\n2. If wet-only drones:")
    print("   -> IDWPT is producing constant-ish output")
    print("\n3. If zero-input produces output:")
    print("   -> IDWPT has DC leakage or stale state")
    print("\n4. If suppressing leaf 0 helps:")
    print("   -> Add DC blocker or zero leaf 0 in tuning")

    print("\n" + "="*60)
    print("Listen to the files in audio_out/drone_debug/")
    print("="*60)
