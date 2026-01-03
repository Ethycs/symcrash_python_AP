"""Test if SafetyEngine DC removal is causing the artifacts."""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame, ParamFrame

BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def generate_with_safety(beta_target, apply_dc_removal, num_blocks=200):
    """Generate audio through full pipeline with control over DC removal."""
    frame = VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        gain_q15=24576,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=beta_target,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )

    stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
    stage.voice_on(0, frame.note, frame.cents)

    all_samples = []
    for _ in range(num_blocks):
        # Get samples from VM (bypasses some safety)
        samples = stage.render_voice(frame)

        # Apply pan
        from symcrash_python_ap.wavelet.sine_core import cosine_q15, PHASE_MASK
        from symcrash_python_ap.fixed import mul_q15_sat
        pan_phase = (32768 << 14) & PHASE_MASK  # Center pan
        left_q15 = cosine_q15(pan_phase)
        L = [mul_q15_sat(s, left_q15) for s in samples]
        R = L.copy()  # Center pan means L=R

        # Apply safety with controlled DC removal
        L, R = stage.safety.step_block(
            L, R, stage.params,
            apply_master_ola=False,
            apply_dc_removal=apply_dc_removal  # CONTROLLED
        )

        all_samples.extend(L)

    return np.array(all_samples, dtype=np.int16)


def analyze(samples, label):
    """Analyze spectrum."""
    print(f"\n{label}:")
    print(f"  RMS: {np.sqrt(np.mean(samples.astype(float)**2)):.1f}")
    print(f"  Max: {np.abs(samples).max()}")

    # Check discontinuities
    disc = 0
    for i in range(64, min(len(samples), 6400), 64):
        prev_d = samples[i-1] - samples[i-2]
        jump = samples[i] - samples[i-1]
        if abs(jump - prev_d) > 100:
            disc += 1
    print(f"  Block discontinuities: {disc}")

    return samples


print("Testing SafetyEngine DC removal effect...")

# Test 1: ETA=0 with DC removal OFF (normal behavior)
print("\n=== ETA=0, DC removal OFF ===")
s1 = generate_with_safety(0, apply_dc_removal=False)
analyze(s1, "ETA=0, DC=OFF")

# Test 2: ETA=0 with DC removal ON
print("\n=== ETA=0, DC removal ON ===")
s2 = generate_with_safety(0, apply_dc_removal=True)
analyze(s2, "ETA=0, DC=ON")

# Test 3: ETA=16384 with DC removal OFF
print("\n=== ETA=16384, DC removal OFF ===")
s3 = generate_with_safety(16384, apply_dc_removal=False)
analyze(s3, "ETA=16384, DC=OFF")

# Test 4: ETA=16384 with DC removal ON (current behavior)
print("\n=== ETA=16384, DC removal ON ===")
s4 = generate_with_safety(16384, apply_dc_removal=True)
analyze(s4, "ETA=16384, DC=ON")

# Compare
print("\n=== Comparisons ===")
print(f"ETA=0: DC ON vs OFF max diff: {np.abs(s1.astype(int) - s2.astype(int)).max()}")
print(f"ETA=16384: DC ON vs OFF max diff: {np.abs(s3.astype(int) - s4.astype(int)).max()}")
print(f"DC OFF: ETA=0 vs ETA=16384 max diff: {np.abs(s1.astype(int) - s3.astype(int)).max()}")
