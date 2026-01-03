"""Test VM directly, bypassing UnifiedOutputStage."""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame

BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def generate(beta_target, num_blocks=200):
    """Generate audio samples directly through VM."""
    frame = VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        gain_q15=32767,  # Full gain
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=beta_target,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60, cents=0)

    all_samples = []
    for _ in range(num_blocks):
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    return np.array(all_samples, dtype=np.int16)


def analyze(samples, label):
    """Analyze spectrum."""
    print(f"\n{label}:")
    print(f"  RMS: {np.sqrt(np.mean(samples.astype(float)**2)):.1f}")
    print(f"  Max: {np.abs(samples).max()}")

    # Check discontinuities at block boundaries
    disc = 0
    for i in range(64, min(len(samples), 6400), 64):
        prev_d = samples[i-1] - samples[i-2]
        jump = samples[i] - samples[i-1]
        if abs(jump - prev_d) > 100:
            disc += 1
    print(f"  Block discontinuities: {disc}")

    # FFT
    window = windows.blackman(len(samples))
    windowed = samples.astype(float) * window
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(samples), 1/SAMPLE_RATE)

    # Find fundamental
    start = int(50 * len(freqs) / (SAMPLE_RATE/2))
    fund_idx = start + np.argmax(spectrum[start:start+2000])
    fund_mag = spectrum[fund_idx]

    # Check for non-harmonic content
    non_harm = []
    for i, (f, m) in enumerate(zip(freqs, spectrum)):
        if f < 50 or m < fund_mag * 1e-3:
            continue
        is_harm = False
        for h in range(1, 20):
            expected = freqs[fund_idx] * h
            if abs(f - expected) < expected * 0.02:
                is_harm = True
                break
        if not is_harm:
            db = 20 * np.log10(m / fund_mag + 1e-10)
            if db > -60:
                non_harm.append((f, db))

    if non_harm:
        print(f"  Non-harmonic peaks ({len(non_harm)}):")
        for f, db in sorted(non_harm, key=lambda x: -x[1])[:5]:
            print(f"    {f:.1f} Hz: {db:.1f} dB")
    else:
        print("  PURE: No non-harmonic content")

    return samples


print("Testing VM directly (no UnifiedOutputStage)...")

# Generate with different ETA values
samples_0 = generate(0)
samples_mid = generate(16384)
samples_max = generate(32767)

analyze(samples_0, "ETA=0")
analyze(samples_mid, "ETA=16384")
analyze(samples_max, "ETA=32767")

# Compare outputs
print("\n--- Direct comparison ---")
diff_0_mid = np.abs(samples_0.astype(int) - samples_mid.astype(int)).max()
diff_0_max = np.abs(samples_0.astype(int) - samples_max.astype(int)).max()
diff_mid_max = np.abs(samples_mid.astype(int) - samples_max.astype(int)).max()

print(f"Max diff ETA=0 vs ETA=16384: {diff_0_mid}")
print(f"Max diff ETA=0 vs ETA=32767: {diff_0_max}")
print(f"Max diff ETA=16384 vs ETA=32767: {diff_mid_max}")

if diff_0_mid == 0 and diff_0_max == 0:
    print("\nCONFIRMED: ETA has NO effect in CCS path (all outputs identical)")
else:
    print("\nWARNING: ETA IS affecting output even in CCS path!")
