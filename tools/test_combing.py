"""Test for combing artifacts in wet path."""
import numpy as np
from scipy import signal
from scipy.io import wavfile

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES


class SimpleFrame:
    """Simple test frame."""
    def __init__(self, f0_hz=440.0, gain_q15=24000, beta_target_q15=32767):
        self.voice_id = 0
        self.note = 69  # A4
        self.cents = 0
        self.f0_hz = f0_hz
        self.gain_q15 = gain_q15
        self.beta_target_q15 = beta_target_q15
        self.ops_enable = []
        self.families = {}
        self.pattern_id = 0
        self.tail_mode = 0
        self.tree_a_id = 0
        self.tree_b_id = 0
        self.tree_blend_q15 = 0
        self.partial_amps_q15 = [32767] + [0] * 15  # Pure sine
        self.event_mask = 0
        self.event_strength_q15 = [0] * 8


def generate_test_audio(num_blocks=200, eta=32767):
    """Generate test audio with specified eta (wet/dry)."""
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=440.0)

    frame = SimpleFrame(f0_hz=440.0, beta_target_q15=eta)

    all_samples = []

    for i in range(num_blocks):
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    return np.array(all_samples, dtype=np.int16)


def analyze_spectrum(samples, sr=48000, title=""):
    """Analyze spectrum for non-harmonic content."""
    # Skip first 10 blocks (latency warmup)
    samples = samples[640:]

    # Compute FFT
    n = len(samples)
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, 1/sr)
    magnitude_db = 20 * np.log10(np.abs(fft) + 1e-10)

    # Find fundamental
    fund_idx = np.argmax(magnitude_db[:len(magnitude_db)//2])
    fund_freq = freqs[fund_idx]
    fund_db = magnitude_db[fund_idx]

    print(f"\n{title}")
    print(f"Fundamental: {fund_freq:.1f} Hz at {fund_db:.1f} dB")

    # Check for non-harmonic peaks (not multiples of fundamental)
    # Define harmonic tolerance (within 2% of expected harmonic)
    threshold_db = fund_db - 60  # Look for peaks above -60 dB relative to fund

    non_harmonic_peaks = []
    for i, (f, db) in enumerate(zip(freqs, magnitude_db)):
        if db < threshold_db:
            continue
        if f < 50 or f > 15000:  # Skip DC and very high
            continue

        # Check if it's a harmonic
        is_harmonic = False
        for h in range(1, 20):
            expected = fund_freq * h
            if abs(f - expected) < expected * 0.02:  # Within 2%
                is_harmonic = True
                break

        if not is_harmonic:
            non_harmonic_peaks.append((f, db - fund_db))

    if non_harmonic_peaks:
        print(f"Non-harmonic peaks (relative to fundamental):")
        for f, db_rel in sorted(non_harmonic_peaks, key=lambda x: -x[1])[:10]:
            print(f"  {f:.1f} Hz: {db_rel:.1f} dB")
    else:
        print("No significant non-harmonic peaks found")

    return non_harmonic_peaks


def check_discontinuities(samples, threshold=1000):
    """Check for sample-to-sample discontinuities."""
    # Skip first 10 blocks
    samples = samples[640:]

    diffs = np.abs(np.diff(samples.astype(np.int32)))
    large_diffs = np.where(diffs > threshold)[0]

    print(f"\nDiscontinuities (diff > {threshold}):")
    if len(large_diffs) > 0:
        print(f"  Found {len(large_diffs)} locations")
        # Check if they're at block boundaries (every 64 samples)
        block_boundary_count = sum(1 for i in large_diffs if (i + 640) % 64 < 3)
        print(f"  At block boundaries: {block_boundary_count}")
    else:
        print("  None found")

    return large_diffs


def main():
    print("=" * 60)
    print("COMBING ARTIFACT ANALYSIS")
    print("=" * 60)

    # Generate with eta=0 (dry only)
    print("\n--- ETA=0 (Carrier Only) ---")
    dry_samples = generate_test_audio(200, eta=0)
    analyze_spectrum(dry_samples, title="Dry Spectrum")
    check_discontinuities(dry_samples)

    # Generate with eta=32767 (full wet)
    print("\n--- ETA=32767 (Full Wet) ---")
    wet_samples = generate_test_audio(200, eta=32767)
    analyze_spectrum(wet_samples, title="Wet Spectrum")
    check_discontinuities(wet_samples)

    # Generate with eta=16384 (50% mix)
    print("\n--- ETA=16384 (50% Mix) ---")
    mix_samples = generate_test_audio(200, eta=16384)
    analyze_spectrum(mix_samples, title="50% Mix Spectrum")
    check_discontinuities(mix_samples)

    # Save for external analysis
    wavfile.write("test_dry.wav", 48000, dry_samples)
    wavfile.write("test_wet.wav", 48000, wet_samples)
    wavfile.write("test_mix.wav", 48000, mix_samples)
    print("\nSaved: test_dry.wav, test_wet.wav, test_mix.wav")


if __name__ == "__main__":
    main()
