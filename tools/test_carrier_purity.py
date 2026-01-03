"""Test carrier signal purity."""
import numpy as np
from scipy import signal
from scipy.io import wavfile

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES


class SimpleFrame:
    """Simple test frame."""
    def __init__(self, f0_hz=440.0, gain_q15=24000, beta_target_q15=0):
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


def generate_ideal_sine(sr=48000, freq=440.0, num_samples=12800, amplitude=20000):
    """Generate an ideal pure sine wave."""
    t = np.arange(num_samples) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def analyze_spectrum_windowed(samples, sr=48000, title=""):
    """Analyze spectrum with proper windowing."""
    n = len(samples)

    # Apply Hann window to reduce spectral leakage
    window = signal.windows.hann(n)
    windowed = samples * window

    # Zero-pad for better frequency resolution
    nfft = n * 4
    fft = np.fft.rfft(windowed, nfft)
    freqs = np.fft.rfftfreq(nfft, 1/sr)
    magnitude_db = 20 * np.log10(np.abs(fft) + 1e-10)

    # Find fundamental
    search_range = np.where((freqs > 400) & (freqs < 500))[0]
    fund_idx = search_range[np.argmax(magnitude_db[search_range])]
    fund_freq = freqs[fund_idx]
    fund_db = magnitude_db[fund_idx]

    print(f"\n{title}")
    print(f"Fundamental: {fund_freq:.2f} Hz at {fund_db:.1f} dB")

    # Check for peaks within ±50 Hz of fundamental that aren't the fundamental
    nearby = np.where((freqs > fund_freq - 50) & (freqs < fund_freq + 50))[0]
    threshold_db = fund_db - 40

    sidebands = []
    for i in nearby:
        if abs(freqs[i] - fund_freq) < 2:  # Skip fundamental itself
            continue
        if magnitude_db[i] > threshold_db:
            sidebands.append((freqs[i], magnitude_db[i] - fund_db))

    if sidebands:
        print(f"Sidebands (within ±50 Hz of fundamental):")
        for f, db_rel in sorted(sidebands, key=lambda x: -x[1])[:5]:
            print(f"  {f:.2f} Hz: {db_rel:.1f} dB (offset: {f - fund_freq:+.2f} Hz)")
    else:
        print("No significant sidebands found")

    return fund_freq, sidebands


def main():
    print("=" * 60)
    print("CARRIER PURITY ANALYSIS")
    print("=" * 60)

    # Generate ideal reference
    print("\n--- IDEAL SINE (Reference) ---")
    ideal = generate_ideal_sine(48000, 440.0, 12800, 20000)
    analyze_spectrum_windowed(ideal, title="Ideal 440 Hz Sine")

    # Generate from VM (dry)
    print("\n--- VM CARRIER (ETA=0) ---")
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=440.0)
    frame = SimpleFrame(f0_hz=440.0, beta_target_q15=0)

    all_samples = []
    for _ in range(200):  # 200 blocks = 12800 samples
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.extend(output.tolist())

    vm_samples = np.array(all_samples, dtype=np.int16)
    analyze_spectrum_windowed(vm_samples[640:], title="VM Carrier (after warmup)")

    # Check phase continuity at block boundaries
    print("\n--- BLOCK BOUNDARY CHECK ---")
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=440.0)

    prev_end = None
    max_boundary_diff = 0
    boundary_diffs = []

    for i in range(50):
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        if prev_end is not None:
            diff = abs(int(output[0]) - int(prev_end))
            boundary_diffs.append(diff)
            max_boundary_diff = max(max_boundary_diff, diff)
        prev_end = output[-1]

    print(f"Max block boundary diff: {max_boundary_diff}")
    print(f"Avg block boundary diff: {np.mean(boundary_diffs):.1f}")

    # Expected diff for continuous 440 Hz sine at 48kHz
    expected_diff_per_sample = abs(20000 * np.sin(2 * np.pi * 440 / 48000))
    print(f"Expected diff for 440 Hz sine: {expected_diff_per_sample:.1f}")


if __name__ == "__main__":
    main()
