"""Ablation study to isolate source of cyclic jaggies."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.wavelet.sine_core import sine_q15_vectorized, Q_SCALE

Q15_ONE = 32767
BLOCK_SAMPLES = 64
SAMPLE_RATE = 48000

def count_spikes(signal, threshold_sigma=3):
    """Count spikes in second derivative."""
    d2 = np.diff(np.diff(signal))
    if d2.std() == 0:
        return 0, 0
    threshold = d2.std() * threshold_sigma
    spikes = np.where(np.abs(d2) > threshold)[0]
    return len(spikes), d2.std()

def test_pure_float_sine(freq, num_samples):
    """Pure float sine - no quantization."""
    t = np.arange(num_samples) / SAMPLE_RATE
    signal = np.sin(2 * np.pi * freq * t) * Q15_ONE
    spikes, std = count_spikes(signal.astype(np.int32))
    return spikes, std, "Pure float sine"

def test_64k_lut_sine(freq, num_samples):
    """64K LUT lookup - quantized phase."""
    phase_inc = int(freq * (2**32) / SAMPLE_RATE)
    phases_32 = np.arange(num_samples, dtype=np.int64) * phase_inc
    phases_16 = ((phases_32 >> 16) & 0xFFFF).astype(np.int32)
    signal = sine_q15_vectorized(phases_16)
    spikes, std = count_spikes(signal)
    return spikes, std, "64K LUT sine"

def test_harmonic_sum_float(freq, num_samples, num_harmonics=8):
    """Float harmonic sum (SAW wave) - no Q15 quantization."""
    t = np.arange(num_samples) / SAMPLE_RATE
    signal = np.zeros(num_samples)
    for h in range(1, num_harmonics + 1):
        amp = 1.0 / h
        signal += amp * np.sin(2 * np.pi * freq * h * t)
    # Normalize
    signal = (signal / signal.max() * Q15_ONE).astype(np.int32)
    spikes, std = count_spikes(signal)
    return spikes, std, "Float harmonic sum"

def test_harmonic_sum_q15(freq, num_samples, num_harmonics=8):
    """Q15 harmonic sum with 64K LUT - like lanes.py."""
    phase_inc = int(freq * (2**32) / SAMPLE_RATE)
    phases_32 = np.arange(num_samples, dtype=np.int64) * phase_inc

    harmonic_amps = [Q15_ONE // h for h in range(1, num_harmonics + 1)]
    total_amp = sum(harmonic_amps)

    carrier = np.zeros(num_samples, dtype=np.int32)
    for h in range(num_harmonics):
        amp = harmonic_amps[h]
        h_phases = (phases_32 * (h + 1)) & 0xFFFFFFFF
        phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
        sin_vals = sine_q15_vectorized(phase_norm)
        carrier += (sin_vals * amp) >> 15

    spikes, std = count_spikes(carrier)
    return spikes, std, "Q15 harmonic sum (raw)"

def test_harmonic_sum_q15_normalized(freq, num_samples, num_harmonics=8):
    """Q15 harmonic sum with normalization - full lanes.py path."""
    phase_inc = int(freq * (2**32) / SAMPLE_RATE)
    phases_32 = np.arange(num_samples, dtype=np.int64) * phase_inc

    harmonic_amps = [Q15_ONE // h for h in range(1, num_harmonics + 1)]
    total_amp = sum(harmonic_amps)

    carrier = np.zeros(num_samples, dtype=np.int32)
    for h in range(num_harmonics):
        amp = harmonic_amps[h]
        h_phases = (phases_32 * (h + 1)) & 0xFFFFFFFF
        phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
        sin_vals = sine_q15_vectorized(phase_norm)
        carrier += (sin_vals * amp) >> 15

    # Normalize
    DC_HEADROOM = 30983
    divisor = max(total_amp, Q15_ONE)
    carrier = (carrier * DC_HEADROOM) // divisor

    spikes, std = count_spikes(carrier)
    return spikes, std, "Q15 harmonic + normalize"

def test_q15_multiply_accumulate(freq, num_samples, num_harmonics=8):
    """Test just the Q15 multiply-accumulate step."""
    phase_inc = int(freq * (2**32) / SAMPLE_RATE)
    phases_32 = np.arange(num_samples, dtype=np.int64) * phase_inc

    # Use fixed amplitude
    amp = Q15_ONE // 2

    carrier = np.zeros(num_samples, dtype=np.int32)
    for h in range(num_harmonics):
        h_phases = (phases_32 * (h + 1)) & 0xFFFFFFFF
        phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
        sin_vals = sine_q15_vectorized(phase_norm)
        # Each step: (sin * amp) >> 15
        carrier += (sin_vals * amp) >> 15

    spikes, std = count_spikes(carrier)
    return spikes, std, "Q15 mul-acc (fixed amp)"

def test_phase_quantization(freq, num_samples):
    """Test phase quantization alone - compare 32-bit vs 16-bit phase."""
    phase_inc = int(freq * (2**32) / SAMPLE_RATE)
    phases_32 = np.arange(num_samples, dtype=np.int64) * phase_inc

    # High precision: use full 32-bit phase
    signal_32 = (np.sin(2 * np.pi * phases_32.astype(np.float64) / (2**32)) * Q15_ONE).astype(np.int32)

    # Low precision: quantize to 16-bit first
    phases_16 = ((phases_32 >> 16) & 0xFFFF)
    signal_16 = (np.sin(2 * np.pi * phases_16.astype(np.float64) / 65536) * Q15_ONE).astype(np.int32)

    spikes_32, std_32 = count_spikes(signal_32)
    spikes_16, std_16 = count_spikes(signal_16)

    return [(spikes_32, std_32, "32-bit phase"),
            (spikes_16, std_16, "16-bit phase (quantized)")]


def test_different_waveforms(freq, num_samples):
    """Test different waveforms."""
    phase_inc = int(freq * (2**32) / SAMPLE_RATE)
    phases_32 = np.arange(num_samples, dtype=np.int64) * phase_inc

    results = []

    # Pure sine (1 harmonic)
    carrier = np.zeros(num_samples, dtype=np.int32)
    phases_16 = ((phases_32 >> 16) & 0xFFFF).astype(np.int32)
    carrier = sine_q15_vectorized(phases_16)
    spikes, std = count_spikes(carrier)
    results.append((spikes, std, "Sine (1 harmonic)"))

    # Square (odd harmonics only)
    carrier = np.zeros(num_samples, dtype=np.int32)
    for h in range(1, 9, 2):  # 1, 3, 5, 7
        amp = Q15_ONE // h
        h_phases = (phases_32 * h) & 0xFFFFFFFF
        phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
        sin_vals = sine_q15_vectorized(phase_norm)
        carrier += (sin_vals * amp) >> 15
    spikes, std = count_spikes(carrier)
    results.append((spikes, std, "Square (odd harmonics)"))

    # Triangle (odd harmonics, 1/n^2)
    carrier = np.zeros(num_samples, dtype=np.int32)
    for h in range(1, 9, 2):
        amp = Q15_ONE // (h * h)
        h_phases = (phases_32 * h) & 0xFFFFFFFF
        phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
        sin_vals = sine_q15_vectorized(phase_norm)
        carrier += (sin_vals * amp) >> 15
    spikes, std = count_spikes(carrier)
    results.append((spikes, std, "Triangle (1/n^2)"))

    # Saw (all harmonics, 1/n)
    carrier = np.zeros(num_samples, dtype=np.int32)
    for h in range(1, 9):
        amp = Q15_ONE // h
        h_phases = (phases_32 * h) & 0xFFFFFFFF
        phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
        sin_vals = sine_q15_vectorized(phase_norm)
        carrier += (sin_vals * amp) >> 15
    spikes, std = count_spikes(carrier)
    results.append((spikes, std, "Saw (1/n)"))

    return results


if __name__ == "__main__":
    freq = 440.0
    samples_per_cycle = int(SAMPLE_RATE / freq)
    num_samples = samples_per_cycle * 5

    print(f"Testing at {freq}Hz, {samples_per_cycle} samples/cycle, {num_samples} total samples")
    print("=" * 70)

    tests = [
        test_pure_float_sine(freq, num_samples),
        test_64k_lut_sine(freq, num_samples),
        test_harmonic_sum_float(freq, num_samples),
        test_harmonic_sum_q15(freq, num_samples),
        test_harmonic_sum_q15_normalized(freq, num_samples),
        test_q15_multiply_accumulate(freq, num_samples),
    ]

    for spikes, std, name in tests:
        spikes_per_cycle = spikes / 5.0
        status = "OK" if spikes_per_cycle < 0.5 else f"SPIKES: {spikes_per_cycle:.1f}/cycle"
        print(f"{name:35s}: d2_std={std:6.1f}, {status}")

    print("\n--- Different Waveforms ---")
    for spikes, std, name in test_different_waveforms(freq, num_samples):
        spikes_per_cycle = spikes / 5.0
        status = "OK" if spikes_per_cycle < 0.5 else f"SPIKES: {spikes_per_cycle:.1f}/cycle"
        print(f"{name:35s}: d2_std={std:6.1f}, {status}")
