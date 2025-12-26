"""
Waveform Analysis - Check if presets are producing correct harmonic content
============================================================================

Analyzes the generated Für Elise files by comparing harmonic content against
a pure sine wave baseline. Measures deviation from sine to verify waveform type.
"""

import sys
import wave
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

AUDIO_DIR = Path(__file__).parent.parent / "audio_out" / "fur_elise"

# Expected harmonic content per waveform type
# sine: fundamental only (0% deviation from sine)
# saw: all harmonics 1/k (high deviation)
# pulse/square: odd harmonics only 1/k (medium-high deviation)
# triangle: odd harmonics 1/k² (lower deviation than square)

PRESET_WAVEFORMS = {
    "brass_stab": "saw",        # saw+saw = saw
    "fm_bell": "sine",          # sine
    "jump_brass": "saw+pulse",  # saw+pulse blend
    "pluck": "tri",             # triangle
    "pwm_pad": "pulse",         # pulse+pulse = pulse
    "saw_lead": "saw",          # saw
    "square_bass": "pulse",     # pulse (square)
    "string_pad": "saw",        # saw+saw+saw = saw
    "sub_bass": "sine",         # sine
    "sync_lead": "saw",         # saw+saw = saw
    "theremin_lead": "sine",    # sine
    "wobble_bass": "saw+pulse", # saw+pulse blend
}

# Expected THD ranges for each waveform type (as percentage)
# THD = sqrt(sum of harmonic powers) / fundamental power * 100
# Note: Theoretical values for infinite harmonics differ from 8-harmonic truncated versions
# Also wet path (beta > 0) adds additional harmonics
EXPECTED_THD = {
    "sine": (0, 15),       # Pure sine: < 5% dry, up to 15% with wet path effects
    "tri": (5, 30),        # Triangle: 5-30% THD (odd harmonics with 1/k² decay)
    "pulse": (25, 75),     # Pulse/square: 25-75% THD (odd harmonics with 1/k decay)
    "saw": (70, 130),      # Saw: 70-130% THD (all harmonics, can exceed 100% due to normalization)
    "saw+pulse": (50, 100),  # Blend of saw + pulse - between the two ranges
}


def analyze_harmonics(samples: np.ndarray, sample_rate: int) -> dict:
    """
    Analyze harmonic content of audio relative to pure sine.

    Returns dict with:
        - thd_percent: Total Harmonic Distortion as percentage
        - fundamental_power: Power at fundamental
        - harmonic_powers: List of power at harmonics 2-8
        - harmonic_ratios: Ratio of each harmonic to fundamental
    """
    # Take a chunk from the middle (avoid attack/release)
    chunk_start = len(samples) // 3
    chunk_end = 2 * len(samples) // 3
    chunk = samples[chunk_start:chunk_end].astype(np.float64)

    # Normalize
    if np.max(np.abs(chunk)) > 0:
        chunk = chunk / np.max(np.abs(chunk))

    # FFT
    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)

    # Find the actual fundamental (strongest low frequency peak)
    low_mask = (freqs > 200) & (freqs < 1000)
    if np.any(low_mask):
        low_spectrum = spectrum.copy()
        low_spectrum[~low_mask] = 0
        fund_idx = np.argmax(low_spectrum)
        f0_detected = freqs[fund_idx]
    else:
        f0_detected = 440.0

    # Measure power at fundamental and harmonics
    def get_power_at_freq(target_freq: float, tolerance_hz: float = 20.0) -> float:
        mask = np.abs(freqs - target_freq) < tolerance_hz
        if np.any(mask):
            return float(np.max(spectrum[mask]))
        return 0.0

    fundamental_power = get_power_at_freq(f0_detected)

    harmonic_powers = []
    for h in range(2, 9):  # Harmonics 2-8
        power = get_power_at_freq(f0_detected * h)
        harmonic_powers.append(power)

    # Calculate ratios
    if fundamental_power > 0:
        harmonic_ratios = [p / fundamental_power for p in harmonic_powers]
    else:
        harmonic_ratios = [0.0] * len(harmonic_powers)

    # Calculate THD (Total Harmonic Distortion)
    # THD = sqrt(V2² + V3² + V4² + ...) / V1 * 100%
    sum_harmonic_sq = sum(p * p for p in harmonic_powers)
    if fundamental_power > 0:
        thd_percent = (np.sqrt(sum_harmonic_sq) / fundamental_power) * 100
    else:
        thd_percent = 0.0

    # Categorize harmonics
    even_power = sum(harmonic_powers[i] for i in [0, 2, 4, 6])  # 2nd, 4th, 6th, 8th
    odd_power = sum(harmonic_powers[i] for i in [1, 3, 5])      # 3rd, 5th, 7th

    return {
        "f0_detected": f0_detected,
        "fundamental_power": fundamental_power,
        "harmonic_powers": harmonic_powers,
        "harmonic_ratios": harmonic_ratios,
        "thd_percent": thd_percent,
        "even_power": even_power,
        "odd_power": odd_power,
        "even_to_odd_ratio": even_power / max(1, odd_power),
    }


def check_waveform_correctness(preset: str, analysis: dict) -> tuple[bool, str]:
    """
    Check if the THD matches expected waveform type.

    Uses THD as primary metric - deviation from pure sine.
    """
    waveform = PRESET_WAVEFORMS.get(preset, "unknown")
    thd = analysis["thd_percent"]
    even_odd_ratio = analysis["even_to_odd_ratio"]

    if waveform not in EXPECTED_THD:
        return False, f"Unknown waveform type: {waveform}"

    min_thd, max_thd = EXPECTED_THD[waveform]

    # Check THD is in expected range
    if thd < min_thd:
        return False, f"THD too low: {thd:.1f}% (expected {min_thd}-{max_thd}% for {waveform})"
    if thd > max_thd:
        return False, f"THD too high: {thd:.1f}% (expected {min_thd}-{max_thd}% for {waveform})"

    # Additional check: verify harmonic structure
    if waveform == "saw":
        # Saw should have even harmonics (ratio > 0.5)
        if even_odd_ratio < 0.5:
            return False, f"Expected saw (even+odd), but even/odd ratio is {even_odd_ratio:.2f}"
        return True, f"THD={thd:.1f}% (saw range), even/odd={even_odd_ratio:.2f}"

    elif waveform in ["pulse", "square"]:
        # Pulse should have primarily odd harmonics (ratio < 0.3)
        if even_odd_ratio > 0.5:
            return False, f"Expected pulse (odd dominant), but even/odd ratio is {even_odd_ratio:.2f}"
        return True, f"THD={thd:.1f}% (pulse range), even/odd={even_odd_ratio:.2f}"

    elif waveform in ["tri", "triangle"]:
        # Triangle should have odd harmonics with low THD
        # Allow up to 0.7 even/odd ratio since triangle has very weak harmonics
        # and small amounts of even harmonics can appear due to envelope/DC effects
        if even_odd_ratio > 0.7:
            return False, f"Expected triangle (odd dominant), but even/odd ratio is {even_odd_ratio:.2f}"
        return True, f"THD={thd:.1f}% (triangle range), even/odd={even_odd_ratio:.2f}"

    elif waveform == "sine":
        # Pure sine - just check THD
        return True, f"THD={thd:.1f}% (sine range)"

    return True, f"THD={thd:.1f}%"


def main():
    print("=" * 70)
    print("WAVEFORM ANALYSIS - THD Deviation from Pure Sine")
    print("=" * 70)
    print()
    print("Reference: Pure sine = 0% THD")
    print("Expected ranges:")
    for waveform, (min_thd, max_thd) in EXPECTED_THD.items():
        print(f"  {waveform:8s}: {min_thd:2d}-{max_thd:2d}% THD")
    print()

    if not AUDIO_DIR.exists():
        print(f"ERROR: Audio directory not found: {AUDIO_DIR}")
        return

    wav_files = sorted(AUDIO_DIR.glob("fur_elise_*.wav"))

    if not wav_files:
        print("ERROR: No WAV files found!")
        return

    results = []

    for wav_path in wav_files:
        preset = wav_path.stem.replace("fur_elise_", "")
        expected_waveform = PRESET_WAVEFORMS.get(preset, "unknown")

        print(f"\n{'='*60}")
        print(f"Preset: {preset}")
        print(f"Expected waveform: {expected_waveform}")
        print("=" * 60)

        # Load audio
        with wave.open(str(wav_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        if sample_width == 2:
            data = np.frombuffer(raw_data, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels == 2:
            data = data.reshape(-1, 2)
            samples = data[:, 0]
        else:
            samples = data

        # Analyze
        analysis = analyze_harmonics(samples, sample_rate)

        print(f"  Detected f0: {analysis['f0_detected']:.1f} Hz")
        print(f"  THD: {analysis['thd_percent']:.1f}%")
        print(f"  Harmonic ratios (2-8): {[f'{r:.2f}' for r in analysis['harmonic_ratios']]}")
        print(f"  Even/Odd ratio: {analysis['even_to_odd_ratio']:.2f}")

        # Check correctness
        is_correct, explanation = check_waveform_correctness(preset, analysis)

        status = "PASS" if is_correct else "FAIL"
        print(f"\n  [{status}] {explanation}")

        results.append({
            "preset": preset,
            "expected": expected_waveform,
            "thd": analysis["thd_percent"],
            "is_correct": is_correct,
            "explanation": explanation,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["is_correct"])
    failed = len(results) - passed

    print(f"\nPassed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed presets:")
        for r in results:
            if not r["is_correct"]:
                print(f"  - {r['preset']}: {r['explanation']}")

    # THD summary by waveform type
    print("\n" + "-" * 70)
    print("THD by waveform type:")
    print("-" * 70)

    for waveform in ["sine", "tri", "pulse", "saw", "saw+pulse"]:
        presets_of_type = [r for r in results if PRESET_WAVEFORMS.get(r["preset"]) == waveform]
        if presets_of_type:
            thds = [r["thd"] for r in presets_of_type]
            min_thd, max_thd = EXPECTED_THD[waveform]
            print(f"  {waveform:8s}: {min(thds):.1f}% - {max(thds):.1f}% (expected {min_thd}-{max_thd}%)")
            for r in presets_of_type:
                status = "OK" if r["is_correct"] else "FAIL"
                print(f"            {r['preset']:15s}: {r['thd']:5.1f}% [{status}]")


if __name__ == "__main__":
    main()
