"""Compare FM vs non-FM spectrum to verify FM is working."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame

FS = 48000
BLOCK_SIZE = 64
DURATION = 1.0  # 1 second


def generate_audio(fm_depth: int, fm_source: int) -> np.ndarray:
    """Generate audio with given FM settings."""
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B
        H_request=8,
        fm_depth_q15=fm_depth,
        fm_source=fm_source,
    )

    samples = []
    num_blocks = int(DURATION * FS / BLOCK_SIZE)
    for _ in range(num_blocks):
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())

    return np.array(samples, dtype=np.float64)


def analyze_spectrum(samples: np.ndarray, name: str):
    """Analyze and print spectrum info."""
    from scipy.fft import rfft, rfftfreq

    fft = np.abs(rfft(samples))
    freqs = rfftfreq(len(samples), 1.0 / FS)

    # Find peaks above noise floor
    noise_floor = np.median(fft)
    peaks = []
    for i in range(1, len(fft) - 1):
        if fft[i] > fft[i - 1] and fft[i] > fft[i + 1] and fft[i] > noise_floor * 10:
            peaks.append((freqs[i], fft[i]))

    # Sort by amplitude and take top 10
    peaks.sort(key=lambda x: -x[1])
    peaks = peaks[:10]

    print(f"\n{name}:")
    print(f"  Total energy: {np.sum(fft ** 2):.2e}")
    print(f"  Peak frequencies:")
    for freq, amp in peaks:
        print(f"    {freq:7.1f} Hz: {amp:10.1f}")


print("=" * 60)
print("FM SPECTRUM COMPARISON")
print("=" * 60)

# Generate without FM
audio_no_fm = generate_audio(fm_depth=0, fm_source=0)
analyze_spectrum(audio_no_fm, "NO FM (pure sine)")

# Generate with 25% FM
audio_fm_25 = generate_audio(fm_depth=8192, fm_source=1)
analyze_spectrum(audio_fm_25, "25% FM FEEDBACK")

# Generate with 50% FM
audio_fm_50 = generate_audio(fm_depth=16384, fm_source=1)
analyze_spectrum(audio_fm_50, "50% FM FEEDBACK")

# Generate with 75% FM
audio_fm_75 = generate_audio(fm_depth=24576, fm_source=1)
analyze_spectrum(audio_fm_75, "75% FM FEEDBACK")

# Waveform comparison
print("\n" + "=" * 60)
print("WAVEFORM STATISTICS")
print("=" * 60)
for name, audio in [
    ("NO FM", audio_no_fm),
    ("25% FM", audio_fm_25),
    ("50% FM", audio_fm_50),
    ("75% FM", audio_fm_75),
]:
    print(f"{name}: min={np.min(audio):6.0f}, max={np.max(audio):6.0f}, std={np.std(audio):6.0f}")
