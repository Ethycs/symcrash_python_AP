"""
Sampler Test with Violin Sample
================================

Tests sample playback mode using a real violin.wav file.
Validates FWPT gateway architecture for audio-domain samples.

Tests:
1. Load violin sample and verify properties
2. Playback at original pitch
3. Pitch shifting across octaves
4. Looped playback
5. Coefficient operator integration (TILT morphing)
6. Write output to WAV for verification
"""

import numpy as np
import pytest
import sys
import os
from pathlib import Path

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import scipy.io.wavfile as wav

from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState, OSC_MODE_SAMPLE
from symcrash_python_ap.vm.sample_source import SampleOscState, BLOCK_SIZE
from symcrash_python_ap.frames import VoiceFrame

SAMPLE_RATE = 48000
PROJECT_ROOT = Path(__file__).parent.parent
VIOLIN_PATH = PROJECT_ROOT / "violin.wav"
OUTPUT_DIR = PROJECT_ROOT / "audio_out" / "sampler_tests"


def load_violin_sample() -> tuple[int, np.ndarray]:
    """Load violin.wav and return (sample_rate, mono_int16_buffer)."""
    if not VIOLIN_PATH.exists():
        pytest.skip(f"violin.wav not found at {VIOLIN_PATH}")

    sr, data = wav.read(str(VIOLIN_PATH))

    # Convert to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Convert to int16 if needed
    if data.dtype == np.float32 or data.dtype == np.float64:
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)

    return sr, data


@pytest.fixture(scope="module")
def violin_sample():
    """Fixture that loads violin sample once for all tests."""
    return load_violin_sample()


@pytest.fixture
def violin_data(violin_sample):
    """Get violin audio data."""
    return violin_sample[1].copy()


@pytest.fixture
def violin_sr(violin_sample):
    """Get violin sample rate."""
    return violin_sample[0]


def write_output(audio: np.ndarray, filename: str):
    """Write output to WAV file for manual verification."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio_norm = (audio.astype(np.float64) / max_val * 28000).astype(np.int16)
    else:
        audio_norm = audio.astype(np.int16)

    wav.write(str(output_path), SAMPLE_RATE, audio_norm)
    print(f"Wrote: {output_path}")


def test_violin_sample_load(violin_sample):
    """Test 1: Load violin sample and verify properties."""
    sr, data = violin_sample
    duration = len(data) / sr

    print(f"\nSample rate: {sr} Hz")
    print(f"Samples: {len(data)}")
    print(f"Duration: {duration:.2f}s")
    print(f"Peak: {np.max(np.abs(data))}")
    print(f"RMS: {np.sqrt(np.mean(data.astype(np.float64)**2)):.1f}")

    assert len(data) > BLOCK_SIZE, "Sample too short"
    assert sr > 0, "Invalid sample rate"


def test_violin_playback_original_pitch(violin_data, violin_sr):
    """Test 2: Playback at original pitch (MIDI 60 = C4)."""
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)  # C4
    state.use_coeff_osc = True

    # Load sample with C4 as root note
    state.sample_state.set_buffer(
        violin_data,
        root_note=60,  # Assume recorded at C4
        loop_enabled=False
    )

    frame = VoiceFrame(
        note=60,  # Play at root
        coeff_osc_mode=OSC_MODE_SAMPLE,
    )

    # Generate blocks for 2 seconds or until sample ends
    max_blocks = int(2.0 * SAMPLE_RATE / BLOCK_SIZE)
    outputs = []

    for i in range(max_blocks):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

        if state.sample_state.is_finished:
            break

    audio = np.concatenate(outputs)

    # Verify output
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    peak = np.max(np.abs(audio))

    print(f"\nOutput blocks: {len(outputs)}")
    print(f"Output samples: {len(audio)}")
    print(f"Output RMS: {rms:.1f}")
    print(f"Output peak: {peak}")

    # Rate should be 1.0x at root note
    assert state.sample_state.rate_q16 == 65536, f"Rate should be 1.0x, got {state.sample_state.rate_q16}"
    assert rms > 100, f"Output too quiet: {rms}"

    write_output(audio, "violin_original_pitch.wav")


def test_violin_pitch_shift(violin_data):
    """Test 3: Pitch shifting across octaves."""
    for semitones, note in [(-12, 48), (-7, 53), (0, 60), (5, 65), (12, 72)]:
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=note)
        state.use_coeff_osc = True
        state.sample_state.set_buffer(violin_data.copy(), root_note=60, loop_enabled=True)

        frame = VoiceFrame(note=note, coeff_osc_mode=OSC_MODE_SAMPLE)

        # Generate 50 blocks
        outputs = []
        for _ in range(50):
            state, output = vm_step_block(state, frame)
            outputs.append(output.copy())

        audio = np.concatenate(outputs)
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        rate = state.sample_state.rate_q16 / 65536
        expected_rate = 2.0 ** (semitones / 12.0)

        print(f"\nNote {note} ({semitones:+d} semitones): rate={rate:.4f}x (expected {expected_rate:.4f}x), RMS={rms:.1f}")

        assert abs(rate - expected_rate) < 0.01, f"Rate mismatch at note {note}"
        assert rms > 50, f"Output too quiet at note {note}"

        if note == 72:
            write_output(audio, "violin_octave_up.wav")
        elif note == 48:
            write_output(audio, "violin_octave_down.wav")


def test_violin_looped_playback(violin_data):
    """Test 4: Looped playback continues beyond sample length."""
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)
    state.use_coeff_osc = True

    # Use only first second of sample for faster looping
    short_buffer = violin_data[:SAMPLE_RATE].copy()
    state.sample_state.set_buffer(short_buffer, root_note=60, loop_enabled=True)

    frame = VoiceFrame(note=60, coeff_osc_mode=OSC_MODE_SAMPLE)

    # Generate 3 seconds of audio (should loop multiple times)
    num_blocks = int(3.0 * SAMPLE_RATE / BLOCK_SIZE)
    outputs = []

    for _ in range(num_blocks):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

    audio = np.concatenate(outputs)

    # Should still be playing (looping)
    assert state.sample_state.is_playing, "Should still be playing"
    assert not state.sample_state.is_finished, "Should not be finished (looping)"

    # Check RMS at start, middle, and end
    third = len(audio) // 3
    rms_start = np.sqrt(np.mean(audio[:third].astype(np.float64) ** 2))
    rms_mid = np.sqrt(np.mean(audio[third:2*third].astype(np.float64) ** 2))
    rms_end = np.sqrt(np.mean(audio[2*third:].astype(np.float64) ** 2))

    print(f"\nRMS: start={rms_start:.1f}, middle={rms_mid:.1f}, end={rms_end:.1f}")

    assert rms_end > 50, f"Loop end too quiet: {rms_end}"

    write_output(audio, "violin_looped.wav")


def test_violin_with_tilt_morphing(violin_data):
    """Test 5: Sample + coefficient operators (brightness/tilt)."""
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)
    state.use_coeff_osc = True
    state.sample_state.set_buffer(violin_data.copy(), root_note=60, loop_enabled=True)

    # Generate with varying tilt
    outputs_dark = []

    # Dark (negative tilt)
    for _ in range(100):
        frame = VoiceFrame(
            note=60,
            coeff_osc_mode=OSC_MODE_SAMPLE,
            tilt_env_q15=8192,  # Low brightness
        )
        state, output = vm_step_block(state, frame)
        outputs_dark.append(output.copy())

    # Reset and do bright
    state2 = VMState.create(voice_id=1)
    state2.reset_for_note_on(note=60)
    state2.use_coeff_osc = True
    state2.sample_state.set_buffer(violin_data.copy(), root_note=60, loop_enabled=True)

    outputs_bright = []
    for _ in range(100):
        frame = VoiceFrame(
            note=60,
            coeff_osc_mode=OSC_MODE_SAMPLE,
            tilt_env_q15=28000,  # High brightness
        )
        state2, output = vm_step_block(state2, frame)
        outputs_bright.append(output.copy())

    audio_dark = np.concatenate(outputs_dark)
    audio_bright = np.concatenate(outputs_bright)

    rms_dark = np.sqrt(np.mean(audio_dark.astype(np.float64) ** 2))
    rms_bright = np.sqrt(np.mean(audio_bright.astype(np.float64) ** 2))

    print(f"\nDark tilt RMS: {rms_dark:.1f}")
    print(f"Bright tilt RMS: {rms_bright:.1f}")

    # Both should have content
    assert rms_dark > 50, f"Dark output too quiet: {rms_dark}"
    assert rms_bright > 50, f"Bright output too quiet: {rms_bright}"

    write_output(audio_dark, "violin_dark.wav")
    write_output(audio_bright, "violin_bright.wav")


def test_violin_precomputed_coefficients(violin_data):
    """Test 6: Pre-computed coefficient mode (FWPT at load time)."""
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)
    state.use_coeff_osc = True

    # Use first 0.5s for precomputed coeffs
    short_buffer = violin_data[:SAMPLE_RATE // 2].copy()
    state.sample_state.set_buffer(
        short_buffer,
        root_note=60,
        precompute_coeffs=True,
        loop_enabled=True
    )

    assert state.sample_state.use_coeff_buffer, "Should use coefficient buffer"
    assert state.sample_state.coeff_buffer is not None, "Coeff buffer should exist"

    frame = VoiceFrame(note=60, coeff_osc_mode=OSC_MODE_SAMPLE)

    # Generate blocks
    outputs = []
    for _ in range(100):
        state, output = vm_step_block(state, frame)
        outputs.append(output.copy())

    audio = np.concatenate(outputs)
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))

    print(f"\nPrecomputed coeff mode RMS: {rms:.1f}")
    assert rms > 500, f"Precomputed mode too quiet: {rms}"

    write_output(audio, "violin_precomputed.wav")


# Entry point for direct execution
def main():
    print("=" * 60)
    print("Sampler Test with Violin Sample")
    print("=" * 60)

    try:
        sr, data = load_violin_sample()
        print(f"\nLoaded violin sample: {sr}Hz, {len(data)} samples")

        # Run tests manually
        test_violin_sample_load((sr, data))
        test_violin_playback_original_pitch(data.copy(), sr)
        test_violin_pitch_shift(data.copy())
        test_violin_looped_playback(data.copy())
        test_violin_with_tilt_morphing(data.copy())
        test_violin_precomputed_coefficients(data.copy())

        print("\n" + "=" * 60)
        print("[PASS] All violin sampler tests passed!")
        return 0

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
