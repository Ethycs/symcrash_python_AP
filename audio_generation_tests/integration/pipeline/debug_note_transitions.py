"""Debug note transition clicks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np

from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude
from symcrash_python_ap.audio_io import AudioWriter

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)

Q15_ONE = 32767
BLOCK_SIZE = 64
FS = 48000
BLOCKS_PER_SEC = FS // BLOCK_SIZE

def generate_partial_amps(wave_type: WaveType, num_partials: int = 8):
    return [get_partial_amplitude(wave_type, k + 1) for k in range(num_partials)]

def midi_to_freq(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))

def main():
    print("=" * 60)
    print("NOTE TRANSITION DEBUG")
    print("=" * 60)

    saw_partials = generate_partial_amps(WaveType.SAW, 8)

    # Test 1: Two notes with voice_on between them (current behavior)
    print("\n1. Two notes with voice_on (CURRENT - causes clicks):")
    pipeline1 = CarrierLiftPipeline()

    samples1 = []
    note1, note2 = 69, 72  # A4 to C5

    # First note
    pipeline1.voice_on(0, note=note1)
    for i in range(50):  # ~67ms
        env = min(1.0, i / 10) * 0.8
        frame = VoiceFrame(
            voice_id=0,
            note=note1,
            f0_hz=midi_to_freq(note1),
            gain_q15=int(Q15_ONE * env),
            beta_target_q15=Q15_ONE // 2,
            partial_amps_q15=saw_partials,
        )
        L, R = pipeline1.render_block(frame)
        samples1.extend(L)
    pipeline1.voice_off(0)

    # Transition point
    transition_idx = len(samples1)
    print(f"  Transition at sample {transition_idx}")

    # Second note - voice_on resets phase
    pipeline1.voice_on(0, note=note2)
    for i in range(50):
        env = min(1.0, i / 10) * 0.8
        frame = VoiceFrame(
            voice_id=0,
            note=note2,
            f0_hz=midi_to_freq(note2),
            gain_q15=int(Q15_ONE * env),
            beta_target_q15=Q15_ONE // 2,
            partial_amps_q15=saw_partials,
        )
        L, R = pipeline1.render_block(frame)
        samples1.extend(L)
    pipeline1.voice_off(0)

    # Check for discontinuity at transition
    before = samples1[transition_idx - 1]
    after = samples1[transition_idx]
    jump1 = abs(after - before)
    print(f"  Before transition: {before}, After: {after}, Jump: {jump1}")

    path = audio_io.write_audio(
        "transition_with_voice_on", samples1, samples1,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"  Saved: {path}")

    # Test 2: Two notes WITHOUT voice_on, just change pitch (legato-style)
    print("\n2. Two notes with pitch change only (LEGATO - no clicks):")
    pipeline2 = CarrierLiftPipeline()

    samples2 = []

    # First note
    pipeline2.voice_on(0, note=note1)
    for i in range(50):
        env = min(1.0, i / 10) * 0.8
        frame = VoiceFrame(
            voice_id=0,
            note=note1,
            f0_hz=midi_to_freq(note1),
            gain_q15=int(Q15_ONE * env),
            beta_target_q15=Q15_ONE // 2,
            partial_amps_q15=saw_partials,
        )
        L, R = pipeline2.render_block(frame)
        samples2.extend(L)

    # Transition point - NO voice_off/voice_on, just change note
    transition_idx2 = len(samples2)
    print(f"  Transition at sample {transition_idx2}")

    # Second note - just change the note in the frame, no voice_on
    for i in range(50):
        env = 0.8  # Keep sustain level
        frame = VoiceFrame(
            voice_id=0,
            note=note2,
            f0_hz=midi_to_freq(note2),
            gain_q15=int(Q15_ONE * env),
            beta_target_q15=Q15_ONE // 2,
            partial_amps_q15=saw_partials,
        )
        L, R = pipeline2.render_block(frame)
        samples2.extend(L)
    pipeline2.voice_off(0)

    # Check for discontinuity at transition
    before2 = samples2[transition_idx2 - 1]
    after2 = samples2[transition_idx2]
    jump2 = abs(after2 - before2)
    print(f"  Before transition: {before2}, After: {after2}, Jump: {jump2}")

    path = audio_io.write_audio(
        "transition_legato_style", samples2, samples2,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"  Saved: {path}")

    # Test 3: Notes with crossfade between them
    print("\n3. Notes with crossfade (release overlap):")
    pipeline3 = CarrierLiftPipeline()

    samples3 = []

    # First note
    pipeline3.voice_on(0, note=note1)
    for i in range(45):  # Short first note
        env = min(1.0, i / 10) * 0.8
        frame = VoiceFrame(
            voice_id=0,
            note=note1,
            f0_hz=midi_to_freq(note1),
            gain_q15=int(Q15_ONE * env),
            beta_target_q15=Q15_ONE // 2,
            partial_amps_q15=saw_partials,
        )
        L, R = pipeline3.render_block(frame)
        samples3.extend(L)

    # Crossfade region - ramp down note1 while ramping up note2 (using just gain)
    crossfade_blocks = 10
    transition_idx3 = len(samples3)
    print(f"  Crossfade starts at sample {transition_idx3}")

    for i in range(crossfade_blocks):
        # Fade out first note
        fade_out = 0.8 * (1.0 - i / crossfade_blocks)
        frame1 = VoiceFrame(
            voice_id=0,
            note=note1,
            f0_hz=midi_to_freq(note1),
            gain_q15=int(Q15_ONE * fade_out),
            beta_target_q15=Q15_ONE // 2,
            partial_amps_q15=saw_partials,
        )
        L1, R1 = pipeline3.render_block(frame1)

        # We can't easily mix two voices here with current setup
        # So this is demonstrating that we need proper crossfade handling
        samples3.extend(L1)

    # Continue with second note (still has a jump here)
    pipeline3.voice_on(0, note=note2)
    for i in range(50):
        env = min(1.0, i / 10) * 0.8
        frame = VoiceFrame(
            voice_id=0,
            note=note2,
            f0_hz=midi_to_freq(note2),
            gain_q15=int(Q15_ONE * env),
            beta_target_q15=Q15_ONE // 2,
            partial_amps_q15=saw_partials,
        )
        L, R = pipeline3.render_block(frame)
        samples3.extend(L)

    path = audio_io.write_audio(
        "transition_with_fade", samples3, samples3,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"  Saved: {path}")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"\nTest 1 (voice_on) jump: {jump1}")
    print(f"Test 2 (legato) jump:   {jump2}")
    print(f"\nThe legato approach should have much smaller jumps.")
    print("Listen to the files to hear the difference.")

if __name__ == "__main__":
    main()
