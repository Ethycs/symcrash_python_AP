#!/usr/bin/env python3
"""
Generate WAV files for each CCS waveform type.

Creates 2-second samples at A4 (440 Hz) for each waveform:
- sine_440.wav
- saw_440.wav
- square_440.wav
- triangle_440.wav
- pulse_440.wav

Usage:
    python tools/generate_waveform_samples.py
    python tools/generate_waveform_samples.py --note 60  # Middle C
    python tools/generate_waveform_samples.py --duration 5  # 5 seconds
"""

import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.vm.coeff_lut_streaming import (
    WAVE_SINE, WAVE_SAW, WAVE_SQUARE, WAVE_TRIANGLE, WAVE_PULSE,
    BLOCK_SIZE, midi_to_hz, _NOTE_LUT_CACHE
)

SAMPLE_RATE = 48000


def generate_waveform_wav(
    wave_id: int,
    wave_name: str,
    midi_note: int,
    duration_sec: float,
    writer: AudioWriter,
) -> str:
    """Generate a WAV file for a specific waveform."""

    # Clear LUT cache to ensure fresh generation
    _NOTE_LUT_CACHE.clear()

    freq_hz = midi_to_hz(midi_note)
    n_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)

    print(f"Generating {wave_name} at {freq_hz:.1f} Hz ({n_blocks} blocks)...")

    # Create voice frame for this waveform
    frame = VoiceFrame(
        voice_id=0,
        note=midi_note,
        cents=0,
        gain_q15=32767,
        pan_q15=0,
        tree_a_id=wave_id,  # wave_id stored in tree_a_id for CCS
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=0,  # Pure carrier (no CMMM wet path)
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )

    # Create VM state
    vm_state = VMState.create(voice_id=0)
    vm_state.reset_for_note_on(note=midi_note, cents=0)

    # Override wave_id in CCS state
    vm_state.ccs_state.wave_id = wave_id

    # Generate audio
    all_samples = []
    for block_idx in range(n_blocks):
        vm_state, block = vm_step_block(vm_state, frame, DEFAULT_TABLES)
        all_samples.extend(block.tolist())

    # Convert to numpy array
    samples = np.array(all_samples, dtype=np.int16)

    filename = f"{wave_name}_{midi_note}"
    meta = writer.write_audio(filename, samples, timestamp=False, verbose=False)
    print(f"  RMS: {meta.rms_amplitude:.0f}, Peak: {meta.peak_amplitude:.0f}")
    print(f"  Written: {meta.filepath}")
    return meta.filepath


def main():
    parser = argparse.ArgumentParser(description='Generate waveform WAV samples')
    parser.add_argument('--note', type=int, default=69, help='MIDI note (default: 69 = A4)')
    parser.add_argument('--duration', type=float, default=2.0, help='Duration in seconds')
    parser.add_argument('--output', type=str, default='waveform_samples', help='Output directory')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    writer = AudioWriter(output_dir=args.output, sample_rate=SAMPLE_RATE)

    print(f"=== Generating Waveform Samples ===")
    print(f"MIDI Note: {args.note} ({midi_to_hz(args.note):.1f} Hz)")
    print(f"Duration: {args.duration} seconds")
    print(f"Output: {args.output}/")
    print()

    waveforms = [
        (WAVE_SINE, 'sine'),
        (WAVE_SAW, 'saw'),
        (WAVE_SQUARE, 'square'),
        (WAVE_TRIANGLE, 'triangle'),
        (WAVE_PULSE, 'pulse'),
    ]

    generated_files = []
    for wave_id, wave_name in waveforms:
        filepath = generate_waveform_wav(
            wave_id=wave_id,
            wave_name=wave_name,
            midi_note=args.note,
            duration_sec=args.duration,
            writer=writer,
        )
        generated_files.append(filepath)

    print()
    print("=== Generation Complete ===")
    print(f"Generated {len(generated_files)} files in {args.output}/")
    for f in generated_files:
        print(f"  {f}")


if __name__ == '__main__':
    main()
