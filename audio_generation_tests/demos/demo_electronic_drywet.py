"""
Electronic Music Demo - Dry/Wet Blend Showcase
==============================================

Demonstrates the CarrierLift pipeline with varying beta (dry/wet) settings.
Creates 5 audio samples showcasing different blend modes.

Demos:
1. Pure Carrier (beta=0) - crisp additive harmonics
2. Pure Wet (beta=1) - rich wavelet texture
3. Half Blend (beta=0.5) - balanced hybrid
4. Beta Sweep (0->1->0) - morphing timbre
5. Electronic Arpeggio at 140 BPM - musical pattern
"""

import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "symcrash_python_ap"))

from symcrash_python_ap.wavelet.carrier import CarrierOscillator
from symcrash_python_ap.anti_click import AntiClickOperator, AntiClickConfig
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, Q15_HALF, Q15_SCALE, BLOCK_SIZE, SAMPLE_RATE
from symcrash_python_ap.wavelet.harmonic_injector import create_harmonic_coeffs, WaveType

# Constants
FS = SAMPLE_RATE  # 48000
BLOCKS_PER_SEC = FS // BLOCK_SIZE  # 750

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def samples_to_stereo(samples: List[int]) -> Tuple[List[int], List[int]]:
    """Convert mono samples to stereo (identical L/R)."""
    return samples, samples


def save_demo(name: str, samples: List[int]) -> str:
    """Save demo to WAV file."""
    L, R = samples_to_stereo(samples)
    metadata = audio_io.write_audio(name, L, R, subdirectory="demos", timestamp=False)
    return metadata.filepath


def generate_wet_block(note: int, wave_type: WaveType = WaveType.SAW,
                       amplitude: int = Q15_HALF) -> np.ndarray:
    """
    Generate a wavelet-style wet block from harmonic coefficients.

    For demo purposes, this creates a spectrally-rich signal by
    summing harmonics with phase variation to simulate IDWPT output.
    """
    freq = 440.0 * (2.0 ** ((note - 69) / 12.0))

    # Create coefficient-based synthesis
    samples = np.zeros(BLOCK_SIZE, dtype=np.float32)

    # Generate harmonics with slight detuning for richness
    for h in range(1, 17):
        h_freq = freq * h
        if h_freq > FS / 2:
            break

        # Wave-type specific amplitude
        if wave_type == WaveType.SAW:
            h_amp = amplitude / h
        elif wave_type == WaveType.TRIANGLE:
            h_amp = amplitude / (h * h) if h % 2 == 1 else 0
        elif wave_type == WaveType.SQUARE:
            h_amp = amplitude / h if h % 2 == 1 else 0
        else:  # SINE
            h_amp = amplitude if h == 1 else 0

        if h_amp == 0:
            continue

        # Add slight inharmonicity for wavelet-like character
        detune = 1.0 + (h - 1) * 0.001 * math.sin(h * 0.5)
        phase_inc = 2.0 * math.pi * h_freq * detune / FS

        for i in range(BLOCK_SIZE):
            samples[i] += h_amp * math.sin(phase_inc * i + h * 0.1)

    # Normalize
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples * (Q15_HALF / peak)

    return samples.astype(np.float32)


# =============================================================================
# DEMO 1: Pure Carrier (beta=0)
# =============================================================================

def demo_1_pure_carrier():
    """Pure dry carrier - crisp saw with 8 harmonics."""
    print("\n" + "=" * 60)
    print("DEMO 1: Pure Carrier (beta=0)")
    print("=" * 60)

    carrier = CarrierOscillator()

    # Configure saw wave on C3 (MIDI 48)
    saw_amps = [Q15_ONE // n for n in range(1, 9)]
    carrier.configure_voice(0, note=48, harmonic_amps=saw_amps, reset_phase=True)
    carrier.gate_on(0)

    samples = []
    duration_sec = 3.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    for _ in range(num_blocks):
        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)

    filepath = save_demo("demo_1_pure_carrier", samples)
    print(f"  Note: C3 (MIDI 48)")
    print(f"  Waveform: Saw (8 harmonics)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 2: Pure Wet (beta=1)
# =============================================================================

def demo_2_pure_wet():
    """Pure wet path - rich wavelet texture."""
    print("\n" + "=" * 60)
    print("DEMO 2: Pure Wet (beta=1)")
    print("=" * 60)

    # Use anticlick with beta=1 (pure wet)
    anticlick = AntiClickOperator(AntiClickConfig())
    state = anticlick.init_state()
    anticlick.note_on(state)
    anticlick.set_wet_target(state, 1.0)

    # Also render carrier (even though we won't use it much)
    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=48, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    samples = []
    duration_sec = 3.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    for block_idx in range(num_blocks):
        # Render carrier
        carrier_block = np.array(carrier.render_block(0, BLOCK_SIZE), dtype=np.float32)

        # Render wet (richer texture)
        wet_block = generate_wet_block(48, WaveType.SAW, Q15_HALF)

        # Blend with beta=1 (pure wet)
        output = anticlick.step(state, carrier_block, wet_block)
        samples.extend(output.astype(int).tolist())

    filepath = save_demo("demo_2_pure_wet", samples)
    print(f"  Note: C3 (MIDI 48)")
    print(f"  Waveform: Wavelet-rich saw")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 3: Half Blend (beta=0.5)
# =============================================================================

def demo_3_half_blend():
    """50/50 blend of carrier and wet."""
    print("\n" + "=" * 60)
    print("DEMO 3: Half Blend (beta=0.5)")
    print("=" * 60)

    anticlick = AntiClickOperator(AntiClickConfig())
    state = anticlick.init_state()
    anticlick.note_on(state)
    anticlick.set_wet_target(state, 0.5)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=48, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    samples = []
    duration_sec = 3.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    for block_idx in range(num_blocks):
        carrier_block = np.array(carrier.render_block(0, BLOCK_SIZE), dtype=np.float32)
        wet_block = generate_wet_block(48, WaveType.SAW, Q15_HALF)

        output = anticlick.step(state, carrier_block, wet_block)
        samples.extend(output.astype(int).tolist())

    filepath = save_demo("demo_3_half_blend", samples)
    print(f"  Note: C3 (MIDI 48)")
    print(f"  Beta: 0.5 (equal-power blend)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 4: Beta Sweep (0 -> 1 -> 0)
# =============================================================================

def demo_4_beta_sweep():
    """Dynamic beta sweep showing morphing timbre."""
    print("\n" + "=" * 60)
    print("DEMO 4: Beta Sweep (0 -> 1 -> 0)")
    print("=" * 60)

    anticlick = AntiClickOperator(AntiClickConfig(bloom_time_s=0.01))  # Fast response
    state = anticlick.init_state()
    anticlick.note_on(state)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=55, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])  # G3
    carrier.gate_on(0)

    samples = []
    duration_sec = 5.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    for block_idx in range(num_blocks):
        # Calculate beta: 0 -> 1 -> 0 over duration
        progress = block_idx / num_blocks
        if progress < 0.5:
            beta = progress * 2.0  # 0 to 1
        else:
            beta = (1.0 - progress) * 2.0  # 1 to 0

        anticlick.set_wet_target(state, beta)
        # Manually advance beta for demo (fast mode)
        state.beta = beta

        carrier_block = np.array(carrier.render_block(0, BLOCK_SIZE), dtype=np.float32)
        wet_block = generate_wet_block(55, WaveType.SAW, Q15_HALF)

        output = anticlick.step(state, carrier_block, wet_block)
        samples.extend(output.astype(int).tolist())

    filepath = save_demo("demo_4_beta_sweep", samples)
    print(f"  Note: G3 (MIDI 55)")
    print(f"  Beta: 0 -> 1 -> 0 sweep")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 5: Electronic Arpeggio
# =============================================================================

def demo_5_arpeggio():
    """140 BPM electronic arpeggio with dynamic beta."""
    print("\n" + "=" * 60)
    print("DEMO 5: Electronic Arpeggio (140 BPM)")
    print("=" * 60)

    anticlick = AntiClickOperator(AntiClickConfig(bloom_time_s=0.02))
    state = anticlick.init_state()
    anticlick.note_on(state)

    carrier = CarrierOscillator()

    # Arpeggio pattern: C3, E3, G3, C4 (minor chord voicing)
    pattern = [48, 52, 55, 60, 55, 52]  # Up and down

    # 140 BPM = 2.33 beats/sec, 16th notes = 9.33 notes/sec
    bpm = 140
    note_duration_sec = 60.0 / bpm / 4  # 16th notes
    note_blocks = int(note_duration_sec * BLOCKS_PER_SEC)

    samples = []
    duration_sec = 8.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    current_note_idx = 0
    blocks_in_note = 0

    for block_idx in range(num_blocks):
        # Change note?
        if blocks_in_note >= note_blocks:
            blocks_in_note = 0
            current_note_idx = (current_note_idx + 1) % len(pattern)

            # Configure new note
            note = pattern[current_note_idx]
            carrier.configure_voice(0, note=note,
                                   harmonic_amps=[Q15_ONE // n for n in range(1, 9)],
                                   reset_phase=False)  # Keep phase for smoothness
            carrier.gate_on(0)

        blocks_in_note += 1
        note = pattern[current_note_idx]

        # Dynamic beta based on position in arpeggio
        # Higher notes = more wet
        note_progress = (note - 48) / 12.0  # 0 at C3, 1 at C4
        beta = 0.3 + 0.5 * note_progress

        anticlick.set_wet_target(state, beta)
        state.beta = beta  # Fast mode for demo

        carrier_block = np.array(carrier.render_block(0, BLOCK_SIZE), dtype=np.float32)
        wet_block = generate_wet_block(note, WaveType.SAW, Q15_HALF)

        output = anticlick.step(state, carrier_block, wet_block)
        samples.extend(output.astype(int).tolist())

    filepath = save_demo("demo_5_arpeggio", samples)
    print(f"  Pattern: C3-E3-G3-C4-G3-E3")
    print(f"  BPM: 140 (16th notes)")
    print(f"  Beta: Dynamic (0.3-0.8 based on pitch)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ELECTRONIC MUSIC DEMO - DRY/WET BLEND SHOWCASE")
    print("=" * 60)
    print(f"\nSample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Output: audio_out/demos/")

    # Run all demos
    demo_1_pure_carrier()
    demo_2_pure_wet()
    demo_3_half_blend()
    demo_4_beta_sweep()
    demo_5_arpeggio()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
    print("\nFiles saved to: audio_out/demos/")
    print("  - demo_1_pure_carrier.wav")
    print("  - demo_2_pure_wet.wav")
    print("  - demo_3_half_blend.wav")
    print("  - demo_4_beta_sweep.wav")
    print("  - demo_5_arpeggio.wav")


if __name__ == "__main__":
    main()
