#!/usr/bin/env python3
"""
FM6 Brass Suite - Render all 32 algorithms with brass settings
===============================================================

Renders each DX7 algorithm with consistent brass-like operator settings
to compare their timbral character.

Output: audio_out/brass_suite/alg_XX_<name>.wav
"""

import sys
from pathlib import Path
import wave
import struct
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.fm6_state import FM6State, Q15_ONE
from symcrash_python_ap.vm.fm6_engine import fm6_generate_block_full
from symcrash_python_ap.vm.fm_algorithms import ALGORITHMS, get_algorithm


def render_algorithm(alg_id: int, duration_s: float = 1.5, note: int = 60) -> np.ndarray:
    """
    Render FM6 audio for a specific algorithm with brass settings.

    Args:
        alg_id: Algorithm number (1-32)
        duration_s: Duration in seconds
        note: MIDI note number (60 = C4)

    Returns:
        Audio samples as int16 array
    """
    sample_rate = 48000
    num_blocks = int(duration_s * sample_rate / 64)

    # Initialize FM6 state with the algorithm
    state = FM6State()
    state.set_algorithm(alg_id)

    alg = get_algorithm(alg_id)

    # Configure operator ratios - slight detune for warmth (reduced to avoid beating)
    # Carriers get detune, modulators stay at 1:1
    for i, op in enumerate(state.operators):
        if alg.is_carrier(i):
            # Carriers: slight detune spread
            detune_cents = [-3, 3, -5, 5, -2, 2][i]
            # Q16: 65536 = 1.0, ~38 per cent
            op.ratio_q16 = 65536 + int(detune_cents * 38)
        else:
            # Modulators: integer ratio for harmonic FM
            op.ratio_q16 = 65536  # 1:1

    # Configure operator levels based on role
    for i, op in enumerate(state.operators):
        if alg.is_carrier(i):
            # Carriers: full level for output
            op.level_q15 = Q15_ONE
        elif alg.feedback_op == i:
            # Feedback operator: moderate level
            op.level_q15 = Q15_ONE // 4
        else:
            # Modulators: reduced level for warmth
            op.level_q15 = Q15_ONE // 3

    # Set feedback on the designated operator
    if alg.feedback_op >= 0:
        state.operators[alg.feedback_op].feedback_q15 = Q15_ONE // 8  # Light feedback

    # Calculate phase increment for the note
    freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
    phase_inc = int((freq / sample_rate) * (2**32))

    # Generate audio with ADSR-like envelope
    attack_blocks = int(0.008 * sample_rate / 64)   # 8ms attack
    decay_blocks = int(0.15 * sample_rate / 64)     # 150ms decay
    sustain_level = Q15_ONE * 6 // 10               # 60% sustain
    release_blocks = int(0.2 * sample_rate / 64)    # 200ms release

    audio_blocks = []

    for block in range(num_blocks):
        # Simple ADSR envelope
        if block < attack_blocks:
            # Attack phase
            env = (block * Q15_ONE) // max(attack_blocks, 1)
        elif block < attack_blocks + decay_blocks:
            # Decay phase
            decay_progress = block - attack_blocks
            env = Q15_ONE - ((Q15_ONE - sustain_level) * decay_progress) // max(decay_blocks, 1)
        elif block < num_blocks - release_blocks:
            # Sustain phase
            env = sustain_level
        else:
            # Release phase
            release_progress = block - (num_blocks - release_blocks)
            env = sustain_level - (sustain_level * release_progress) // max(release_blocks, 1)

        # Apply envelope to all operators
        for op in state.operators:
            op.env_value_q15 = max(0, min(Q15_ONE, env))

        # Generate audio block
        block_audio = fm6_generate_block_full(state, phase_inc)
        audio_blocks.append(block_audio)

    return np.concatenate(audio_blocks)


def save_wav(audio: np.ndarray, path: Path, sample_rate: int = 48000):
    """Save audio to WAV file."""
    with wave.open(str(path), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f'<{len(audio)}h', *audio.tolist()))


def main():
    output_dir = Path(__file__).parent.parent.parent / "audio_out" / "brass_suite"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("FM6 Brass Suite - All 32 Algorithms")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()

    # Select algorithms to render
    # Default: all 32, or specify subset for brass-focused comparison
    brass_algorithms = list(range(1, 33))  # All 32

    # Could also focus on brass-relevant algorithms:
    # brass_algorithms = [1, 5, 16, 17, 18, 19, 22, 28, 32]

    for alg_id in brass_algorithms:
        alg = get_algorithm(alg_id)

        # Sanitize name for filename
        safe_name = alg.name.lower().replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        filename = f"alg_{alg_id:02d}_{safe_name}.wav"
        filepath = output_dir / filename

        print(f"Rendering Algorithm {alg_id:2d}: {alg.name:30s} ... ", end="", flush=True)

        try:
            audio = render_algorithm(alg_id)
            save_wav(audio, filepath)

            # Stats
            rms = np.sqrt(np.mean(audio.astype(float)**2))
            peak = max(abs(audio.max()), abs(audio.min()))

            print(f"OK  (peak:{peak:5d}, rms:{rms:6.0f})")
        except Exception as e:
            print(f"FAIL: {e}")

    print()
    print("=" * 70)
    print(f"Brass suite complete: {len(brass_algorithms)} algorithms rendered")
    print(f"Output: {output_dir}")

    # Print algorithm characteristics summary
    print()
    print("Algorithm Characteristics:")
    print("-" * 70)
    print(f"{'ID':>3} {'Name':30} {'Carriers':10} {'Modulators':10} {'FB Op'}")
    print("-" * 70)
    for alg_id in brass_algorithms:
        alg = get_algorithm(alg_id)
        carriers = alg.get_carriers()
        num_mods = 6 - len(carriers)
        fb = f"Op{alg.feedback_op + 1}" if alg.feedback_op >= 0 else "None"
        print(f"{alg_id:3d} {alg.name:30} {len(carriers):10d} {num_mods:10d} {fb}")


if __name__ == "__main__":
    main()
