#!/usr/bin/env python3
"""
Musical FM Brass - Properly voiced brass presets
=================================================

The key to good FM brass:
1. INTEGER RATIOS for modulators (1:1, 2:1, 3:1) - creates harmonic overtones
2. DYNAMIC MOD INDEX - brightness peaks during attack, settles in sustain
3. FORMANT EMPHASIS - brass resonances at 500Hz, 1.5kHz, 3kHz regions
4. MINIMAL CARRIER DETUNE - just enough for life, not beating
5. ATTACK TRANSIENT - brief spike in mod index creates "blat"

DX7 Brass secrets:
- Algorithm 22 is classic brass (4 carriers, 2:1 and 1:1 modulators)
- Modulator envelope faster than carrier = brightness decay
- Feedback adds "buzz" harmonics
- Slight detune on carriers only (not modulators)
"""

import sys
from pathlib import Path
import wave
import struct
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.fm6_state import FM6State, Q15_ONE, Q16_ONE
from symcrash_python_ap.vm.fm6_engine import fm6_generate_block_full
from symcrash_python_ap.vm.fm_algorithms import get_algorithm


def render_musical_brass(
    preset_name: str,
    algorithm: int,
    mod_ratios: list,      # Frequency ratios for each operator
    carrier_levels: list,  # Carrier output levels
    mod_depths: list,      # Modulator depths (will be enveloped)
    feedback: float,       # Feedback amount (0-1)
    attack_brightness: float,  # Extra mod depth during attack (1.0 = 2x brightness)
    carrier_detune_cents: list,  # Per-carrier detune
    duration_s: float = 2.0,
    note: int = 60,
) -> np.ndarray:
    """
    Render musically-voiced FM brass.

    Key difference from mechanical rendering:
    - Modulator envelope decays faster than carrier = brightness decay
    - Attack has boosted mod index for "blat"
    """
    sample_rate = 48000
    num_blocks = int(duration_s * sample_rate / 64)

    state = FM6State()
    state.set_algorithm(algorithm)
    alg = get_algorithm(algorithm)

    # Configure frequency ratios
    for i, ratio in enumerate(mod_ratios):
        # Apply detune only to carriers
        if alg.is_carrier(i) and i < len(carrier_detune_cents):
            cents = carrier_detune_cents[i] if i < len(carrier_detune_cents) else 0
            ratio_adjusted = ratio * (2 ** (cents / 1200))
        else:
            ratio_adjusted = ratio
        state.operators[i].ratio_q16 = int(ratio_adjusted * Q16_ONE)

    # Configure base levels
    for i in range(6):
        if alg.is_carrier(i):
            level = carrier_levels[i] if i < len(carrier_levels) else 0.8
        else:
            level = mod_depths[i] if i < len(mod_depths) else 0.5
        state.operators[i].level_q15 = int(level * Q15_ONE)

    # Configure feedback
    if alg.feedback_op >= 0:
        state.operators[alg.feedback_op].feedback_q15 = int(feedback * Q15_ONE)

    # Phase increment for note
    freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
    phase_inc = int((freq / sample_rate) * (2**32))

    # Envelope timing (in blocks)
    attack_time = 0.012  # 12ms attack
    decay_time = 0.25    # 250ms decay to sustain
    sustain = 0.7        # 70% sustain level
    release_time = 0.3   # 300ms release

    mod_decay_time = 0.08  # Modulator decays faster (80ms)
    mod_sustain = 0.4      # Modulator sustains lower

    attack_blocks = int(attack_time * sample_rate / 64)
    decay_blocks = int(decay_time * sample_rate / 64)
    mod_decay_blocks = int(mod_decay_time * sample_rate / 64)
    release_start = num_blocks - int(release_time * sample_rate / 64)

    audio_blocks = []

    for block in range(num_blocks):
        block_time = block * 64 / sample_rate

        # Carrier envelope (ADSR)
        if block < attack_blocks:
            carrier_env = block / max(attack_blocks, 1)
        elif block < attack_blocks + decay_blocks:
            t = (block - attack_blocks) / max(decay_blocks, 1)
            carrier_env = 1.0 - (1.0 - sustain) * t
        elif block < release_start:
            carrier_env = sustain
        else:
            t = (block - release_start) / max(num_blocks - release_start, 1)
            carrier_env = sustain * (1.0 - t)

        # Modulator envelope (faster decay = brightness decay)
        if block < attack_blocks:
            # Attack: boost mod index for "blat"
            mod_env = (block / max(attack_blocks, 1)) * (1.0 + attack_brightness)
        elif block < attack_blocks + mod_decay_blocks:
            # Fast decay from boosted attack
            t = (block - attack_blocks) / max(mod_decay_blocks, 1)
            start_level = 1.0 + attack_brightness
            mod_env = start_level - (start_level - mod_sustain) * t
        elif block < release_start:
            mod_env = mod_sustain
        else:
            t = (block - release_start) / max(num_blocks - release_start, 1)
            mod_env = mod_sustain * (1.0 - t)

        # Apply envelopes to operators
        for i in range(6):
            if alg.is_carrier(i):
                env = carrier_env
            else:
                env = mod_env

            base_level = state.operators[i].level_q15 / Q15_ONE
            effective_level = min(1.0, base_level * env)
            state.operators[i].env_value_q15 = int(effective_level * Q15_ONE)

        # Generate audio
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
    output_dir = Path(__file__).parent.parent.parent / "audio_out" / "musical_brass"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Musical FM Brass Presets")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print()

    presets = [
        # =====================================================================
        # PRESET 1: Classic DX7 Brass (Algorithm 22)
        # 4 carriers with slight detune, 2:1 and 1:1 modulators
        # =====================================================================
        {
            "name": "dx7_brass",
            "algorithm": 22,
            "mod_ratios": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],  # 2:1 mod on Op6
            "carrier_levels": [1.0, 0.85, 0.85, 0.75, 0.0, 0.0],
            "mod_depths": [0.0, 0.0, 0.0, 0.0, 0.6, 0.7],  # Op5, Op6 are mods
            "feedback": 0.15,
            "attack_brightness": 0.8,  # 80% extra brightness on attack
            "carrier_detune_cents": [0, 2, -2, 4, 0, 0],  # Subtle detune
        },

        # =====================================================================
        # PRESET 2: Mellow Brass Section (Algorithm 19)
        # More carriers, less modulation = warmer
        # =====================================================================
        {
            "name": "mellow_section",
            "algorithm": 19,
            "mod_ratios": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # All 1:1
            "carrier_levels": [1.0, 0.9, 0.9, 0.8, 0.0, 0.0],
            "mod_depths": [0.0, 0.0, 0.0, 0.0, 0.4, 0.3],  # Lower mod
            "feedback": 0.08,
            "attack_brightness": 0.4,  # Softer attack
            "carrier_detune_cents": [0, 3, -3, 5, 0, 0],
        },

        # =====================================================================
        # PRESET 3: Bright Trumpet (Algorithm 16)
        # Single carrier with 5 modulators = very bright
        # =====================================================================
        {
            "name": "bright_trumpet",
            "algorithm": 16,  # 5 mods into 1 carrier
            "mod_ratios": [1.0, 1.0, 2.0, 3.0, 4.0, 2.0],  # Harmonic series
            "carrier_levels": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Only Op1 carrier
            "mod_depths": [0.0, 0.5, 0.4, 0.3, 0.2, 0.5],  # Decreasing harmonics
            "feedback": 0.2,
            "attack_brightness": 1.2,  # Very bright attack
            "carrier_detune_cents": [0, 0, 0, 0, 0, 0],  # No detune (solo)
        },

        # =====================================================================
        # PRESET 4: French Horn (Algorithm 22)
        # Darker, more fundamental, less edge
        # =====================================================================
        {
            "name": "french_horn",
            "algorithm": 22,
            "mod_ratios": [1.0, 1.0, 1.0, 1.0, 0.5, 1.0],  # Sub-harmonic on Op5
            "carrier_levels": [1.0, 0.7, 0.7, 0.5, 0.0, 0.0],
            "mod_depths": [0.0, 0.0, 0.0, 0.0, 0.5, 0.3],  # Lower mod
            "feedback": 0.05,  # Very little feedback
            "attack_brightness": 0.3,  # Soft attack
            "carrier_detune_cents": [0, 1, -1, 2, 0, 0],  # Very subtle
        },

        # =====================================================================
        # PRESET 5: Trombone (Algorithm 22)
        # Low brass character - more fundamental, smooth slide feel
        # =====================================================================
        {
            "name": "trombone",
            "algorithm": 22,
            "mod_ratios": [1.0, 1.0, 1.0, 1.0, 1.0, 1.5],  # 3:2 ratio
            "carrier_levels": [1.0, 0.8, 0.6, 0.4, 0.0, 0.0],  # Decreasing
            "mod_depths": [0.0, 0.0, 0.0, 0.0, 0.55, 0.45],
            "feedback": 0.12,
            "attack_brightness": 0.6,
            "carrier_detune_cents": [0, 2, -2, 3, 0, 0],
        },

        # =====================================================================
        # PRESET 6: Synth Brass Stab (Algorithm 19)
        # Punchy, aggressive attack
        # =====================================================================
        {
            "name": "synth_stab",
            "algorithm": 19,
            "mod_ratios": [1.0, 1.0, 1.0, 1.0, 2.0, 3.0],  # Brighter mods
            "carrier_levels": [1.0, 1.0, 1.0, 0.9, 0.0, 0.0],  # All carriers hot
            "mod_depths": [0.0, 0.0, 0.0, 0.0, 0.7, 0.6],  # Higher mod
            "feedback": 0.25,  # More feedback for edge
            "attack_brightness": 1.5,  # Very bright attack
            "carrier_detune_cents": [0, 5, -5, 8, 0, 0],  # More detune for width
        },

        # =====================================================================
        # PRESET 7: Warm Pad Brass (Algorithm 32)
        # Additive-style, no modulation, just detuned carriers
        # =====================================================================
        {
            "name": "warm_pad",
            "algorithm": 32,  # All 6 carriers (additive)
            "mod_ratios": [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],  # Harmonic stack
            "carrier_levels": [1.0, 0.9, 0.5, 0.4, 0.25, 0.15],  # Rolloff
            "mod_depths": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # No modulation
            "feedback": 0.02,  # Tiny feedback for warmth
            "attack_brightness": 0.0,  # No brightness change
            "carrier_detune_cents": [0, 3, -3, 5, -5, 7],  # All detuned
        },

        # =====================================================================
        # PRESET 8: Jump Brass (Algorithm 19)
        # Van Halen style - thick unison
        # =====================================================================
        {
            "name": "jump_brass",
            "algorithm": 19,
            "mod_ratios": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # All 1:1
            "carrier_levels": [1.0, 1.0, 1.0, 0.9, 0.0, 0.0],
            "mod_depths": [0.0, 0.0, 0.0, 0.0, 0.35, 0.25],  # Subtle mod
            "feedback": 0.08,
            "attack_brightness": 0.5,
            "carrier_detune_cents": [0, 4, -4, 7, 0, 0],  # Ensemble detune
        },
    ]

    for preset in presets:
        name = preset["name"]
        filepath = output_dir / f"{name}.wav"

        print(f"Rendering {name:20s} (alg {preset['algorithm']:2d}) ... ", end="", flush=True)

        try:
            audio = render_musical_brass(
                preset_name=name,
                algorithm=preset["algorithm"],
                mod_ratios=preset["mod_ratios"],
                carrier_levels=preset["carrier_levels"],
                mod_depths=preset["mod_depths"],
                feedback=preset["feedback"],
                attack_brightness=preset["attack_brightness"],
                carrier_detune_cents=preset["carrier_detune_cents"],
            )

            save_wav(audio, filepath)

            rms = np.sqrt(np.mean(audio.astype(float)**2))
            peak = max(abs(audio.max()), abs(audio.min()))
            print(f"OK  (peak:{peak:5d}, rms:{rms:6.0f})")

        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print("Musical brass presets complete!")
    print()
    print("Key differences from mechanical rendering:")
    print("  - Modulator envelope decays faster than carrier (brightness decay)")
    print("  - Attack has boosted mod index ('blat' transient)")
    print("  - Integer ratios for harmonic FM (2:1, 3:1, etc.)")
    print("  - Minimal carrier detune (warmth without beating)")


if __name__ == "__main__":
    main()
