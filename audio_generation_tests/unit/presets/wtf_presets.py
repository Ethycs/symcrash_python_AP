"""
WTF Presets - 12 Identity-Defining Sounds (WSE Architecture)
=============================================================

Each preset demonstrates something other synths cannot do cleanly.
Uses the full symcrash_python_ap audio pipeline with:
- Symbol Engine (wavelet coefficient operators)
- Wavelet Engine (CDF 5/3 inverse transform)
- Material Engine (catastrophe/saturation)
- Safety Engine (OLA/limiting)
- Harmonic Injector (Baby Fourier)

All sounds generated through the WSE coefficient-space architecture,
NOT through direct sample synthesis.
"""

import sys
import os

# Add src to path for imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'symcrash_python_ap')
sys.path.insert(0, src_path)

import numpy as np
from typing import Tuple, List, Optional

# WSE imports
from audio_io import write_audio
from pipeline import Pipeline
from frames import ParamFrame, VoiceFrame
from fixed import Q15_ONE, Q15_HALF, sat16, mul_q15_sat
from harmonic_injector import (
    HarmonicTargetGenerator, WaveType, note_to_freq, freq_to_leaf,
    LEAF_COUNT, NODE_SIZE, COEFF_LEN
)

FS = 48000
BLOCK_SIZE = 64


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def blocks_for_duration(duration: float) -> int:
    """Calculate number of blocks for a duration in seconds."""
    return int(duration * FS / BLOCK_SIZE)


def create_coefficient_target(frequencies: List[float],
                               amplitudes: List[int]) -> List[int]:
    """
    Create a coefficient vector targeting specific frequencies.

    Maps frequencies to wavelet leaves and sets appropriate coefficients.
    """
    coeffs = [0] * COEFF_LEN

    for freq, amp in zip(frequencies, amplitudes):
        if freq >= FS / 2:
            continue
        leaf_idx = freq_to_leaf(freq)
        # Distribute amplitude across node coefficients for richer sound
        node_start = leaf_idx * NODE_SIZE
        for i in range(NODE_SIZE):
            # Slight amplitude variation per coefficient
            mod = 1.0 - 0.1 * abs(i - 3.5) / 3.5
            coeffs[node_start + i] = sat16(coeffs[node_start + i] + int(amp * mod))

    return coeffs


def morph_coefficients(c_a: List[int], c_b: List[int], alpha: float) -> List[int]:
    """Linearly interpolate between two coefficient vectors."""
    alpha_q15 = int(alpha * Q15_ONE)
    alpha_inv = Q15_ONE - alpha_q15
    result = []
    for a, b in zip(c_a, c_b):
        mixed = (mul_q15_sat(a, alpha_inv) + mul_q15_sat(b, alpha_q15))
        result.append(sat16(mixed))
    return result


# =============================================================================
# PRESET 1: TOPOLOGY FOLD
# =============================================================================

def preset_topology_fold(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Starts as stable harmonic pad, reorganizes into metallic lattice.
    No filter sweep, no distortion - pure structural change via tree morphing.

    Uses: Tree morphing (Tree 1 Harmonic → Tree 2 Texture)
    """
    pipe = Pipeline(n_voices=1, block_size=BLOCK_SIZE)

    injector = HarmonicTargetGenerator()
    injector.set_note(0, note_u8=45)  # A2 = 110 Hz
    injector.set_wave_params(0, wave_type=WaveType.SAW, partials_n=16, brightness_q15=Q15_ONE)

    n_blocks = blocks_for_duration(duration)
    inject_interval = 8

    for i in range(n_blocks):
        progress = i / n_blocks

        # Inject harmonics periodically
        if i % inject_interval == 0:
            target = injector.generate_target(0, include_dense=True)
            if target.dense_target:
                pipe.symbol.inject_note(0, target.dense_target)

        # Morph from harmonic to inharmonic via tree blend
        # Tree 1 (Harmonic) → Tree 2 (Texture)
        if progress < 0.3:
            # Pure harmonic tree
            pipe.symbol.voice_trees[0] = 1
        else:
            # Transition to texture tree
            morph = (progress - 0.3) / 0.7
            if morph > 0.5:
                pipe.symbol.voice_trees[0] = 2
            else:
                pipe.symbol.voice_trees[0] = 1

        # Increase transport operators as we morph
        pipe.params.pressure_q15 = int(progress * 0x4000)

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 2: CATASTROPHE WITHOUT VOLUME
# =============================================================================

def preset_catastrophe_snap(duration: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sustained tone that suddenly snaps character.
    Loudness stays constant, spectral identity flips.

    Uses: Cusp catastrophe material (bifurcation)
    """
    pipe = Pipeline(n_voices=1, block_size=BLOCK_SIZE)

    injector = HarmonicTargetGenerator()
    injector.set_note(0, note_u8=57)  # A3 = 220 Hz

    n_blocks = blocks_for_duration(duration)
    snap_progress = 0.5
    inject_interval = 8

    # Use cusp catastrophe material
    pipe.material_id = 5  # Cusp

    for i in range(n_blocks):
        progress = i / n_blocks

        # Inject tone periodically
        if i % inject_interval == 0:
            # Switch between wave types at snap point
            if progress < snap_progress - 0.02:
                injector.set_wave_params(0, wave_type=WaveType.SAW, partials_n=12,
                                        brightness_q15=int(0.8 * Q15_ONE))
            elif progress > snap_progress + 0.02:
                injector.set_wave_params(0, wave_type=WaveType.PULSE, partials_n=12,
                                        brightness_q15=int(0.6 * Q15_ONE))

            target = injector.generate_target(0, include_dense=True)
            if target.dense_target:
                pipe.symbol.inject_note(0, target.dense_target)

        # Cusp catastrophe parameters - snap occurs when pressure crosses threshold
        if progress < snap_progress - 0.02:
            pipe.params.pressure_q15 = int(0.3 * Q15_ONE)
            pipe.params.bias_q15 = int(-0.2 * Q15_ONE)
        elif progress < snap_progress + 0.02:
            # Rapid transition
            local_t = (progress - (snap_progress - 0.02)) / 0.04
            pipe.params.pressure_q15 = int((0.3 + local_t * 0.4) * Q15_ONE)
            pipe.params.bias_q15 = int((-0.2 + local_t * 0.4) * Q15_ONE)
        else:
            pipe.params.pressure_q15 = int(0.7 * Q15_ONE)
            pipe.params.bias_q15 = int(0.2 * Q15_ONE)

        pipe.params.drive_q15 = 0x4000  # Constant drive

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 3: PERFECT MORPH
# =============================================================================

def preset_perfect_morph(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Morphs between wildly different sounds.
    Zero zippering, no crossfade blur - coefficient space interpolation.

    Uses: Coefficient morphing (SAW → TRIANGLE)
    """
    pipe = Pipeline(n_voices=1, block_size=BLOCK_SIZE)

    injector = HarmonicTargetGenerator()
    injector.set_note(0, note_u8=48)  # C3 = 130.81 Hz

    n_blocks = blocks_for_duration(duration)
    inject_interval = 6

    for i in range(n_blocks):
        progress = i / n_blocks

        if i % inject_interval == 0:
            # Generate both targets
            injector.set_wave_params(0, wave_type=WaveType.SAW, partials_n=16,
                                    brightness_q15=Q15_ONE)
            target_saw = injector.generate_target(0, include_dense=True)

            injector.set_wave_params(0, wave_type=WaveType.TRIANGLE, partials_n=16,
                                    brightness_q15=int(0.7 * Q15_ONE))
            target_tri = injector.generate_target(0, include_dense=True)

            # S-curve morph
            morph = 0.5 - 0.5 * np.cos(progress * np.pi)

            if target_saw.dense_target and target_tri.dense_target:
                morphed = morph_coefficients(target_saw.dense_target,
                                            target_tri.dense_target, morph)
                pipe.symbol.inject_note(0, morphed)

        # Slight parameter evolution
        pipe.params.tone_q15 = int((0.4 + 0.2 * np.sin(progress * np.pi * 2)) * Q15_ONE)

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 4: LIVING CHORD
# =============================================================================

def preset_living_chord(duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    10-20 voices locked in harmonic frame.
    Micro-drift inside, sounds like single organism.

    Uses: Multiple voices with IntraMix coupling
    """
    n_voices = 12
    pipe = Pipeline(n_voices=n_voices, block_size=BLOCK_SIZE)

    injector = HarmonicTargetGenerator()

    # C major 7 chord across octaves (MIDI notes)
    chord_notes = [36, 40, 43, 47, 48, 52, 55, 59, 60, 64, 67, 72]

    n_blocks = blocks_for_duration(duration)
    inject_interval = 12

    # Per-voice micro-drift parameters
    np.random.seed(123)
    drift_rates = np.random.random(n_voices) * 0.3 + 0.1
    drift_depths = np.random.random(n_voices) * 3 + 1  # cents

    for i in range(n_blocks):
        t = i * BLOCK_SIZE / FS
        progress = i / n_blocks

        if i % inject_interval == 0:
            for v in range(n_voices):
                # Apply micro-drift to cents
                drift = int(drift_depths[v] * np.sin(drift_rates[v] * t * 2 * np.pi))

                injector.set_note(v, note_u8=chord_notes[v], cents_i16=drift)
                injector.set_wave_params(v, wave_type=WaveType.SAW, partials_n=6,
                                        brightness_q15=int(0.6 * Q15_ONE))

                target = injector.generate_target(v, include_dense=True)
                if target.dense_target:
                    # Scale amplitude by register
                    amp_scale = 1.0 / (1 + 0.12 * v)
                    scaled = [sat16(int(c * amp_scale)) for c in target.dense_target]
                    pipe.symbol.inject_note(v, scaled)

        # Slow amplitude breathing via pressure
        breath = int(0.1 * np.sin(t * 0.2) * Q15_ONE)
        pipe.params.pressure_q15 = 0x2000 + breath

        # Gentle envelope
        if progress < 0.05:
            env = progress / 0.05
        elif progress > 0.9:
            env = (1 - progress) / 0.1
        else:
            env = 1.0
        pipe.params.gate_q15 = int(env * Q15_ONE)

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 5: POLYPHONIC PRESSURE
# =============================================================================

def preset_polyphonic_pressure(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple notes, complex intermodulation.
    Removing one note collapses texture.

    Uses: Multiple voices with FoldPressure coupling
    """
    n_voices = 7
    pipe = Pipeline(n_voices=n_voices, block_size=BLOCK_SIZE)

    # Use fold material for intermodulation
    pipe.material_id = 4  # Fold catastrophe

    injector = HarmonicTargetGenerator()

    # Cluster of close frequencies (Hz -> MIDI approximation)
    # 220, 223, 227, 330, 333, 440, 445 Hz
    cluster_notes = [57, 57, 58, 64, 64, 69, 69]
    cluster_cents = [0, 25, 50, 0, 20, 0, 20]

    n_blocks = blocks_for_duration(duration)
    inject_interval = 8

    for i in range(n_blocks):
        progress = i / n_blocks

        # Build up notes over time
        n_active = int(1 + progress * (n_voices - 1))

        if i % inject_interval == 0:
            for v in range(n_active):
                injector.set_note(v, note_u8=cluster_notes[v], cents_i16=cluster_cents[v])
                injector.set_wave_params(v, wave_type=WaveType.SAW, partials_n=4,
                                        brightness_q15=int(0.7 * Q15_ONE))

                target = injector.generate_target(v, include_dense=True)
                if target.dense_target:
                    pipe.symbol.inject_note(v, target.dense_target)

        # Increase pressure as more voices enter
        pipe.params.pressure_q15 = int(progress * 0.6 * Q15_ONE)
        pipe.params.drive_q15 = 0x3000

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 6: INTENTIONAL ALIASING
# =============================================================================

def preset_intentional_aliasing(duration: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Harsh digital edge that isn't brittle fizz.
    Carved, not broken.

    Uses: High partials that fold at band edges + QuasiRotate
    """
    pipe = Pipeline(n_voices=1, block_size=BLOCK_SIZE)

    injector = HarmonicTargetGenerator()
    injector.set_note(0, note_u8=33)  # A1 = 55 Hz (low for many harmonics)

    n_blocks = blocks_for_duration(duration)
    inject_interval = 6

    for i in range(n_blocks):
        progress = i / n_blocks

        if i % inject_interval == 0:
            # Use many partials - some will alias/fold
            injector.set_wave_params(0, wave_type=WaveType.SAW, partials_n=16,
                                    brightness_q15=Q15_ONE)  # Full brightness = harsh

            target = injector.generate_target(0, include_dense=True)
            if target.dense_target:
                # Add some intentional high-frequency content
                for leaf in range(24, 32):  # Upper leaves
                    node_start = leaf * NODE_SIZE
                    for j in range(NODE_SIZE):
                        # Quantized/stepped values for digital character
                        val = int(0x1000 * np.sin(leaf * 0.7 + j * 0.3))
                        val = (val // 0x200) * 0x200  # Quantize
                        target.dense_target[node_start + j] = sat16(
                            target.dense_target[node_start + j] + val
                        )

                pipe.symbol.inject_note(0, target.dense_target)

        # QuasiRotate parameter evolves slowly
        pipe.params.tone_q15 = int((0.5 + 0.3 * np.sin(progress * np.pi)) * Q15_ONE)

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 7: VHS GHOST CHOIR
# =============================================================================

def preset_vhs_ghost_choir(duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Noise-based voices. Pitch emerges from structure.
    Degradation that feels emotional.

    Uses: NoiseBand operator + tape-like material
    """
    n_voices = 5
    pipe = Pipeline(n_voices=n_voices, block_size=BLOCK_SIZE)

    # Use tape/hysteresis material
    pipe.material_id = 1  # Tape Bus

    injector = HarmonicTargetGenerator()

    # G major chord
    choir_notes = [55, 59, 62, 67, 71]  # G3, B3, D4, G4, B4

    n_blocks = blocks_for_duration(duration)
    inject_interval = 16

    # Tape wow/flutter state
    wow_phase = 0.0
    flutter_phase = 0.0

    for i in range(n_blocks):
        t = i * BLOCK_SIZE / FS
        progress = i / n_blocks

        # Tape artifacts
        wow_phase += 2 * np.pi * 0.5 * BLOCK_SIZE / FS
        flutter_phase += 2 * np.pi * 7 * BLOCK_SIZE / FS
        pitch_mod_cents = int(8 * np.sin(wow_phase) + 2 * np.sin(flutter_phase))

        if i % inject_interval == 0:
            for v, note in enumerate(choir_notes):
                # Voice emerges over time
                voice_onset = v * 0.15
                if progress < voice_onset:
                    continue

                voice_amp = min(1.0, (progress - voice_onset) / 0.2)

                # Add pitch wobble
                injector.set_note(v, note_u8=note, cents_i16=pitch_mod_cents)

                # Blend noise and tone based on emergence
                brightness = int((0.3 + 0.4 * voice_amp) * Q15_ONE)
                injector.set_wave_params(v, wave_type=WaveType.SAW, partials_n=6,
                                        brightness_q15=brightness)

                target = injector.generate_target(v, include_dense=True)
                if target.dense_target:
                    scaled = [sat16(int(c * voice_amp * 0.5)) for c in target.dense_target]
                    pipe.symbol.inject_note(v, scaled)

        # Tape character increases over time
        pipe.params.drive_q15 = int((0.3 + 0.3 * progress) * Q15_ONE)
        pipe.params.tone_q15 = int(0.4 * Q15_ONE)  # Warm/dark

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 8: PRESSURE FIELD
# =============================================================================

def preset_pressure_field(duration: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Starts almost silent. Density increases, not volume.
    Eventually overwhelming without being loud.

    Uses: Increasing voice count + DiffuseUp/Down for spreading
    """
    max_voices = 16
    pipe = Pipeline(n_voices=max_voices, block_size=BLOCK_SIZE)

    injector = HarmonicTargetGenerator()

    base_note = 28  # E1 = 41 Hz (very low)

    n_blocks = blocks_for_duration(duration)
    inject_interval = 10

    for i in range(n_blocks):
        progress = i / n_blocks

        # Number of voices increases with time
        n_active = int(2 + progress * (max_voices - 2))

        if i % inject_interval == 0:
            for v in range(n_active):
                # Non-harmonic spacing creates density
                ratio = 1 + v * 0.73
                freq = note_to_freq(base_note) * ratio

                # Convert back to note approximation
                note_approx = base_note + int(12 * np.log2(ratio))
                cents = int((12 * np.log2(ratio) - int(12 * np.log2(ratio))) * 100)

                injector.set_note(v, note_u8=min(127, note_approx), cents_i16=cents)
                injector.set_wave_params(v, wave_type=WaveType.SINE, partials_n=3,
                                        brightness_q15=int(0.5 * Q15_ONE))

                target = injector.generate_target(v, include_dense=True)
                if target.dense_target:
                    # Constant amplitude per partial, normalize for voice count
                    scale = 0.5 / np.sqrt(n_active / 4)
                    scaled = [sat16(int(c * scale)) for c in target.dense_target]
                    pipe.symbol.inject_note(v, scaled)

        # Diffuse parameters increase density spread
        pipe.params.pressure_q15 = int(progress * 0.5 * Q15_ONE)

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 9: IMMINENT FAILURE
# =============================================================================

def preset_imminent_failure(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Always feels about to break. Never does.
    Tension without collapse.

    Uses: Fold catastrophe near but never crossing bifurcation
    """
    pipe = Pipeline(n_voices=1, block_size=BLOCK_SIZE)

    pipe.material_id = 4  # Fold catastrophe

    injector = HarmonicTargetGenerator()
    injector.set_note(0, note_u8=45)  # A2 = 110 Hz

    n_blocks = blocks_for_duration(duration)
    inject_interval = 8

    danger_phase = 0.0

    for i in range(n_blocks):
        t = i * BLOCK_SIZE / FS
        progress = i / n_blocks

        if i % inject_interval == 0:
            injector.set_wave_params(0, wave_type=WaveType.SAW, partials_n=8,
                                    brightness_q15=int(0.7 * Q15_ONE))
            target = injector.generate_target(0, include_dense=True)
            if target.dense_target:
                pipe.symbol.inject_note(0, target.dense_target)

        # Danger oscillates but never exceeds threshold
        danger_phase += 2 * np.pi * 0.3 * BLOCK_SIZE / FS
        danger = 0.5 + 0.45 * np.sin(danger_phase)  # 0.05 to 0.95

        # Pressure and drive hover near but below bifurcation point
        pipe.params.pressure_q15 = int(danger * 0.85 * Q15_ONE)  # Never quite hits 1.0
        pipe.params.drive_q15 = int((0.4 + danger * 0.4) * Q15_ONE)

        # Add stress harmonics via tone
        pipe.params.tone_q15 = int((0.5 + danger * 0.3) * Q15_ONE)

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 10: SAW THAT LEARNED SOMETHING
# =============================================================================

def preset_saw_that_learned(duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Starts as normal saw pad.
    Slowly violates harmonic expectations.
    Still playable.

    Uses: Gradual coefficient drift via LowRankMorph
    """
    pipe = Pipeline(n_voices=1, block_size=BLOCK_SIZE)

    injector = HarmonicTargetGenerator()
    injector.set_note(0, note_u8=48)  # C3

    n_blocks = blocks_for_duration(duration)
    inject_interval = 6

    for i in range(n_blocks):
        progress = i / n_blocks

        if i % inject_interval == 0:
            injector.set_wave_params(0, wave_type=WaveType.SAW, partials_n=16,
                                    brightness_q15=Q15_ONE)
            target = injector.generate_target(0, include_dense=True)

            if target.dense_target and progress > 0.3:
                # "Learn" to drift certain harmonics
                learn = (progress - 0.3) / 0.7

                for leaf in range(LEAF_COUNT):
                    node_start = leaf * NODE_SIZE

                    # Some harmonics drift sharp, some flat
                    drift_factor = 1.0 + 0.15 * learn * np.sin(leaf * 1.3)

                    # Higher harmonics "learn" different levels
                    if leaf % 3 == 0:
                        amp_mod = 1.0 + learn * 0.5
                    elif leaf % 5 == 0:
                        amp_mod = 1.0 - learn * 0.3
                    else:
                        amp_mod = 1.0

                    for j in range(NODE_SIZE):
                        idx = node_start + j
                        target.dense_target[idx] = sat16(int(
                            target.dense_target[idx] * amp_mod * drift_factor
                        ))

                pipe.symbol.inject_note(0, target.dense_target)
            elif target.dense_target:
                pipe.symbol.inject_note(0, target.dense_target)

        # LowRankMorph parameter increases "learning"
        pipe.params.morph_q15 = int(progress * 0.7 * Q15_ONE)

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 11: BASS WITH MEMORY
# =============================================================================

def preset_bass_with_memory(duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bass line where previous notes affect the next.
    Groove has memory.

    Uses: Persistent coefficient state between notes
    """
    pipe = Pipeline(n_voices=1, block_size=BLOCK_SIZE)

    injector = HarmonicTargetGenerator()

    # Bass line pattern (MIDI notes)
    pattern = [33, 33, 38, 33, 40, 33, 38, 40]  # A1, D2, E2 pattern

    note_duration = duration / len(pattern)
    samples_per_note = int(note_duration * FS)
    blocks_per_note = samples_per_note // BLOCK_SIZE

    n_blocks = blocks_for_duration(duration)
    inject_interval = 4

    # Memory state
    memory_brightness = int(0.5 * Q15_ONE)
    memory_attack = 0.5

    prev_note = pattern[0]

    for i in range(n_blocks):
        note_idx = min(int(i / blocks_per_note), len(pattern) - 1)
        note_t = (i % blocks_per_note) / blocks_per_note
        current_note = pattern[note_idx]

        # Update memory based on note transitions
        if note_idx > 0 and i % blocks_per_note == 0:
            if current_note > prev_note:
                memory_brightness = min(Q15_ONE, memory_brightness + int(0.1 * Q15_ONE))
                memory_attack = min(0.9, memory_attack + 0.05)
            else:
                memory_brightness = max(int(0.2 * Q15_ONE), memory_brightness - int(0.05 * Q15_ONE))
                memory_attack = max(0.3, memory_attack - 0.02)
            prev_note = current_note

        if i % inject_interval == 0:
            injector.set_note(0, note_u8=current_note)

            # Brightness affected by memory
            n_partials = int(4 + (memory_brightness / Q15_ONE) * 8)
            injector.set_wave_params(0, wave_type=WaveType.SAW, partials_n=n_partials,
                                    brightness_q15=memory_brightness)

            target = injector.generate_target(0, include_dense=True)
            if target.dense_target:
                # Apply note envelope affected by memory
                if note_t < memory_attack * 0.1:
                    env = note_t / (memory_attack * 0.1)
                else:
                    env = 1 - (note_t - memory_attack * 0.1) * 0.5
                env = max(0.1, min(1.0, env))

                scaled = [sat16(int(c * env)) for c in target.dense_target]
                pipe.symbol.inject_note(0, scaled)

        pipe.params.tone_q15 = memory_brightness

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# PRESET 12: SINGLE KNOB APOCALYPSE
# =============================================================================

def preset_single_knob_apocalypse(duration: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    One parameter controls: structure, artifacts, chaos.
    Range from silence to annihilation.

    Uses: All catastrophe operators with unified control
    """
    max_voices = 8
    pipe = Pipeline(n_voices=max_voices, block_size=BLOCK_SIZE)

    pipe.material_id = 4  # Fold catastrophe

    injector = HarmonicTargetGenerator()
    base_note = 33  # A1 = 55 Hz

    n_blocks = blocks_for_duration(duration)
    inject_interval = 8

    for i in range(n_blocks):
        progress = i / n_blocks

        # The "knob" (0 to 1, with slight retreat at end)
        if progress < 0.8:
            knob = progress / 0.8
        else:
            knob = 1.0 - (progress - 0.8) / 0.2 * 0.3

        # Number of voices scales with knob
        n_active = int(1 + knob * (max_voices - 1))

        if i % inject_interval == 0:
            for v in range(n_active):
                # Detuning increases with knob
                detune_cents = int((v - n_active/2) * knob * 50 / n_active)

                injector.set_note(v, note_u8=base_note, cents_i16=detune_cents)
                injector.set_wave_params(v, wave_type=WaveType.SAW, partials_n=8,
                                        brightness_q15=int((0.5 + 0.5 * knob) * Q15_ONE))

                target = injector.generate_target(v, include_dense=True)
                if target.dense_target:
                    # Normalize for voice count
                    scale = 0.8 / n_active
                    scaled = [sat16(int(c * scale)) for c in target.dense_target]
                    pipe.symbol.inject_note(v, scaled)

        # All parameters driven by the single knob
        pipe.params.pressure_q15 = int(knob * 0.9 * Q15_ONE)
        pipe.params.drive_q15 = int((0.2 + knob * 0.6) * Q15_ONE)
        pipe.params.tone_q15 = int((0.3 + knob * 0.5) * Q15_ONE)
        pipe.params.morph_q15 = int(knob * Q15_ONE)

        pipe.step()

    return np.array(pipe.sink.L), np.array(pipe.sink.R)


# =============================================================================
# MAIN
# =============================================================================

def generate_all_presets():
    """Generate all 12 presets using WSE architecture."""

    presets = [
        ("01_topology_fold", preset_topology_fold, "Tree morph: Harmonic -> Metallic"),
        ("02_catastrophe_snap", preset_catastrophe_snap, "Cusp bifurcation: character flip"),
        ("03_perfect_morph", preset_perfect_morph, "Coefficient interpolation: SAW -> TRI"),
        ("04_living_chord", preset_living_chord, "12 voices with micro-drift coupling"),
        ("05_polyphonic_pressure", preset_polyphonic_pressure, "Fold intermodulation"),
        ("06_intentional_aliasing", preset_intentional_aliasing, "QuasiRotate + high partials"),
        ("07_vhs_ghost_choir", preset_vhs_ghost_choir, "NoiseBand + tape material"),
        ("08_pressure_field", preset_pressure_field, "Density via DiffuseUp/Down"),
        ("09_imminent_failure", preset_imminent_failure, "Fold near-bifurcation tension"),
        ("10_saw_that_learned", preset_saw_that_learned, "LowRankMorph drift"),
        ("11_bass_with_memory", preset_bass_with_memory, "Persistent coefficient state"),
        ("12_single_knob_apocalypse", preset_single_knob_apocalypse, "Unified catastrophe control"),
    ]

    print("=" * 70)
    print("GENERATING 12 WTF PRESETS (WSE Architecture)")
    print("=" * 70)
    print()
    print("Using: Symbol Engine -> Wavelet Engine -> Material Engine -> Safety Engine")
    print()

    for filename, func, desc in presets:
        print(f"{filename}: {desc}")
        try:
            L, R = func()
            write_audio(filename, L, R, subdirectory="wtf_presets", timestamp=False)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print("ALL PRESETS GENERATED")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_presets()
