"""
Exotic Wet Effects Demo - IDWPT Wavelet Coefficient Manipulation
=================================================================

Demonstrates advanced spectral effects using IDWPT wavelet synthesis
with coefficient-space manipulation techniques.

Effects:
1. Spectral Freeze - Hold a moment's spectrum frozen in time
2. Formant Filter - Vowel-like resonances (AH, EE, OH)
3. Spectral Blur/Smear - Dreamy, ethereal smearing
4. Band-Selective Distortion - Gritty mid-range character
5. Frequency Comb - Metallic, resonant comb filtering

Each effect demonstrates manipulation in coefficient space rather than
audio domain, showcasing the power of IDWPT synthesis.
"""

import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from symcrash_python_ap.carrier import CarrierOscillator
from symcrash_python_ap.anti_click import AntiClickOperator, AntiClickConfig
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.carrier_lift import Q15_ONE, Q15_HALF, Q15_SCALE, BLOCK_SIZE, SAMPLE_RATE
from symcrash_python_ap.harmonic_injector import (
    WaveType, LEAF_COUNT, NODE_SIZE, COEFF_LEN, NYQUIST,
    LEAF_CENTER_HZ, LEAF_EDGES_HZ, freq_to_leaf
)
from symcrash_python_ap.fixed import sat16, mul_q15_sat

# Constants
FS = SAMPLE_RATE
BLOCKS_PER_SEC = FS // BLOCK_SIZE

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def midi_to_freq(note: int) -> float:
    """Convert MIDI note to frequency."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def save_demo(name: str, samples: List[int]) -> str:
    """Save demo to WAV file."""
    metadata = audio_io.write_audio(name, samples, samples,
                                    subdirectory="exotic_fx", timestamp=False)
    return metadata.filepath


def generate_harmonic_coeffs(note: int, wave_type: WaveType = WaveType.SAW,
                             amplitude: int = Q15_HALF) -> np.ndarray:
    """
    Generate coefficient vector for a pitched note.
    Maps harmonics to wavelet leaves based on frequency.
    Also spreads energy to adjacent coefficients for richer sound.
    """
    freq = midi_to_freq(note)
    coeffs = np.zeros(COEFF_LEN, dtype=np.float32)

    # Generate harmonics
    for h in range(1, 33):  # Up to 32 harmonics
        h_freq = freq * h
        if h_freq >= NYQUIST:
            break

        # Wave-type specific amplitude
        if wave_type == WaveType.SAW:
            h_amp = amplitude / h
        elif wave_type == WaveType.TRIANGLE:
            h_amp = amplitude / (h * h) if h % 2 == 1 else 0
        elif wave_type == WaveType.PULSE:
            h_amp = amplitude / h if h % 2 == 1 else 0
        else:  # SINE
            h_amp = amplitude if h == 1 else 0

        if h_amp < 1:
            continue

        # Map to leaf
        leaf_idx = freq_to_leaf(h_freq)

        # Distribute across coefficients in leaf
        band_width = NYQUIST / LEAF_COUNT
        band_low = leaf_idx * band_width
        pos_in_band = (h_freq - band_low) / band_width
        coeff_idx = int(pos_in_band * NODE_SIZE)
        coeff_idx = min(coeff_idx, NODE_SIZE - 1)

        idx = leaf_idx * NODE_SIZE + coeff_idx
        if 0 <= idx < COEFF_LEN:
            coeffs[idx] += h_amp
            # Spread to adjacent coefficients for richer timbre
            if idx > 0:
                coeffs[idx - 1] += h_amp * 0.3
            if idx < COEFF_LEN - 1:
                coeffs[idx + 1] += h_amp * 0.3

    return coeffs


def synthesize_from_coeffs(coeffs: np.ndarray, phase_state: np.ndarray,
                           block_size: int = BLOCK_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize audio from coefficient vector using frequency-domain synthesis.
    Returns (audio_block, updated_phase_state).

    This uses the coefficient amplitude to modulate sinusoids at band-center frequencies,
    creating a distinctly different timbre from the carrier's additive synthesis.
    """
    audio = np.zeros(block_size, dtype=np.float32)

    band_width = NYQUIST / LEAF_COUNT

    for leaf_idx in range(LEAF_COUNT):
        # Use band center frequency for synthesis
        freq_center = LEAF_CENTER_HZ[leaf_idx]

        # Sum all coefficients in this leaf to get band energy
        leaf_start = leaf_idx * NODE_SIZE
        leaf_energy = 0.0
        for coeff_idx in range(NODE_SIZE):
            leaf_energy += abs(coeffs[leaf_start + coeff_idx])

        if leaf_energy < 1:
            continue

        # Phase increment for band center
        omega = 2.0 * math.pi * freq_center / FS

        # Generate with phase continuity
        phase = phase_state[leaf_idx]  # Use leaf index for phase state
        for i in range(block_size):
            audio[i] += leaf_energy * math.sin(phase + omega * i)

        # Update phase
        phase_state[leaf_idx] = (phase + omega * block_size) % (2.0 * math.pi)

    return audio, phase_state


# =============================================================================
# EFFECT 1: SPECTRAL FREEZE
# =============================================================================

def apply_spectral_freeze(coeffs: np.ndarray, frozen_coeffs: Optional[np.ndarray],
                          freeze_amount: float) -> np.ndarray:
    """
    Blend between live coefficients and frozen snapshot.
    freeze_amount: 0.0 = live, 1.0 = fully frozen
    """
    if frozen_coeffs is None:
        return coeffs

    return (1.0 - freeze_amount) * coeffs + freeze_amount * frozen_coeffs


def demo_1_spectral_freeze():
    """Spectral freeze - capture and hold a moment's spectrum."""
    print("\n" + "=" * 60)
    print("EFFECT 1: Spectral Freeze")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=48, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    anticlick = AntiClickOperator(AntiClickConfig(bloom_time_s=0.01))
    state = anticlick.init_state()
    anticlick.note_on(state)
    anticlick.set_wet_target(state, 0.8)

    phase_state = np.zeros(LEAF_COUNT, dtype=np.float32)

    samples = []
    duration_sec = 5.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    frozen_coeffs = None
    freeze_block = int(1.0 * BLOCKS_PER_SEC)  # Freeze at 1 second

    for block_idx in range(num_blocks):
        # Generate carrier
        carrier_block = np.array(carrier.render_block(0, BLOCK_SIZE), dtype=np.float32)

        # Change note at different times to show freeze effect
        if block_idx == int(0.5 * BLOCKS_PER_SEC):
            carrier.configure_voice(0, note=52, reset_phase=False)  # E3
        elif block_idx == int(2.5 * BLOCKS_PER_SEC):
            carrier.configure_voice(0, note=55, reset_phase=False)  # G3
        elif block_idx == int(3.5 * BLOCKS_PER_SEC):
            carrier.configure_voice(0, note=60, reset_phase=False)  # C4

        # Generate coefficients from current carrier state
        current_note = 48
        if block_idx >= int(0.5 * BLOCKS_PER_SEC):
            current_note = 52
        if block_idx >= int(2.5 * BLOCKS_PER_SEC):
            current_note = 55
        if block_idx >= int(3.5 * BLOCKS_PER_SEC):
            current_note = 60

        coeffs = generate_harmonic_coeffs(current_note, WaveType.SAW, Q15_HALF)

        # Capture freeze at specified block
        if block_idx == freeze_block:
            frozen_coeffs = coeffs.copy()
            print(f"  Frozen spectrum at {freeze_block / BLOCKS_PER_SEC:.2f}s")

        # Apply freeze effect (gradually increases)
        if block_idx > freeze_block:
            progress = (block_idx - freeze_block) / (num_blocks - freeze_block)
            freeze_amount = min(0.9, progress * 1.5)  # Ramp up freeze
            coeffs = apply_spectral_freeze(coeffs, frozen_coeffs, freeze_amount)

        # Synthesize wet path
        wet_block, phase_state = synthesize_from_coeffs(coeffs, phase_state)

        # 100% wet to hear the effect clearly
        anticlick.set_wet_target(state, 1.0)
        output = anticlick.step(state, carrier_block, wet_block.astype(np.float32))
        samples.extend(output.astype(int).tolist())

    filepath = save_demo("fx1_spectral_freeze", samples)
    print(f"  Note sequence: C3 -> E3 -> G3 -> C4")
    print(f"  Freeze captures E3 spectrum, holds it through changes")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


# =============================================================================
# EFFECT 2: FORMANT FILTER
# =============================================================================

# Formant frequencies for vowel sounds (F1, F2, F3)
FORMANTS = {
    'AH': [700, 1220, 2600],   # "ah" as in "father"
    'EE': [270, 2300, 3000],   # "ee" as in "beet"
    'OH': [400, 800, 2600],    # "oh" as in "boat"
    'OO': [300, 870, 2250],    # "oo" as in "boot"
}


def apply_formant_filter(coeffs: np.ndarray, formant_freqs: List[float],
                         resonance: float = 0.8) -> np.ndarray:
    """
    Apply formant resonances by boosting energy at formant frequencies
    and cutting frequencies between formants for stronger effect.
    """
    result = coeffs.copy()
    band_width = NYQUIST / LEAF_COUNT

    # First apply a gentle cut to all coefficients (will be boosted at formants)
    result *= 0.3

    for f_freq in formant_freqs:
        # Find the leaf containing this formant
        target_leaf = freq_to_leaf(f_freq)

        # Create resonant peak (Gaussian-ish)
        for leaf_offset in range(-6, 7):
            leaf_idx = target_leaf + leaf_offset
            if 0 <= leaf_idx < LEAF_COUNT:
                # Distance from peak
                leaf_center = LEAF_CENTER_HZ[leaf_idx]
                distance = abs(leaf_center - f_freq)

                # Gaussian weight (narrower = more resonant)
                q_width = 200 + 150 * (1.0 - resonance)  # Hz
                weight = math.exp(-(distance / q_width) ** 2)

                # Strong boost at formant frequencies
                boost = 1.0 + weight * resonance * 6.0

                for coeff_idx in range(NODE_SIZE):
                    idx = leaf_idx * NODE_SIZE + coeff_idx
                    result[idx] *= boost

    return result


def demo_2_formant_filter():
    """Formant filter - vowel-like resonances with melody."""
    print("\n" + "=" * 60)
    print("EFFECT 2: Formant Filter (Vowel Resonances)")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=48, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    anticlick = AntiClickOperator(AntiClickConfig(bloom_time_s=0.01))
    state = anticlick.init_state()
    anticlick.note_on(state)

    phase_state = np.zeros(LEAF_COUNT, dtype=np.float32)

    samples = []
    duration_sec = 8.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    # Vowel + note pattern for musical variation
    # Each segment: (vowel, note, duration_beats)
    pattern = [
        ('AH', 48, 2),  # C3
        ('AH', 55, 2),  # G3
        ('EE', 52, 2),  # E3
        ('EE', 60, 2),  # C4
        ('OH', 48, 2),  # C3
        ('OH', 53, 2),  # F3
        ('OO', 55, 2),  # G3
        ('OO', 48, 2),  # C3
    ]

    # Calculate blocks per beat (at 90 BPM)
    bpm = 90
    blocks_per_beat = int(BLOCKS_PER_SEC * 60 / bpm)

    segment_idx = 0
    blocks_in_segment = 0
    current_vowel = 'AH'
    current_note = 48

    for block_idx in range(num_blocks):
        # Check for segment change
        segment_duration_blocks = pattern[segment_idx][2] * blocks_per_beat
        if blocks_in_segment >= segment_duration_blocks:
            segment_idx = (segment_idx + 1) % len(pattern)
            blocks_in_segment = 0
            current_vowel, current_note, _ = pattern[segment_idx]
            carrier.configure_voice(0, note=current_note, reset_phase=False)
            # Reset phase state for new note to avoid clicks
            phase_state = np.zeros(LEAF_COUNT, dtype=np.float32)

        carrier_block = np.array(carrier.render_block(0, BLOCK_SIZE), dtype=np.float32)

        # Generate base coefficients for current note
        coeffs = generate_harmonic_coeffs(current_note, WaveType.SAW, Q15_HALF)

        # Apply formant filter
        coeffs = apply_formant_filter(coeffs, FORMANTS[current_vowel], resonance=0.85)

        # Normalize
        peak = np.max(np.abs(coeffs))
        if peak > Q15_HALF:
            coeffs = coeffs * (Q15_HALF / peak)

        # Synthesize
        wet_block, phase_state = synthesize_from_coeffs(coeffs, phase_state)

        # 100% wet to hear formants clearly
        anticlick.set_wet_target(state, 1.0)
        output = anticlick.step(state, carrier_block, wet_block.astype(np.float32))
        samples.extend(output.astype(int).tolist())

        blocks_in_segment += 1

    filepath = save_demo("fx2_formant_filter", samples)
    print(f"  Vowel/Note sequence with melody")
    print(f"  BPM: {bpm}")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


# =============================================================================
# EFFECT 3: SPECTRAL BLUR/SMEAR
# =============================================================================

def apply_spectral_blur(coeffs: np.ndarray, blur_strength: float) -> np.ndarray:
    """
    Blur coefficients across the entire spectrum, mixing energy between leaves.
    Creates dreamy, ethereal smearing effect.
    """
    result = coeffs.copy()

    # Cross-leaf blur - average with neighboring leaves
    temp = np.zeros_like(result)

    for idx in range(COEFF_LEN):
        neighbors = [result[idx]]
        # Add neighbors from adjacent positions
        for offset in [-1, 1, -NODE_SIZE, NODE_SIZE, -NODE_SIZE-1, -NODE_SIZE+1, NODE_SIZE-1, NODE_SIZE+1]:
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < COEFF_LEN:
                neighbors.append(result[neighbor_idx] * blur_strength * 0.4)

        # Keep center weight dominant but mix in neighbors
        center_weight = 1.0 - blur_strength * 0.7
        neighbor_sum = sum(neighbors[1:]) if len(neighbors) > 1 else 0
        temp[idx] = center_weight * result[idx] + neighbor_sum / max(1, len(neighbors) - 1)

    return temp


def apply_cross_band_diffusion(coeffs: np.ndarray, diffusion: float) -> np.ndarray:
    """
    Diffuse energy across adjacent bands for broader smearing.
    """
    result = coeffs.copy()

    for leaf_idx in range(1, LEAF_COUNT):
        for coeff_idx in range(NODE_SIZE):
            curr_idx = leaf_idx * NODE_SIZE + coeff_idx
            prev_idx = (leaf_idx - 1) * NODE_SIZE + coeff_idx

            # Exchange energy between bands
            transfer = result[curr_idx] * diffusion * 0.1
            result[curr_idx] -= transfer
            result[prev_idx] += transfer

    return result


def demo_3_spectral_blur():
    """Spectral blur - dreamy, ethereal smearing with arpeggio."""
    print("\n" + "=" * 60)
    print("EFFECT 3: Spectral Blur (Dreamy Smear)")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=55, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    anticlick = AntiClickOperator(AntiClickConfig(bloom_time_s=0.01))
    state = anticlick.init_state()
    anticlick.note_on(state)

    phase_state = np.zeros(LEAF_COUNT, dtype=np.float32)

    samples = []
    duration_sec = 8.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    # Slow arpeggio pattern for dreamy feel (60 BPM)
    bpm = 60
    blocks_per_beat = int(BLOCKS_PER_SEC * 60 / bpm)
    notes = [55, 60, 64, 67, 64, 60, 55, 52]  # G3 C4 E4 G4 E4 C4 G3 E3
    note_idx = 0
    blocks_in_note = 0
    current_note = notes[0]

    for block_idx in range(num_blocks):
        # Change note every beat
        if blocks_in_note >= blocks_per_beat:
            note_idx = (note_idx + 1) % len(notes)
            current_note = notes[note_idx]
            carrier.configure_voice(0, note=current_note, reset_phase=False)
            blocks_in_note = 0

        carrier_block = np.array(carrier.render_block(0, BLOCK_SIZE), dtype=np.float32)

        # Generate coefficients for current note
        coeffs = generate_harmonic_coeffs(current_note, WaveType.SAW, Q15_HALF)

        # Progressive blur that increases over time
        progress = block_idx / num_blocks
        blur_amount = progress * 0.85  # 0 to 0.85

        # Apply multiple blur passes for dreamy effect
        for _ in range(3):
            coeffs = apply_spectral_blur(coeffs, blur_amount)

        # Cross-band diffusion
        coeffs = apply_cross_band_diffusion(coeffs, blur_amount * 0.5)

        # Normalize
        peak = np.max(np.abs(coeffs))
        if peak > Q15_HALF:
            coeffs = coeffs * (Q15_HALF / peak)

        # Synthesize
        wet_block, phase_state = synthesize_from_coeffs(coeffs, phase_state)

        # 100% wet to hear blur clearly
        anticlick.set_wet_target(state, 1.0)
        output = anticlick.step(state, carrier_block, wet_block.astype(np.float32))
        samples.extend(output.astype(int).tolist())

        blocks_in_note += 1

    filepath = save_demo("fx3_spectral_blur", samples)
    print(f"  Arpeggio: G3 C4 E4 G4 E4 C4 G3 E3")
    print(f"  BPM: {bpm}")
    print(f"  Blur: 0% -> 85% progressive")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


# =============================================================================
# EFFECT 4: BAND-SELECTIVE DISTORTION
# =============================================================================

def apply_band_distortion(coeffs: np.ndarray, band_start: int, band_end: int,
                          drive: float) -> np.ndarray:
    """
    Apply soft-clipping distortion to a specific band range.
    Also adds harmonic content by folding energy.
    """
    result = coeffs.copy()

    for idx in range(band_start, min(band_end, COEFF_LEN)):
        # Scale by drive
        driven = result[idx] * drive

        # Soft clipper (tanh-like polynomial)
        if abs(driven) < Q15_ONE:
            x_norm = driven / Q15_ONE
            # y = x - x^3/3 approximates tanh, add odd harmonics
            clipped = x_norm - (x_norm ** 3) / 3.0 + (x_norm ** 5) / 5.0
            result[idx] = clipped * Q15_ONE
        else:
            result[idx] = Q15_ONE * 0.95 if driven > 0 else -Q15_ONE * 0.95

        # Add harmonic folding - creates new harmonics above
        if idx + NODE_SIZE < COEFF_LEN and abs(result[idx]) > Q15_HALF * 0.5:
            result[idx + NODE_SIZE] += result[idx] * 0.3 * (drive - 1.0) / 4.0

    return result


def demo_4_band_distortion():
    """Band-selective distortion - gritty mid-range character with riff."""
    print("\n" + "=" * 60)
    print("EFFECT 4: Band-Selective Distortion")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=36, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    anticlick = AntiClickOperator(AntiClickConfig(bloom_time_s=0.01))
    state = anticlick.init_state()
    anticlick.note_on(state)

    phase_state = np.zeros(LEAF_COUNT, dtype=np.float32)

    samples = []
    duration_sec = 8.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    # Mid-range bands (leaves 8-40 for bass notes)
    mid_start = 8 * NODE_SIZE
    mid_end = 40 * NODE_SIZE

    # Rock/metal style riff at 120 BPM (eighth notes)
    bpm = 120
    blocks_per_eighth = int(BLOCKS_PER_SEC * 60 / bpm / 2)
    # Power chord roots: E2, E2, G2, A2, E2, E2, D2, E2 (repeat)
    riff_notes = [40, 40, 43, 45, 40, 40, 38, 40]
    note_idx = 0
    blocks_in_note = 0
    current_note = riff_notes[0]

    for block_idx in range(num_blocks):
        # Change note on eighth notes
        if blocks_in_note >= blocks_per_eighth:
            note_idx = (note_idx + 1) % len(riff_notes)
            current_note = riff_notes[note_idx]
            carrier.configure_voice(0, note=current_note, reset_phase=False)
            blocks_in_note = 0

        carrier_block = np.array(carrier.render_block(0, BLOCK_SIZE), dtype=np.float32)

        # Generate coefficients for current note
        coeffs = generate_harmonic_coeffs(current_note, WaveType.SAW, Q15_HALF)

        # Progressive distortion building up
        progress = block_idx / num_blocks
        drive = 1.5 + progress * 3.5  # 1.5x to 5x drive

        # Apply distortion to mid-range only
        coeffs = apply_band_distortion(coeffs, mid_start, mid_end, drive)

        # Normalize
        peak = np.max(np.abs(coeffs))
        if peak > Q15_HALF:
            coeffs = coeffs * (Q15_HALF / peak)

        # Synthesize
        wet_block, phase_state = synthesize_from_coeffs(coeffs, phase_state)

        # 100% wet to hear distortion clearly
        anticlick.set_wet_target(state, 1.0)
        output = anticlick.step(state, carrier_block, wet_block.astype(np.float32))
        samples.extend(output.astype(int).tolist())

        blocks_in_note += 1

    filepath = save_demo("fx4_band_distortion", samples)
    print(f"  Riff: E2 E2 G2 A2 E2 E2 D2 E2")
    print(f"  BPM: {bpm}")
    print(f"  Drive: 1.5x -> 5x progressive")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


# =============================================================================
# EFFECT 5: FREQUENCY COMB
# =============================================================================

def apply_spectral_comb(coeffs: np.ndarray, comb_freq: float,
                        depth: float = 0.5) -> np.ndarray:
    """
    Create comb filter by adding delayed copies at harmonic intervals.
    """
    result = coeffs.copy()
    band_width = NYQUIST / LEAF_COUNT

    for leaf_idx in range(LEAF_COUNT):
        freq_center = LEAF_CENTER_HZ[leaf_idx]

        # Calculate comb peaks and nulls
        # Comb creates peaks at multiples of comb_freq
        harmonic_num = round(freq_center / comb_freq)
        if harmonic_num > 0:
            nearest_peak = harmonic_num * comb_freq
            distance = abs(freq_center - nearest_peak)

            # Create alternating peaks and nulls
            # Peak at multiples, null halfway between
            half_spacing = comb_freq / 2
            if distance < half_spacing * 0.5:
                # Near a peak - boost
                boost = 1.0 + depth * (1.0 - distance / half_spacing)
            else:
                # Near a null - cut
                boost = 1.0 - depth * 0.5

            for coeff_idx in range(NODE_SIZE):
                idx = leaf_idx * NODE_SIZE + coeff_idx
                result[idx] *= boost

    return result


def demo_5_frequency_comb():
    """Frequency comb - metallic, resonant filtering with sequence."""
    print("\n" + "=" * 60)
    print("EFFECT 5: Frequency Comb (Metallic Resonance)")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=48, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    anticlick = AntiClickOperator(AntiClickConfig(bloom_time_s=0.01))
    state = anticlick.init_state()
    anticlick.note_on(state)

    phase_state = np.zeros(LEAF_COUNT, dtype=np.float32)

    samples = []
    duration_sec = 8.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    # Synth sequence at 130 BPM (16th notes)
    bpm = 130
    blocks_per_16th = int(BLOCKS_PER_SEC * 60 / bpm / 4)
    # Minor scale sequence for metallic feel
    sequence = [48, 48, 51, 48, 53, 51, 48, 46,  # C3 C3 Eb3 C3 F3 Eb3 C3 Bb2
                48, 48, 51, 48, 55, 53, 51, 48]  # C3 C3 Eb3 C3 G3 F3 Eb3 C3
    note_idx = 0
    blocks_in_note = 0
    current_note = sequence[0]

    for block_idx in range(num_blocks):
        # Change note on 16th notes
        if blocks_in_note >= blocks_per_16th:
            note_idx = (note_idx + 1) % len(sequence)
            current_note = sequence[note_idx]
            carrier.configure_voice(0, note=current_note, reset_phase=False)
            blocks_in_note = 0

        carrier_block = np.array(carrier.render_block(0, BLOCK_SIZE), dtype=np.float32)

        # Generate coefficients for current note
        coeffs = generate_harmonic_coeffs(current_note, WaveType.SAW, Q15_HALF)

        # Sweep comb frequency for phaser-like effect (slower sweep)
        progress = block_idx / num_blocks
        # Sweep from 200 Hz to 800 Hz and back, twice
        comb_freq = 200 + 600 * abs(math.sin(progress * math.pi * 4))

        # Apply comb filter
        coeffs = apply_spectral_comb(coeffs, comb_freq, depth=0.75)

        # Normalize
        peak = np.max(np.abs(coeffs))
        if peak > Q15_HALF:
            coeffs = coeffs * (Q15_HALF / peak)

        # Synthesize
        wet_block, phase_state = synthesize_from_coeffs(coeffs, phase_state)

        # 100% wet to hear comb filter clearly
        anticlick.set_wet_target(state, 1.0)
        output = anticlick.step(state, carrier_block, wet_block.astype(np.float32))
        samples.extend(output.astype(int).tolist())

        blocks_in_note += 1

    filepath = save_demo("fx5_frequency_comb", samples)
    print(f"  Sequence: C minor pattern")
    print(f"  BPM: {bpm}")
    print(f"  Comb sweep: 200Hz -> 800Hz (x2)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


# =============================================================================
# DRY VERSIONS (for contrast)
# =============================================================================

def demo_1_dry():
    """Dry version of spectral freeze demo - pure carrier."""
    print("\n" + "=" * 60)
    print("DRY 1: Spectral Freeze (Carrier Only)")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=48, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    samples = []
    duration_sec = 5.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    for block_idx in range(num_blocks):
        # Change note at same times as wet version
        if block_idx == int(0.5 * BLOCKS_PER_SEC):
            carrier.configure_voice(0, note=52, reset_phase=False)
        elif block_idx == int(2.5 * BLOCKS_PER_SEC):
            carrier.configure_voice(0, note=55, reset_phase=False)
        elif block_idx == int(3.5 * BLOCKS_PER_SEC):
            carrier.configure_voice(0, note=60, reset_phase=False)

        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)

    filepath = save_demo("fx1_dry_carrier", samples)
    print(f"  Note sequence: C3 -> E3 -> G3 -> C4")
    print(f"  Pure carrier (no freeze effect)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


def demo_2_dry():
    """Dry version of formant filter demo - pure carrier."""
    print("\n" + "=" * 60)
    print("DRY 2: Formant Filter (Carrier Only)")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=48, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    samples = []
    duration_sec = 8.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    pattern = [
        ('AH', 48, 2), ('AH', 55, 2), ('EE', 52, 2), ('EE', 60, 2),
        ('OH', 48, 2), ('OH', 53, 2), ('OO', 55, 2), ('OO', 48, 2),
    ]

    bpm = 90
    blocks_per_beat = int(BLOCKS_PER_SEC * 60 / bpm)

    segment_idx = 0
    blocks_in_segment = 0
    current_note = 48

    for block_idx in range(num_blocks):
        segment_duration_blocks = pattern[segment_idx][2] * blocks_per_beat
        if blocks_in_segment >= segment_duration_blocks:
            segment_idx = (segment_idx + 1) % len(pattern)
            blocks_in_segment = 0
            _, current_note, _ = pattern[segment_idx]
            carrier.configure_voice(0, note=current_note, reset_phase=False)

        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)
        blocks_in_segment += 1

    filepath = save_demo("fx2_dry_carrier", samples)
    print(f"  Same melody, no formant filtering")
    print(f"  BPM: {bpm}")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


def demo_3_dry():
    """Dry version of spectral blur demo - pure carrier."""
    print("\n" + "=" * 60)
    print("DRY 3: Spectral Blur (Carrier Only)")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=55, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    samples = []
    duration_sec = 8.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    bpm = 60
    blocks_per_beat = int(BLOCKS_PER_SEC * 60 / bpm)
    notes = [55, 60, 64, 67, 64, 60, 55, 52]
    note_idx = 0
    blocks_in_note = 0

    for block_idx in range(num_blocks):
        if blocks_in_note >= blocks_per_beat:
            note_idx = (note_idx + 1) % len(notes)
            carrier.configure_voice(0, note=notes[note_idx], reset_phase=False)
            blocks_in_note = 0

        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)
        blocks_in_note += 1

    filepath = save_demo("fx3_dry_carrier", samples)
    print(f"  Arpeggio: G3 C4 E4 G4 E4 C4 G3 E3")
    print(f"  BPM: {bpm}")
    print(f"  Pure carrier (no blur)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


def demo_4_dry():
    """Dry version of band distortion demo - pure carrier."""
    print("\n" + "=" * 60)
    print("DRY 4: Band Distortion (Carrier Only)")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=40, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    samples = []
    duration_sec = 8.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    bpm = 120
    blocks_per_eighth = int(BLOCKS_PER_SEC * 60 / bpm / 2)
    riff_notes = [40, 40, 43, 45, 40, 40, 38, 40]
    note_idx = 0
    blocks_in_note = 0

    for block_idx in range(num_blocks):
        if blocks_in_note >= blocks_per_eighth:
            note_idx = (note_idx + 1) % len(riff_notes)
            carrier.configure_voice(0, note=riff_notes[note_idx], reset_phase=False)
            blocks_in_note = 0

        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)
        blocks_in_note += 1

    filepath = save_demo("fx4_dry_carrier", samples)
    print(f"  Riff: E2 E2 G2 A2 E2 E2 D2 E2")
    print(f"  BPM: {bpm}")
    print(f"  Pure carrier (no distortion)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


def demo_5_dry():
    """Dry version of frequency comb demo - pure carrier."""
    print("\n" + "=" * 60)
    print("DRY 5: Frequency Comb (Carrier Only)")
    print("=" * 60)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note=48, harmonic_amps=[Q15_ONE // n for n in range(1, 9)])
    carrier.gate_on(0)

    samples = []
    duration_sec = 8.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    bpm = 130
    blocks_per_16th = int(BLOCKS_PER_SEC * 60 / bpm / 4)
    sequence = [48, 48, 51, 48, 53, 51, 48, 46,
                48, 48, 51, 48, 55, 53, 51, 48]
    note_idx = 0
    blocks_in_note = 0

    for block_idx in range(num_blocks):
        if blocks_in_note >= blocks_per_16th:
            note_idx = (note_idx + 1) % len(sequence)
            carrier.configure_voice(0, note=sequence[note_idx], reset_phase=False)
            blocks_in_note = 0

        block = carrier.render_block(0, BLOCK_SIZE)
        samples.extend(block)
        blocks_in_note += 1

    filepath = save_demo("fx5_dry_carrier", samples)
    print(f"  Sequence: C minor pattern")
    print(f"  BPM: {bpm}")
    print(f"  Pure carrier (no comb filter)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("EXOTIC WET EFFECTS DEMO - IDWPT COEFFICIENT MANIPULATION")
    print("=" * 60)
    print(f"\nSample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Coefficients: {COEFF_LEN} ({LEAF_COUNT} leaves x {NODE_SIZE} per leaf)")
    print(f"Output: audio_out/exotic_fx/")

    # Run wet demos
    print("\n" + "=" * 60)
    print("WET EFFECTS (with IDWPT processing)")
    print("=" * 60)
    demo_1_spectral_freeze()
    demo_2_formant_filter()
    demo_3_spectral_blur()
    demo_4_band_distortion()
    demo_5_frequency_comb()

    # Run dry demos for contrast
    print("\n" + "=" * 60)
    print("DRY VERSIONS (carrier only, for contrast)")
    print("=" * 60)
    demo_1_dry()
    demo_2_dry()
    demo_3_dry()
    demo_4_dry()
    demo_5_dry()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
    print("\nFiles saved to: audio_out/exotic_fx/")
    print("\nWET (with effects):")
    print("  - fx1_spectral_freeze.wav")
    print("  - fx2_formant_filter.wav")
    print("  - fx3_spectral_blur.wav")
    print("  - fx4_band_distortion.wav")
    print("  - fx5_frequency_comb.wav")
    print("\nDRY (carrier only, for contrast):")
    print("  - fx1_dry_carrier.wav")
    print("  - fx2_dry_carrier.wav")
    print("  - fx3_dry_carrier.wav")
    print("  - fx4_dry_carrier.wav")
    print("  - fx5_dry_carrier.wav")


if __name__ == "__main__":
    main()
