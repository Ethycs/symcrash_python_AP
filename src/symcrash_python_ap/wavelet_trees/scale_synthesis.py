"""
Scale-Based Wavelet Synthesis
=============================

Maps wavelet packet leaves to musical scale degrees instead of
linear frequency bands. This makes coefficient algebra musically
meaningful:

- Permutation operators -> transposition, inversion
- Diagonal operators -> per-note dynamics
- Banded operators -> harmonic coupling
- Low-rank operators -> chord voicing

Supports multiple scale types and tuning systems.
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass

from ..fixed import sat16, mul_q15_sat, Q15_ONE
from . import LEAF_COUNT, NODE_SIZE, BLOCK_SIZE


# =============================================================================
# MUSICAL SCALES AND TUNINGS
# =============================================================================

@dataclass
class Scale:
    """Definition of a musical scale."""
    name: str
    degrees: List[float]  # Semitone offsets from root (0 = root)
    octaves: float        # How many octaves to span with 32 leaves
    root_hz: float        # Root frequency


# Common scales (semitone offsets)
SCALES = {
    # 12-tone chromatic: 32 leaves = 2.67 octaves
    "chromatic": Scale(
        name="chromatic",
        degrees=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        octaves=2.67,
        root_hz=65.41,  # C2
    ),

    # Major scale: 32 leaves = 4.57 octaves
    "major": Scale(
        name="major",
        degrees=[0, 2, 4, 5, 7, 9, 11],  # W W H W W W H
        octaves=4.57,
        root_hz=65.41,  # C2
    ),

    # Minor scale: 32 leaves = 4.57 octaves
    "minor": Scale(
        name="minor",
        degrees=[0, 2, 3, 5, 7, 8, 10],  # W H W W H W W
        octaves=4.57,
        root_hz=65.41,
    ),

    # Pentatonic: 32 leaves = 6.4 octaves (huge range!)
    "pentatonic": Scale(
        name="pentatonic",
        degrees=[0, 2, 4, 7, 9],  # Major pentatonic
        octaves=6.4,
        root_hz=32.70,  # C1
    ),

    # Harmonic series: 32 leaves = partials 1-32
    "harmonic": Scale(
        name="harmonic",
        degrees=None,  # Special case - computed differently
        octaves=5.0,   # log2(32) octaves
        root_hz=55.0,  # A1
    ),

    # Whole tone: 32 leaves = 5.33 octaves
    "whole_tone": Scale(
        name="whole_tone",
        degrees=[0, 2, 4, 6, 8, 10],
        octaves=5.33,
        root_hz=65.41,
    ),

    # Blues: 32 leaves = 5.33 octaves
    "blues": Scale(
        name="blues",
        degrees=[0, 3, 5, 6, 7, 10],
        octaves=5.33,
        root_hz=65.41,
    ),

    # Gamelan pelog (approximate): 32 leaves = 6.4 octaves
    "pelog": Scale(
        name="pelog",
        degrees=[0, 1, 3, 7, 8],  # Approximate pelog
        octaves=6.4,
        root_hz=65.41,
    ),

    # Quarter-tone (24-TET): 32 leaves = 1.33 octaves (microtonal)
    "quarter_tone": Scale(
        name="quarter_tone",
        degrees=[i * 0.5 for i in range(24)],
        octaves=1.33,
        root_hz=130.81,  # C3 (narrower range)
    ),

    # Just intonation major
    "just_major": Scale(
        name="just_major",
        degrees=None,  # Computed from ratios
        octaves=4.57,
        root_hz=65.41,
    ),
}

# Just intonation ratios (for just_major)
JUST_RATIOS = [1/1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8]  # Major scale


def compute_scale_frequencies(scale: Scale, n_leaves: int = 32) -> List[float]:
    """Compute frequency for each leaf based on scale."""

    if scale.name == "harmonic":
        # Harmonic series: leaf n = root * (n + 1)
        return [scale.root_hz * (i + 1) for i in range(n_leaves)]

    elif scale.name == "just_major":
        # Just intonation
        freqs = []
        for i in range(n_leaves):
            octave = i // len(JUST_RATIOS)
            degree = i % len(JUST_RATIOS)
            freq = scale.root_hz * JUST_RATIOS[degree] * (2 ** octave)
            freqs.append(freq)
        return freqs

    else:
        # Equal temperament scales
        freqs = []
        n_degrees = len(scale.degrees)

        for i in range(n_leaves):
            # Which octave and scale degree?
            scale_position = i % n_degrees
            octave = i // n_degrees

            # Semitones from root
            semitones = scale.degrees[scale_position] + (octave * 12)

            # Frequency (12-TET)
            freq = scale.root_hz * (2 ** (semitones / 12))
            freqs.append(freq)

        return freqs


# =============================================================================
# SCALE-AWARE SYNTHESIZER
# =============================================================================

class ScaleSynthesizer:
    """
    Wavelet packet synthesis with musical scale mapping.

    Each leaf synthesizes at a specific scale degree frequency,
    not at the wavelet band center frequency.
    """

    def __init__(self, scale_name: str = "major", root_hz: float = None):
        """
        Initialize with a scale.

        scale_name: One of the predefined scales or custom
        root_hz: Override the default root frequency
        """
        if scale_name not in SCALES:
            raise ValueError(f"Unknown scale: {scale_name}")

        self.scale = SCALES[scale_name]
        if root_hz is not None:
            # Create modified scale with new root
            self.scale = Scale(
                name=self.scale.name,
                degrees=self.scale.degrees,
                octaves=self.scale.octaves,
                root_hz=root_hz,
            )

        # Compute frequencies for all 32 leaves
        self.leaf_frequencies = compute_scale_frequencies(self.scale, LEAF_COUNT)

        # Precompute angular frequencies (radians per sample at 48kHz)
        self.sample_rate = 48000.0
        self.leaf_omegas = [
            2 * np.pi * f / self.sample_rate
            for f in self.leaf_frequencies
        ]

        # Phase accumulators for continuous synthesis
        self.phases = [0.0] * LEAF_COUNT

    def get_scale_info(self) -> Dict:
        """Return scale information."""
        return {
            "name": self.scale.name,
            "root_hz": self.scale.root_hz,
            "frequencies": self.leaf_frequencies,
            "range_hz": (min(self.leaf_frequencies), max(self.leaf_frequencies)),
            "range_midi": (
                self._hz_to_midi(min(self.leaf_frequencies)),
                self._hz_to_midi(max(self.leaf_frequencies)),
            ),
        }

    def _hz_to_midi(self, hz: float) -> float:
        """Convert Hz to MIDI note number."""
        return 69 + 12 * np.log2(hz / 440.0)

    def synthesize(self, coeff_nodes: List[List[int]],
                   continuous_phase: bool = True) -> List[int]:
        """
        Synthesize audio from coefficient nodes using scale frequencies.

        coeff_nodes: 32 nodes of 8 coefficients each
        continuous_phase: Maintain phase continuity across blocks
        """
        output = [0] * BLOCK_SIZE

        for leaf_idx in range(LEAF_COUNT):
            node = coeff_nodes[leaf_idx]

            # Skip silent nodes
            node_energy = sum(abs(c) for c in node)
            if node_energy < 10:
                continue

            # Get angular frequency for this scale degree
            omega = self.leaf_omegas[leaf_idx]

            # Compute amplitude from node (RMS)
            rms = int(np.sqrt(sum(c*c for c in node) / NODE_SIZE))

            # Starting phase (continuous or fresh)
            phase = self.phases[leaf_idx] if continuous_phase else 0.0

            # Generate samples at exact scale frequency
            for i in range(BLOCK_SIZE):
                carrier = np.sin(phase)
                sample = int(rms * carrier)
                output[i] = sat16(output[i] + sample)

                # Advance phase
                phase += omega

            # Store phase for continuity (wrap to avoid float overflow)
            self.phases[leaf_idx] = phase % (2 * np.pi)

        return output

    def reset_phases(self):
        """Reset all phase accumulators."""
        self.phases = [0.0] * LEAF_COUNT


# =============================================================================
# SCALE-AWARE NOTE INJECTION
# =============================================================================

def inject_scale_note(
    coeffs: List[int],
    degree: int,           # Scale degree (0 = root, 1 = 2nd, etc.)
    amplitude: int,        # Q15 amplitude
    n_harmonics: int = 8,  # How many harmonics to include
    scale_synth: ScaleSynthesizer = None,
) -> List[int]:
    """
    Inject a note at a specific scale degree.

    Unlike frequency-based injection, this places energy directly
    in the leaf corresponding to the scale degree.
    """
    if scale_synth is None:
        scale_synth = ScaleSynthesizer("major")

    # Primary note goes in the corresponding leaf
    leaf_idx = degree % LEAF_COUNT
    node_start = leaf_idx * NODE_SIZE

    # Fill node with amplitude envelope
    for i in range(NODE_SIZE):
        mod = 1.0 + 0.05 * np.sin(2 * np.pi * i / NODE_SIZE)
        coeffs[node_start + i] = sat16(int(amplitude * mod))

    # Add harmonics (if they fall within scale)
    fund_freq = scale_synth.leaf_frequencies[leaf_idx]

    for h in range(2, n_harmonics + 1):
        harm_freq = fund_freq * h
        harm_amp = amplitude // h

        # Find closest scale degree to this harmonic
        closest_leaf = None
        closest_dist = float('inf')

        for idx, freq in enumerate(scale_synth.leaf_frequencies):
            dist = abs(freq - harm_freq) / harm_freq  # Relative distance
            if dist < closest_dist and dist < 0.1:  # Within 10%
                closest_dist = dist
                closest_leaf = idx

        if closest_leaf is not None:
            node_start = closest_leaf * NODE_SIZE
            for i in range(NODE_SIZE):
                mod = 1.0 + 0.05 * np.sin(2 * np.pi * i / NODE_SIZE + h)
                coeffs[node_start + i] = sat16(
                    coeffs[node_start + i] + int(harm_amp * mod)
                )

    return coeffs


def inject_chord(
    coeffs: List[int],
    degrees: List[int],    # List of scale degrees
    amplitudes: List[int], # Amplitude per note (or single value)
    scale_synth: ScaleSynthesizer = None,
) -> List[int]:
    """Inject a chord (multiple scale degrees simultaneously)."""
    if isinstance(amplitudes, int):
        amplitudes = [amplitudes] * len(degrees)

    for deg, amp in zip(degrees, amplitudes):
        coeffs = inject_scale_note(coeffs, deg, amp, n_harmonics=4,
                                   scale_synth=scale_synth)

    return coeffs


# =============================================================================
# SCALE-AWARE OPERATORS
# =============================================================================

def make_transpose_permutation(semitones: int, scale: Scale) -> List[int]:
    """
    Create permutation that transposes by N semitones within scale.

    Returns: permutation[old_leaf] = new_leaf
    """
    n_degrees = len(scale.degrees) if scale.degrees else 12

    # How many scale degrees is this transposition?
    # Find closest scale degree to target interval
    scale_steps = round(semitones * n_degrees / 12)

    perm = []
    for i in range(LEAF_COUNT):
        new_idx = (i + scale_steps) % LEAF_COUNT
        perm.append(new_idx)

    return perm


def make_inversion_permutation(axis_degree: int = 0) -> List[int]:
    """
    Create permutation that inverts melody around an axis.

    axis_degree: Scale degree to invert around
    """
    perm = []
    for i in range(LEAF_COUNT):
        # Mirror around axis
        new_idx = (2 * axis_degree - i) % LEAF_COUNT
        perm.append(new_idx)

    return perm


def make_octave_fold_permutation() -> List[int]:
    """
    Create permutation that folds higher octaves onto lower.

    Useful for creating denser voicings.
    """
    # Assuming 7-note scale, fold octaves 2+ onto octave 1
    perm = []
    for i in range(LEAF_COUNT):
        folded = i % 7  # Fold to first octave (7 notes)
        perm.append(folded)

    return perm


# =============================================================================
# DEMO AND TEST
# =============================================================================

def demo_scales():
    """Demonstrate scale-based synthesis."""
    print("=" * 60)
    print("SCALE-BASED SYNTHESIS")
    print("=" * 60)

    for scale_name in ["chromatic", "major", "pentatonic", "harmonic", "just_major"]:
        scale = SCALES[scale_name]
        freqs = compute_scale_frequencies(scale)

        print(f"\n{scale_name.upper()} (root={scale.root_hz:.1f} Hz):")
        print(f"  Range: {freqs[0]:.1f} Hz - {freqs[-1]:.1f} Hz")
        print(f"  First 8 leaves: {[f'{f:.1f}' for f in freqs[:8]]}")

        if scale_name == "major":
            # Show note names
            notes = ["C", "D", "E", "F", "G", "A", "B"]
            print(f"  Notes: ", end="")
            for i, f in enumerate(freqs[:14]):
                note = notes[i % 7]
                octave = 2 + i // 7
                print(f"{note}{octave}({f:.0f}) ", end="")
            print("...")


def test_scale_synthesis():
    """Test scale-aware synthesis."""
    print("\n" + "=" * 60)
    print("SYNTHESIS TEST")
    print("=" * 60)

    synth = ScaleSynthesizer("major", root_hz=130.81)  # C3
    info = synth.get_scale_info()

    print(f"\nScale: {info['name']}")
    print(f"Range: {info['range_hz'][0]:.1f} - {info['range_hz'][1]:.1f} Hz")
    print(f"MIDI range: {info['range_midi'][0]:.1f} - {info['range_midi'][1]:.1f}")

    # Create test coefficients (C major chord: degrees 0, 2, 4)
    coeffs = [0] * (LEAF_COUNT * NODE_SIZE)

    # Root (C)
    for i in range(8):
        coeffs[0*8 + i] = 8000

    # Third (E)
    for i in range(8):
        coeffs[2*8 + i] = 6000

    # Fifth (G)
    for i in range(8):
        coeffs[4*8 + i] = 6000

    nodes = [coeffs[i*8:(i+1)*8] for i in range(LEAF_COUNT)]
    audio = synth.synthesize(nodes)

    print(f"\nC major chord synthesis:")
    print(f"  Peak: {max(abs(x) for x in audio)}")
    print(f"  Expected frequencies: {info['frequencies'][0]:.1f}, {info['frequencies'][2]:.1f}, {info['frequencies'][4]:.1f} Hz")

    # Check zero crossings
    zc = sum(1 for i in range(1, len(audio)) if (audio[i] > 0) != (audio[i-1] > 0))
    print(f"  Zero crossings: {zc}")
