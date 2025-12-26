"""
Safety Engine Test - Chromatic Scale and Musical Patterns
==========================================================

LEGACY TEST - Requires StandardWaveletPipeline which has been replaced by CarrierLiftPipeline.
This test needs to be rewritten to use the current architecture.

Comprehensive test exercising the safety engine with Baby Fourier (WaveletEngine)
using standard anti-click features across various musical patterns:

1. Chromatic scale C3-C5 (stepwise pitch motion)
2. Arpeggio patterns (chord tones, wider intervals)
3. Rapid transitions x100 (stress test)
4. Classic synth arpeggio (fast, rhythmic patterns)

Uses standard pipeline anti-click features:
- WaveletEngine RegularityOperator (coefficient smoothing, micro-bridge, slope projection)
- SafetyEngine declick (voice on/off microfades)
- SafetyEngine step_block (limiting)

This test focuses on evaluating the Safety Engine's effectiveness.
"""

import pytest
pytest.skip("Legacy test - needs rewrite for CarrierLiftPipeline", allow_module_level=True)

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC
from audio_io import write_audio

from symcrash_python_ap.wavelet.wavelet_engine import WaveletEngine
from symcrash_python_ap.safety.safety_engine import SafetyEngine
from symcrash_python_ap.frames import VoiceFrame, ParamFrame
from symcrash_python_ap.wavelet.wavelet_packet_trees import COEFF_LEN
from symcrash_python_ap.fixed import sat16
from symcrash_python_ap.safety.unified_renderer import StandardWaveletPipeline

# =============================================================================
# CONSTANTS
# =============================================================================

OUTPUT_SUBDIR = "safety_tests"
Q15_ONE = 32767
Q15_HALF = 16384
NODE_SIZE = 8
N_LEAVES = 32


def midi_to_freq(note: int) -> float:
    """MIDI note to frequency."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def generate_coeffs(freq: float, amplitude: int = 8000) -> List[int]:
    """
    Generate wavelet coefficients for a frequency using Baby Fourier mapping.

    Maps harmonics to leaf nodes using logarithmic frequency distribution.
    """
    coeffs = [0] * COEFF_LEN
    nyquist = FS / 2

    def freq_to_leaf(f: float) -> int:
        if f < 20:
            return 0
        if f > 20000:
            return N_LEAVES - 1
        log_pos = np.log2(f / 20) / np.log2(20000 / 20)
        return int(log_pos * (N_LEAVES - 1))

    # Generate harmonics
    for h in range(1, 16):
        h_freq = freq * h
        if h_freq >= nyquist * 0.95:
            break

        leaf_idx = freq_to_leaf(h_freq)
        leaf_idx = max(0, min(N_LEAVES - 1, leaf_idx))

        # Harmonic amplitude with natural rolloff
        h_amp = amplitude // (h ** 0.7)

        # Fill node with oscillating pattern
        node_start = leaf_idx * NODE_SIZE
        for i in range(NODE_SIZE):
            t = i / NODE_SIZE
            val = int(h_amp * np.sin(2 * np.pi * t * h))
            coeffs[node_start + i] = sat16(coeffs[node_start + i] + val)

    return coeffs


# =============================================================================
# STANDARD PIPELINE RENDERERS
# =============================================================================

class StandardPipelineRenderer:
    """
    Standard Baby Fourier renderer using built-in anti-click features.

    Uses:
    - Persistent WaveletEngine with RegularityOperator (coefficient smoothing,
      micro-bridge at transitions, slope projection)
    - SafetyEngine for declick and limiting
    """

    def __init__(self):
        self.wavelet = WaveletEngine(block_size=BLOCK_SIZE)
        self.safety = SafetyEngine(block_size=BLOCK_SIZE)
        self.params = ParamFrame()

        # Allocate voice 0
        self.wavelet.voice_allocate(0)
        self.safety.voice_on(0)

    def render_block(
        self,
        freq: float,
        amplitude: int = 8000,
        tree_id: int = 1,
    ) -> Tuple[List[int], List[int]]:
        """
        Render one block through the standard pipeline.

        Returns (L, R) stereo samples.
        """
        coeffs = generate_coeffs(freq, amplitude)

        vf = VoiceFrame(
            voice_id=0,
            tree_a_id=tree_id,
            tree_b_id=tree_id,
            tree_blend_q15=0,
            gain_q15=Q15_HALF,
            pan_q15=0,
            coeff_q15=coeffs,
        )

        # Render through WaveletEngine (applies RegularityOperator internally)
        audio = self.wavelet.render_voice(vf, apply_regularity=True)

        # Apply SafetyEngine declick
        audio = self.safety.apply_declick(audio, 0)

        # Mix to stereo (mono duplicated)
        L, R = self.wavelet.mix_block([(vf, audio)])

        # Apply SafetyEngine limiting
        L, R = self.safety.step_block(L, R, self.params)

        return L, R


class UnifiedPipelineRenderer:
    """
    Unified pipeline renderer using StandardWaveletPipeline.

    This is the RECOMMENDED renderer that uses dual-render crossfade
    for pitch transitions through the UnifiedRenderer wrapper.

    Uses:
    - StandardWaveletPipeline (UnifiedRenderer + WaveletBackend + SafetyEngine)
    - Automatic dual-render crossfade on pitch changes
    """

    def __init__(self):
        self.pipeline = StandardWaveletPipeline(block_size=BLOCK_SIZE)
        self.pipeline.voice_on(0)
        self._last_note = None

    def render_block(
        self,
        freq: float,
        amplitude: int = 8000,
        tree_id: int = 1,
    ) -> Tuple[List[int], List[int]]:
        """
        Render one block through the unified pipeline.

        Returns (L, R) stereo samples.
        """
        # Convert frequency to MIDI note
        import math
        note = int(round(69 + 12 * math.log2(freq / 440.0)))
        note = max(0, min(127, note))

        coeffs = generate_coeffs(freq, amplitude)

        L, R = self.pipeline.render_block(
            voice_id=0,
            note=note,
            coeffs=coeffs,
            gain_q15=Q15_HALF,
            tree_a_id=tree_id,
            tree_b_id=tree_id,
        )

        self._last_note = note
        return L, R


# Default to unified pipeline (recommended)
DefaultPipelineRenderer = UnifiedPipelineRenderer


# =============================================================================
# PATTERN GENERATORS
# =============================================================================

def generate_chromatic_scale(
    start_note: int = 48,  # C3
    end_note: int = 72,    # C5
    blocks_per_note: int = None,
    duration_per_note_sec: float = 0.25,
) -> List[Tuple[int, bool]]:
    """
    Generate chromatic scale pattern.

    Returns list of (midi_note, is_transition) for each block.
    """
    if blocks_per_note is None:
        blocks_per_note = int(duration_per_note_sec * BLOCKS_PER_SEC)

    pattern = []
    for note in range(start_note, end_note + 1):
        for block in range(blocks_per_note):
            is_transition = (block == 0)
            pattern.append((note, is_transition))

    return pattern


def generate_arpeggio(
    root_note: int = 48,   # C3
    pattern: List[int] = None,  # Semitone offsets
    octaves: int = 2,
    blocks_per_note: int = None,
    duration_per_note_sec: float = 0.125,
) -> List[Tuple[int, bool]]:
    """
    Generate arpeggio pattern.

    Default pattern is major triad: [0, 4, 7, 12] (root, 3rd, 5th, octave)
    """
    if pattern is None:
        pattern = [0, 4, 7, 12]  # Major triad + octave

    if blocks_per_note is None:
        blocks_per_note = int(duration_per_note_sec * BLOCKS_PER_SEC)

    result = []
    for octave in range(octaves):
        for offset in pattern:
            note = root_note + octave * 12 + offset
            for block in range(blocks_per_note):
                is_transition = (block == 0)
                result.append((note, is_transition))
        # Descending
        for offset in reversed(pattern[:-1]):
            note = root_note + octave * 12 + offset
            for block in range(blocks_per_note):
                is_transition = (block == 0)
                result.append((note, is_transition))

    return result


def generate_rapid_transitions(
    notes: List[int] = None,
    transitions_per_sec: int = 100,
    duration_sec: float = 2.0,
) -> List[Tuple[int, bool]]:
    """
    Generate rapid note transitions for stress testing.

    Alternates between notes at specified rate.
    """
    if notes is None:
        notes = [48, 60, 55, 67]  # C3, C4, G3, G4

    total_blocks = int(duration_sec * BLOCKS_PER_SEC)
    blocks_per_transition = max(1, BLOCKS_PER_SEC // transitions_per_sec)

    result = []
    note_idx = 0
    for block in range(total_blocks):
        is_transition = (block % blocks_per_transition == 0)
        if is_transition and block > 0:
            note_idx = (note_idx + 1) % len(notes)
        result.append((notes[note_idx], is_transition))

    return result


def generate_classic_synth_arpeggio(
    root_note: int = 48,
    bpm: int = 140,
    note_division: int = 16,  # 16th notes
    duration_sec: float = 4.0,
) -> List[Tuple[int, bool]]:
    """
    Generate classic synth arpeggio (like 80s synth patterns).

    Uses minor 7th arpeggio: [0, 3, 7, 10, 12, 10, 7, 3]
    """
    # Classic minor 7th pattern
    pattern = [0, 3, 7, 10, 12, 10, 7, 3]

    # Calculate timing
    beats_per_sec = bpm / 60.0
    notes_per_sec = beats_per_sec * (note_division / 4)
    blocks_per_note = max(1, int(BLOCKS_PER_SEC / notes_per_sec))

    total_blocks = int(duration_sec * BLOCKS_PER_SEC)

    result = []
    pattern_idx = 0
    blocks_since_transition = 0

    for block in range(total_blocks):
        if blocks_since_transition >= blocks_per_note:
            pattern_idx = (pattern_idx + 1) % len(pattern)
            blocks_since_transition = 0

        is_transition = (blocks_since_transition == 0)
        note = root_note + pattern[pattern_idx]
        result.append((note, is_transition))
        blocks_since_transition += 1

    return result


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_pattern_test(
    name: str,
    pattern: List[Tuple[int, bool]],
    tree_id: int = 1,
    use_unified: bool = True,
) -> Tuple[List[int], List[int], Dict]:
    """
    Run a pattern through pipeline with safety processing.

    Args:
        name: Test name
        pattern: List of (midi_note, is_transition) tuples
        tree_id: Wavelet tree ID
        use_unified: If True (default), use UnifiedRenderer with dual-render crossfade.
                    If False, use legacy StandardPipelineRenderer.

    Returns (L, R, metrics)
    """
    if use_unified:
        renderer = UnifiedPipelineRenderer()
    else:
        renderer = StandardPipelineRenderer()

    all_L = []
    all_R = []

    transition_count = 0

    for midi_note, is_transition in pattern:
        freq = midi_to_freq(midi_note)

        if is_transition:
            transition_count += 1

        # Render through standard pipeline
        L, R = renderer.render_block(
            freq=freq,
            amplitude=8000,
            tree_id=tree_id,
        )

        all_L.extend(L)
        all_R.extend(R)

    # Compute metrics
    L_arr = np.array(all_L, dtype=np.float32)
    R_arr = np.array(all_R, dtype=np.float32)

    # Click detection
    deltas = np.abs(np.diff(L_arr))
    clicks_3000 = np.sum(deltas > 3000)
    clicks_5000 = np.sum(deltas > 5000)
    max_delta = np.max(deltas) if len(deltas) > 0 else 0

    metrics = {
        'samples': len(all_L),
        'duration_sec': len(all_L) / FS,
        'transitions': transition_count,
        'peak_L': int(np.max(np.abs(L_arr))) if len(L_arr) > 0 else 0,
        'peak_R': int(np.max(np.abs(R_arr))) if len(R_arr) > 0 else 0,
        'rms': int(np.sqrt(np.mean(L_arr ** 2))) if len(L_arr) > 0 else 0,
        'clicks_3000': int(clicks_3000),
        'clicks_5000': int(clicks_5000),
        'max_delta': int(max_delta),
    }

    return all_L, all_R, metrics


def save_audio(L: List[int], R: List[int], name: str) -> str:
    """Save audio and return filepath."""
    metadata = write_audio(name, L, R, timestamp=True,
                          subdirectory=OUTPUT_SUBDIR, verbose=False)
    return metadata.filepath


# =============================================================================
# MAIN TEST SUITE
# =============================================================================

def run_all_tests():
    """Run all safety tests with unified pipeline (default)."""

    print("=" * 70)
    print("SAFETY ENGINE TEST - UNIFIED PIPELINE (DEFAULT)")
    print("(UnifiedRenderer + WaveletBackend + SafetyEngine)")
    print("  * Dual-render crossfade on pitch transitions")
    print("  * RegularityOperator coefficient smoothing")
    print("  * SafetyEngine limiting")
    print("=" * 70)
    print()

    results = {}

    # 1. Chromatic Scale C3-C5
    print("1. Chromatic Scale C3-C5 (250ms per note)")
    print("-" * 50)
    pattern = generate_chromatic_scale(48, 72, duration_per_note_sec=0.25)
    L, R, metrics = run_pattern_test("chromatic_c3_c5", pattern)
    filepath = save_audio(L, R, "safety_chromatic_c3_c5")
    results['chromatic'] = metrics
    print(f"   Duration: {metrics['duration_sec']:.2f}s, Transitions: {metrics['transitions']}")
    print(f"   Peak: {metrics['peak_L']}, RMS: {metrics['rms']}")
    print(f"   Clicks >3000: {metrics['clicks_3000']}, >5000: {metrics['clicks_5000']}")
    print(f"   Max delta: {metrics['max_delta']}")
    print(f"   Saved: {filepath}")
    print()

    # 2. Major Arpeggio
    print("2. Major Arpeggio (125ms per note, 2 octaves)")
    print("-" * 50)
    pattern = generate_arpeggio(48, [0, 4, 7, 12], octaves=2, duration_per_note_sec=0.125)
    L, R, metrics = run_pattern_test("arpeggio_major", pattern)
    filepath = save_audio(L, R, "safety_arpeggio_major")
    results['arpeggio_major'] = metrics
    print(f"   Duration: {metrics['duration_sec']:.2f}s, Transitions: {metrics['transitions']}")
    print(f"   Peak: {metrics['peak_L']}, RMS: {metrics['rms']}")
    print(f"   Clicks >3000: {metrics['clicks_3000']}, >5000: {metrics['clicks_5000']}")
    print(f"   Max delta: {metrics['max_delta']}")
    print(f"   Saved: {filepath}")
    print()

    # 3. Minor 7th Arpeggio (wider intervals)
    print("3. Minor 7th Arpeggio (100ms per note)")
    print("-" * 50)
    pattern = generate_arpeggio(48, [0, 3, 7, 10, 12], octaves=2, duration_per_note_sec=0.1)
    L, R, metrics = run_pattern_test("arpeggio_min7", pattern)
    filepath = save_audio(L, R, "safety_arpeggio_min7")
    results['arpeggio_min7'] = metrics
    print(f"   Duration: {metrics['duration_sec']:.2f}s, Transitions: {metrics['transitions']}")
    print(f"   Peak: {metrics['peak_L']}, RMS: {metrics['rms']}")
    print(f"   Clicks >3000: {metrics['clicks_3000']}, >5000: {metrics['clicks_5000']}")
    print(f"   Max delta: {metrics['max_delta']}")
    print(f"   Saved: {filepath}")
    print()

    # 4. Rapid Transitions x50
    print("4. Rapid Transitions (50/sec)")
    print("-" * 50)
    pattern = generate_rapid_transitions(
        notes=[48, 60, 55, 67, 52, 64],
        transitions_per_sec=50,
        duration_sec=2.0
    )
    L, R, metrics = run_pattern_test("rapid_50", pattern)
    filepath = save_audio(L, R, "safety_rapid_50")
    results['rapid_50'] = metrics
    print(f"   Duration: {metrics['duration_sec']:.2f}s, Transitions: {metrics['transitions']}")
    print(f"   Peak: {metrics['peak_L']}, RMS: {metrics['rms']}")
    print(f"   Clicks >3000: {metrics['clicks_3000']}, >5000: {metrics['clicks_5000']}")
    print(f"   Max delta: {metrics['max_delta']}")
    print(f"   Saved: {filepath}")
    print()

    # 5. Rapid Transitions x100 (stress test)
    print("5. Rapid Transitions x100 (STRESS TEST)")
    print("-" * 50)
    pattern = generate_rapid_transitions(
        notes=[48, 60, 55, 67, 52, 64, 59, 71],
        transitions_per_sec=100,
        duration_sec=2.0
    )
    L, R, metrics = run_pattern_test("rapid_100", pattern)
    filepath = save_audio(L, R, "safety_rapid_100")
    results['rapid_100'] = metrics
    print(f"   Duration: {metrics['duration_sec']:.2f}s, Transitions: {metrics['transitions']}")
    print(f"   Peak: {metrics['peak_L']}, RMS: {metrics['rms']}")
    print(f"   Clicks >3000: {metrics['clicks_3000']}, >5000: {metrics['clicks_5000']}")
    print(f"   Max delta: {metrics['max_delta']}")
    print(f"   Saved: {filepath}")
    print()

    # 6. Classic Synth Arpeggio 120 BPM
    print("6. Classic Synth Arpeggio (120 BPM, 16th notes)")
    print("-" * 50)
    pattern = generate_classic_synth_arpeggio(48, bpm=120, note_division=16, duration_sec=4.0)
    L, R, metrics = run_pattern_test("synth_120", pattern)
    filepath = save_audio(L, R, "safety_synth_arp_120bpm")
    results['synth_120'] = metrics
    print(f"   Duration: {metrics['duration_sec']:.2f}s, Transitions: {metrics['transitions']}")
    print(f"   Peak: {metrics['peak_L']}, RMS: {metrics['rms']}")
    print(f"   Clicks >3000: {metrics['clicks_3000']}, >5000: {metrics['clicks_5000']}")
    print(f"   Max delta: {metrics['max_delta']}")
    print(f"   Saved: {filepath}")
    print()

    # 7. Classic Synth Arpeggio 140 BPM (faster)
    print("7. Classic Synth Arpeggio (140 BPM, 16th notes)")
    print("-" * 50)
    pattern = generate_classic_synth_arpeggio(48, bpm=140, note_division=16, duration_sec=4.0)
    L, R, metrics = run_pattern_test("synth_140", pattern)
    filepath = save_audio(L, R, "safety_synth_arp_140bpm")
    results['synth_140'] = metrics
    print(f"   Duration: {metrics['duration_sec']:.2f}s, Transitions: {metrics['transitions']}")
    print(f"   Peak: {metrics['peak_L']}, RMS: {metrics['rms']}")
    print(f"   Clicks >3000: {metrics['clicks_3000']}, >5000: {metrics['clicks_5000']}")
    print(f"   Max delta: {metrics['max_delta']}")
    print(f"   Saved: {filepath}")
    print()

    # 8. Classic Synth Arpeggio 180 BPM (very fast)
    print("8. Classic Synth Arpeggio (180 BPM, 16th notes - FAST)")
    print("-" * 50)
    pattern = generate_classic_synth_arpeggio(48, bpm=180, note_division=16, duration_sec=4.0)
    L, R, metrics = run_pattern_test("synth_180", pattern)
    filepath = save_audio(L, R, "safety_synth_arp_180bpm")
    results['synth_180'] = metrics
    print(f"   Duration: {metrics['duration_sec']:.2f}s, Transitions: {metrics['transitions']}")
    print(f"   Peak: {metrics['peak_L']}, RMS: {metrics['rms']}")
    print(f"   Clicks >3000: {metrics['clicks_3000']}, >5000: {metrics['clicks_5000']}")
    print(f"   Max delta: {metrics['max_delta']}")
    print(f"   Saved: {filepath}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Test':<25} {'Trans':>7} {'Peak':>7} {'RMS':>7} {'>3k':>7} {'>5k':>7} {'MaxD':>7}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<25} {m['transitions']:>7} {m['peak_L']:>7} {m['rms']:>7} "
              f"{m['clicks_3000']:>7} {m['clicks_5000']:>7} {m['max_delta']:>7}")
    print()
    print("All files saved to audio_generation_tests/audio_out/safety_tests/")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_all_tests()
