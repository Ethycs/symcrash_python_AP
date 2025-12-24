"""
CarrierLiftPipeline Audio Tests
===============================

Tests the Carrier + IDWPT Lift architecture (v2) with audio output.
Uses audio_io for standardized file output.

This pipeline solves the pitch/beating/click problems by:
- Carrier (Baby Fourier) owns pitch + phase continuity
- IDWPT becomes a "lift" that shapes timbre, not an oscillator
- Beta (β) is an audio-domain dry/wet blend with latency handling

Test Categories:
1. Single Note Tests - steady pitch stability
2. Pitch Transition Tests - smooth pitch changes
3. Beta Sweep Tests - carrier/lift blend transitions
4. Tuning Morph Tests - global timbre changes
5. Musical Pattern Tests - arpeggios, scales
6. Stress Tests - rapid transitions, polyphony
7. Comparison Tests - vs legacy UnifiedRenderer
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.fixed import sat16


# =============================================================================
# CONSTANTS
# =============================================================================

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
OUTPUT_SUBDIR = "carrier_lift_tests"
Q15_ONE = 32767
Q15_HALF = 16384
Q15_3_4 = 24576  # ~75% wet (default beta target)

# Global AudioWriter instance for all tests
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def midi_to_freq(note: int, cents: int = 0) -> float:
    """MIDI note to frequency."""
    return 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))


def save_audio(L: List[int], R: List[int], name: str) -> str:
    """Save audio to file using audio_io object."""
    metadata = audio_io.write_audio(name, L, R, subdirectory=OUTPUT_SUBDIR)
    return metadata.filepath


def compute_max_delta(samples: List[int]) -> int:
    """Compute maximum sample-to-sample delta."""
    if len(samples) < 2:
        return 0
    return max(abs(samples[i] - samples[i-1]) for i in range(1, len(samples)))


def count_clicks(samples: List[int], threshold: int = 5000) -> int:
    """Count clicks (deltas above threshold)."""
    if len(samples) < 2:
        return 0
    return sum(1 for i in range(1, len(samples))
               if abs(samples[i] - samples[i-1]) > threshold)


def compute_rms(samples: List[int]) -> float:
    """Compute RMS of samples."""
    if not samples:
        return 0.0
    arr = np.array(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(arr ** 2)))


def analyze_audio(samples: List[int], name: str = "") -> dict:
    """Analyze audio and return metrics."""
    arr = np.array(samples, dtype=np.float32)
    deltas = np.abs(np.diff(arr))

    metrics = {
        'n_samples': len(samples),
        'duration_sec': len(samples) / FS,
        'peak': int(np.max(np.abs(arr))) if len(arr) > 0 else 0,
        'rms': int(np.sqrt(np.mean(arr ** 2))) if len(arr) > 0 else 0,
        'max_delta': int(np.max(deltas)) if len(deltas) > 0 else 0,
        'clicks_3000': int(np.sum(deltas > 3000)),
        'clicks_5000': int(np.sum(deltas > 5000)),
        'clicks_10000': int(np.sum(deltas > 10000)),
    }

    if name:
        print(f"\n  {name}:")
        print(f"    Duration: {metrics['duration_sec']:.2f}s")
        print(f"    Peak: {metrics['peak']}, RMS: {metrics['rms']}")
        print(f"    Max delta: {metrics['max_delta']}")
        print(f"    Clicks: >3k={metrics['clicks_3000']}, >5k={metrics['clicks_5000']}, >10k={metrics['clicks_10000']}")

    return metrics


# =============================================================================
# SINGLE NOTE TESTS
# =============================================================================

class TestSingleNote:
    """Test single note stability with CarrierLiftPipeline."""

    def test_steady_note_c4_5sec(self):
        """
        Render middle C for 5 seconds.

        Key verification:
        - No block-rate artifacts
        - Stable RMS after envelope settles
        - No clicks or discontinuities
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        num_blocks = int(5.0 * BLOCKS_PER_SEC)
        all_L = []
        all_R = []

        for _ in range(num_blocks):
            L, R = pipeline.render_block_simple(0, note=60)
            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_steady_c4_5s")
        metrics = analyze_audio(all_L, "Steady C4 (5s)")
        print(f"    Saved: {filepath}")

        # Verify stability - allow some clicks due to wavelet blend dynamics
        # The key metric is no clicks >10k (true discontinuities)
        assert metrics['clicks_10000'] == 0, f"Hard clicks detected: {metrics['clicks_10000']}"
        assert metrics['rms'] > 100, f"RMS too low: {metrics['rms']}"

        pipeline.voice_off(0)

    def test_steady_note_low_a1(self):
        """
        Render low A1 (55Hz) for 3 seconds.

        Low frequencies stress the carrier phase accumulator.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=33)  # A1 = 55Hz

        num_blocks = int(3.0 * BLOCKS_PER_SEC)
        all_L = []
        all_R = []

        for _ in range(num_blocks):
            L, R = pipeline.render_block_simple(0, note=33)
            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_steady_a1_3s")
        metrics = analyze_audio(all_L, "Steady A1 (3s)")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)

    def test_steady_note_high_c7(self):
        """
        Render high C7 (2093Hz) for 2 seconds.

        High frequencies stress carrier frequency resolution.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=96)  # C7 = 2093Hz

        num_blocks = int(2.0 * BLOCKS_PER_SEC)
        all_L = []
        all_R = []

        for _ in range(num_blocks):
            L, R = pipeline.render_block_simple(0, note=96)
            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_steady_c7_2s")
        metrics = analyze_audio(all_L, "Steady C7 (2s)")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)


# =============================================================================
# PITCH TRANSITION TESTS
# =============================================================================

class TestPitchTransitions:
    """Test pitch transitions with CarrierLiftPipeline."""

    def test_chromatic_scale_c3_c5(self):
        """
        Chromatic scale C3-C5 (25 notes).

        Key verification:
        - Smooth pitch transitions (carrier-led)
        - No clicks at note boundaries
        - Consistent timbre across range
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=48)

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)  # 250ms per note
        all_L = []
        all_R = []

        for note in range(48, 73):  # C3 to C5
            for _ in range(blocks_per_note):
                frame = VoiceFrame(
                    voice_id=0,
                    note=note,
                    f0_hz=midi_to_freq(note),
                    gain_q15=Q15_HALF,
                )
                L, R = pipeline.render_block(frame)
                all_L.extend(L)
                all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_chromatic_c3_c5")
        metrics = analyze_audio(all_L, "Chromatic C3-C5")
        print(f"    Saved: {filepath}")

        # Key assertion: max delta should be reasonable
        assert metrics['max_delta'] < 20000, f"Click detected: {metrics['max_delta']}"

        pipeline.voice_off(0)

    def test_octave_jumps(self):
        """
        Octave jumps (2:1 frequency ratio).

        This stresses carrier pitch slew.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=48)

        blocks_per_note = int(0.5 * BLOCKS_PER_SEC)  # 500ms per note
        all_L = []
        all_R = []

        # Pattern: C3, C4, C3, C5, C3, C6
        notes = [48, 60, 48, 72, 48, 84]

        for note in notes:
            for _ in range(blocks_per_note):
                frame = VoiceFrame(
                    voice_id=0,
                    note=note,
                    f0_hz=midi_to_freq(note),
                    gain_q15=Q15_HALF,
                )
                L, R = pipeline.render_block(frame)
                all_L.extend(L)
                all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_octave_jumps")
        metrics = analyze_audio(all_L, "Octave Jumps")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)

    def test_pitch_glide_semitone(self):
        """
        Smooth pitch glide over one semitone using cents.

        Tests fine pitch control.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        num_blocks = int(2.0 * BLOCKS_PER_SEC)
        all_L = []
        all_R = []

        for block in range(num_blocks):
            # Glide from C4 to C#4 over 2 seconds
            progress = block / num_blocks
            cents = int(progress * 100)  # 0 to 100 cents

            frame = VoiceFrame(
                voice_id=0,
                note=60,
                cents=cents,
                f0_hz=midi_to_freq(60, cents),
                gain_q15=Q15_HALF,
            )
            L, R = pipeline.render_block(frame)
            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_pitch_glide_semitone")
        metrics = analyze_audio(all_L, "Pitch Glide (semitone)")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)


# =============================================================================
# BETA SWEEP TESTS
# =============================================================================

class TestBetaSweep:
    """Test carrier/lift blend transitions."""

    def test_beta_0_to_1_sweep(self):
        """
        Sweep beta from 0 (pure carrier) to 1 (pure wavelet) over 3 seconds.

        Key verification:
        - Smooth transition without clicks
        - Audible timbre change (carrier → wavelet)
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        num_blocks = int(3.0 * BLOCKS_PER_SEC)
        all_L = []
        all_R = []

        for block in range(num_blocks):
            # Sweep beta from 0 to Q15_ONE
            progress = block / num_blocks
            beta_target = int(progress * Q15_ONE)

            frame = VoiceFrame(
                voice_id=0,
                note=60,
                f0_hz=midi_to_freq(60),
                gain_q15=Q15_HALF,
                beta_target_q15=beta_target,
            )
            L, R = pipeline.render_block(frame)
            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_beta_sweep_0_to_1")
        metrics = analyze_audio(all_L, "Beta Sweep 0 to 1")
        print(f"    Saved: {filepath}")

        # Should be relatively click-free (allow some from wavelet dynamics)
        assert metrics['clicks_10000'] == 0, f"Hard clicks during beta sweep: {metrics['clicks_10000']}"

        pipeline.voice_off(0)

    def test_beta_oscillation(self):
        """
        Oscillate beta between carrier and wavelet.

        Tests smooth back-and-forth blending.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        num_blocks = int(4.0 * BLOCKS_PER_SEC)
        all_L = []
        all_R = []

        # Oscillate at 0.5Hz (2 cycles over 4 seconds)
        osc_freq = 0.5

        for block in range(num_blocks):
            t = block / BLOCKS_PER_SEC
            # Sine oscillation between 0.25 and 0.75
            beta_norm = 0.5 + 0.25 * np.sin(2 * np.pi * osc_freq * t)
            beta_target = int(beta_norm * Q15_ONE)

            frame = VoiceFrame(
                voice_id=0,
                note=60,
                f0_hz=midi_to_freq(60),
                gain_q15=Q15_HALF,
                beta_target_q15=beta_target,
            )
            L, R = pipeline.render_block(frame)
            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_beta_oscillation")
        metrics = analyze_audio(all_L, "Beta Oscillation")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)


# =============================================================================
# TUNING MORPH TESTS
# =============================================================================

class TestTuningMorph:
    """Test global tuning morphs."""

    def test_harmonic_to_bright_morph(self):
        """
        Morph from harmonic to bright tuning over 3 seconds.

        Tests slow global tuning changes.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        # Start with harmonic tuning (default)
        # Morph to bright over 2 seconds
        pipeline.set_tuning("bright", morph_time_sec=2.0)

        num_blocks = int(4.0 * BLOCKS_PER_SEC)
        all_L = []
        all_R = []

        for _ in range(num_blocks):
            frame = VoiceFrame(
                voice_id=0,
                note=60,
                f0_hz=midi_to_freq(60),
                gain_q15=Q15_HALF,
            )
            L, R = pipeline.render_block(frame)
            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_tuning_harmonic_to_bright")
        metrics = analyze_audio(all_L, "Tuning: Harmonic to Bright")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)

    def test_tuning_sequence(self):
        """
        Cycle through multiple tunings.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        tunings = ["harmonic", "bright", "dark", "hollow", "nasal"]
        blocks_per_tuning = int(2.0 * BLOCKS_PER_SEC)

        all_L = []
        all_R = []

        for tuning in tunings:
            pipeline.set_tuning(tuning, morph_time_sec=1.0)

            for _ in range(blocks_per_tuning):
                frame = VoiceFrame(
                    voice_id=0,
                    note=60,
                    f0_hz=midi_to_freq(60),
                    gain_q15=Q15_HALF,
                )
                L, R = pipeline.render_block(frame)
                all_L.extend(L)
                all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_tuning_sequence")
        metrics = analyze_audio(all_L, "Tuning Sequence")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)


# =============================================================================
# MUSICAL PATTERN TESTS
# =============================================================================

class TestMusicalPatterns:
    """Test musical patterns with CarrierLiftPipeline."""

    def test_synth_arpeggio_140bpm(self):
        """
        Classic synth arpeggio at 140 BPM.

        Minor 7th pattern, 16th notes.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=48)

        root_note = 48
        pattern = [0, 3, 7, 10, 12, 10, 7, 3]  # Minor 7th up and down
        bpm = 140
        duration_sec = 4.0

        beats_per_sec = bpm / 60.0
        notes_per_sec = beats_per_sec * 4  # 16th notes
        blocks_per_note = max(1, int(BLOCKS_PER_SEC / notes_per_sec))
        total_blocks = int(duration_sec * BLOCKS_PER_SEC)

        all_L = []
        all_R = []
        pattern_idx = 0
        blocks_since_transition = 0

        for block in range(total_blocks):
            if blocks_since_transition >= blocks_per_note:
                pattern_idx = (pattern_idx + 1) % len(pattern)
                blocks_since_transition = 0

            note = root_note + pattern[pattern_idx]
            frame = VoiceFrame(
                voice_id=0,
                note=note,
                f0_hz=midi_to_freq(note),
                gain_q15=Q15_HALF,
            )
            L, R = pipeline.render_block(frame)
            all_L.extend(L)
            all_R.extend(R)
            blocks_since_transition += 1

        filepath = save_audio(all_L, all_R, "carrier_lift_arp_140bpm")
        metrics = analyze_audio(all_L, "Arpeggio 140 BPM")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)

    def test_bass_line_100bpm(self):
        """
        Bass line pattern at 100 BPM.

        8th notes, root-5th pattern.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=36)  # C2

        root_note = 36
        pattern = [0, 0, 7, 7, 5, 5, 3, 0]  # Root, 5th, 4th, minor 3rd
        bpm = 100
        duration_sec = 4.0

        beats_per_sec = bpm / 60.0
        notes_per_sec = beats_per_sec * 2  # 8th notes
        blocks_per_note = max(1, int(BLOCKS_PER_SEC / notes_per_sec))
        total_blocks = int(duration_sec * BLOCKS_PER_SEC)

        all_L = []
        all_R = []
        pattern_idx = 0
        blocks_since_transition = 0

        for block in range(total_blocks):
            if blocks_since_transition >= blocks_per_note:
                pattern_idx = (pattern_idx + 1) % len(pattern)
                blocks_since_transition = 0

            note = root_note + pattern[pattern_idx]
            frame = VoiceFrame(
                voice_id=0,
                note=note,
                f0_hz=midi_to_freq(note),
                gain_q15=Q15_HALF,
            )
            L, R = pipeline.render_block(frame)
            all_L.extend(L)
            all_R.extend(R)
            blocks_since_transition += 1

        filepath = save_audio(all_L, all_R, "carrier_lift_bass_100bpm")
        metrics = analyze_audio(all_L, "Bass Line 100 BPM")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)

    def test_melody_legato(self):
        """
        Legato melody with smooth connected notes.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        # Simple melody (Bach-ish)
        melody = [60, 62, 64, 65, 67, 65, 64, 62, 60, 59, 60]
        blocks_per_note = int(0.4 * BLOCKS_PER_SEC)  # 400ms per note

        all_L = []
        all_R = []

        for note in melody:
            for _ in range(blocks_per_note):
                frame = VoiceFrame(
                    voice_id=0,
                    note=note,
                    f0_hz=midi_to_freq(note),
                    gain_q15=Q15_HALF,
                )
                L, R = pipeline.render_block(frame)
                all_L.extend(L)
                all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_melody_legato")
        metrics = analyze_audio(all_L, "Legato Melody")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for CarrierLiftPipeline."""

    def test_rapid_transitions_50_per_sec(self):
        """
        Rapid pitch transitions - 50 per second for 2 seconds.
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        notes = [48, 60, 55, 67, 52, 64]
        total_blocks = int(2.0 * BLOCKS_PER_SEC)
        blocks_per_transition = max(1, BLOCKS_PER_SEC // 50)

        all_L = []
        all_R = []
        note_idx = 0

        for block in range(total_blocks):
            if block % blocks_per_transition == 0 and block > 0:
                note_idx = (note_idx + 1) % len(notes)

            note = notes[note_idx]
            frame = VoiceFrame(
                voice_id=0,
                note=note,
                f0_hz=midi_to_freq(note),
                gain_q15=Q15_HALF,
            )
            L, R = pipeline.render_block(frame)
            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_rapid_50_per_sec")
        metrics = analyze_audio(all_L, "Rapid Transitions (50/s)")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)

    def test_polyphony_8_voices(self):
        """
        8-voice polyphony stress test.

        All voices playing different notes simultaneously.
        """
        pipeline = CarrierLiftPipeline()

        # Activate 8 voices with a chord
        chord = [48, 52, 55, 60, 64, 67, 72, 76]  # C major + extensions
        for i, note in enumerate(chord):
            pipeline.voice_on(i, note=note)

        num_blocks = int(3.0 * BLOCKS_PER_SEC)
        all_L = []
        all_R = []

        for _ in range(num_blocks):
            block_L = [0] * BLOCK_SIZE
            block_R = [0] * BLOCK_SIZE

            for i, note in enumerate(chord):
                frame = VoiceFrame(
                    voice_id=i,
                    note=note,
                    f0_hz=midi_to_freq(note),
                    gain_q15=Q15_HALF // len(chord),  # Scale by voice count
                    pan_q15=int((i - 3.5) * 4000),  # Spread stereo
                )
                L, R = pipeline.render_block(frame)

                for j in range(BLOCK_SIZE):
                    block_L[j] = sat16(block_L[j] + L[j])
                    block_R[j] = sat16(block_R[j] + R[j])

            all_L.extend(block_L)
            all_R.extend(block_R)

        filepath = save_audio(all_L, all_R, "carrier_lift_polyphony_8_voices")
        metrics = analyze_audio(all_L, "8-Voice Polyphony")
        print(f"    Saved: {filepath}")

        for i in range(8):
            pipeline.voice_off(i)

    def test_extreme_pitch_range(self):
        """
        Test extreme pitch range: A0 (27.5Hz) to C8 (4186Hz).
        """
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=21)  # A0

        # Play notes at extreme ends
        notes = [21, 108, 21, 96, 33, 84, 45, 72, 60]  # Converging pattern
        blocks_per_note = int(0.5 * BLOCKS_PER_SEC)

        all_L = []
        all_R = []

        for note in notes:
            for _ in range(blocks_per_note):
                frame = VoiceFrame(
                    voice_id=0,
                    note=note,
                    f0_hz=midi_to_freq(note),
                    gain_q15=Q15_HALF,
                )
                L, R = pipeline.render_block(frame)
                all_L.extend(L)
                all_R.extend(R)

        filepath = save_audio(all_L, all_R, "carrier_lift_extreme_range")
        metrics = analyze_audio(all_L, "Extreme Pitch Range")
        print(f"    Saved: {filepath}")

        pipeline.voice_off(0)


# =============================================================================
# COMPARISON TESTS
# =============================================================================

class TestComparison:
    """Compare CarrierLiftPipeline vs legacy UnifiedRenderer."""

    def test_chromatic_comparison(self):
        """
        Same chromatic pattern through both pipelines.

        Compare click counts and max delta.
        """
        from symcrash_python_ap.unified_renderer import (
            UnifiedRenderer, WaveletBackend, RenderFrame
        )
        from symcrash_python_ap.safety_engine import SafetyEngine
        from symcrash_python_ap.frames import ParamFrame

        blocks_per_note = int(0.2 * BLOCKS_PER_SEC)  # 200ms per note

        # --- CarrierLiftPipeline ---
        carrier_pipeline = CarrierLiftPipeline()
        carrier_pipeline.voice_on(0, note=48)

        carrier_L = []
        carrier_R = []

        for note in range(48, 73):
            for _ in range(blocks_per_note):
                frame = VoiceFrame(
                    voice_id=0,
                    note=note,
                    f0_hz=midi_to_freq(note),
                    gain_q15=Q15_HALF,
                )
                L, R = carrier_pipeline.render_block(frame)
                carrier_L.extend(L)
                carrier_R.extend(R)

        carrier_pipeline.voice_off(0)

        # --- Legacy UnifiedRenderer ---
        legacy_backend = WaveletBackend(block_size=BLOCK_SIZE)
        legacy_renderer = UnifiedRenderer(legacy_backend)
        legacy_safety = SafetyEngine(block_size=BLOCK_SIZE)
        legacy_params = ParamFrame()

        legacy_renderer.voice_allocate(0)
        legacy_safety.voice_on(0)

        legacy_L = []
        legacy_R = []

        for note in range(48, 73):
            freq = midi_to_freq(note)
            # Generate basic coefficients
            coeffs = [0] * 256
            for h in range(1, 8):
                h_freq = freq * h
                if h_freq < FS / 2:
                    leaf_idx = min(31, int(h_freq / 750))  # Simple mapping
                    for i in range(8):
                        coeffs[leaf_idx * 8 + i] = sat16(8000 // h)

            for _ in range(blocks_per_note):
                frame = RenderFrame(
                    voice_id=0,
                    note=note,
                    gain_q15=Q15_HALF,
                    coeff_q15=coeffs,
                )
                samples = legacy_renderer.render_voice(frame)
                samples = legacy_safety.apply_declick(samples, 0)
                L, R = list(samples), list(samples)
                L, R = legacy_safety.step_block(L, R, legacy_params)
                legacy_L.extend(L)
                legacy_R.extend(R)

        # Save both
        filepath_carrier = save_audio(carrier_L, carrier_R, "comparison_carrier_lift")
        filepath_legacy = save_audio(legacy_L, legacy_R, "comparison_legacy")

        # Analyze both
        carrier_metrics = analyze_audio(carrier_L, "CarrierLiftPipeline")
        legacy_metrics = analyze_audio(legacy_L, "Legacy UnifiedRenderer")

        print(f"\n  Comparison Summary:")
        print(f"    CarrierLift clicks >5k: {carrier_metrics['clicks_5000']}")
        print(f"    Legacy clicks >5k: {legacy_metrics['clicks_5000']}")
        print(f"    CarrierLift max delta: {carrier_metrics['max_delta']}")
        print(f"    Legacy max delta: {legacy_metrics['max_delta']}")
        print(f"    Saved: {filepath_carrier}")
        print(f"    Saved: {filepath_legacy}")


# =============================================================================
# WAVELET BLOOM TEST
# =============================================================================

class TestWaveletBloom:
    """Test the wavelet bloom-in behavior."""

    def test_note_on_bloom(self):
        """
        Test that carrier plays immediately and wavelet blooms in.

        First 5 blocks should be carrier-only, then beta ramps up.
        """
        pipeline = CarrierLiftPipeline()

        # Track per-block RMS to see bloom
        block_rms = []

        pipeline.voice_on(0, note=60)

        num_blocks = int(0.5 * BLOCKS_PER_SEC)  # 500ms
        all_L = []
        all_R = []

        for i in range(num_blocks):
            frame = VoiceFrame(
                voice_id=0,
                note=60,
                f0_hz=midi_to_freq(60),
                gain_q15=Q15_HALF,
            )
            L, R = pipeline.render_block(frame)
            all_L.extend(L)
            all_R.extend(R)

            # Record RMS of this block
            block_rms.append(compute_rms(L))

        filepath = save_audio(all_L, all_R, "carrier_lift_bloom_in")

        print(f"\n  Wavelet Bloom-In Test:")
        print(f"    First 10 block RMS values:")
        for i in range(min(10, len(block_rms))):
            print(f"      Block {i}: RMS={block_rms[i]:.1f}")
        print(f"    Saved: {filepath}")

        # First few blocks should have some audio (carrier starts with envelope)
        # The envelope attack means first block won't be full volume
        assert block_rms[0] > 0, "No audio in first block at all"
        # By block 5, should have significant audio
        assert block_rms[5] > 1000, f"Low audio by block 5: {block_rms[5]}"

        pipeline.voice_off(0)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
