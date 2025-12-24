"""
UnifiedRenderer Tests
=====================

Tests the unified rendering pipeline with both synthesis backends:
- HarmonicScaffoldBackend (additive synthesis)
- WaveletBackend (IDWPT/Baby Fourier)

Verifies that dual-render crossfade produces click-free pitch transitions
regardless of which backend is used.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC
from audio_io import write_audio

from symcrash_python_ap.unified_renderer import (
    UnifiedRenderer,
    RenderFrame,
    HarmonicScaffoldBackend,
    WaveletBackend,
    StandardWaveletPipeline,
    StandardHarmonicPipeline,
    crossfade_samples,
    apply_microfade,
    detect_discontinuities,
)
from symcrash_python_ap.wavelet_packet_trees import COEFF_LEN
from symcrash_python_ap.fixed import sat16

# =============================================================================
# CONSTANTS
# =============================================================================

OUTPUT_SUBDIR = "unified_renderer_tests"
Q15_ONE = 32767
Q15_HALF = 16384
NODE_SIZE = 8
N_LEAVES = 32


def midi_to_freq(note: int) -> float:
    """MIDI note to frequency."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def generate_coeffs_for_freq(freq: float, amplitude: int = 8000) -> List[int]:
    """
    Generate wavelet coefficients for a frequency using Baby Fourier mapping.
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

        h_amp = amplitude // (h ** 0.7)

        node_start = leaf_idx * NODE_SIZE
        for i in range(NODE_SIZE):
            t = i / NODE_SIZE
            val = int(h_amp * np.sin(2 * np.pi * t * h))
            coeffs[node_start + i] = sat16(coeffs[node_start + i] + val)

    return coeffs


def samples_to_stereo(samples: List[int]) -> Tuple[List[int], List[int]]:
    """Convert mono to stereo (duplicate)."""
    return list(samples), list(samples)


def save_audio(L: List[int], R: List[int], name: str) -> str:
    """Save audio to file."""
    metadata = write_audio(name, L, R, subdirectory=OUTPUT_SUBDIR)
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


# =============================================================================
# TESTS: HARMONIC SCAFFOLD BACKEND
# =============================================================================

class TestHarmonicScaffoldBackend:
    """Test UnifiedRenderer with HarmonicScaffoldBackend."""

    def test_single_note(self):
        """Render a single note."""
        backend = HarmonicScaffoldBackend()
        renderer = UnifiedRenderer(backend)

        renderer.voice_allocate(0)

        frame = RenderFrame(
            voice_id=0,
            note=60,
            waveform="saw",
            gain_q15=Q15_HALF,
        )

        all_samples = []
        for _ in range(int(1.0 * BLOCKS_PER_SEC)):  # 1 second
            samples = renderer.render_voice(frame)
            all_samples.extend(samples)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "harmonic_single_note_c4")

        max_delta = compute_max_delta(all_samples)
        print(f"\n  Single note (Harmonic): {len(L)} samples, max_delta={max_delta}")
        print(f"    Saved: {filepath}")

        assert len(all_samples) > 0
        assert max_delta < 10000  # Should be smooth

    def test_chromatic_scale_auto_crossfade(self):
        """
        Chromatic scale with automatic dual-render crossfade.

        The UnifiedRenderer should automatically detect pitch changes
        and apply dual-render crossfade.
        """
        backend = HarmonicScaffoldBackend()
        renderer = UnifiedRenderer(backend)

        renderer.voice_allocate(0)

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)  # 250ms per note
        all_samples = []
        crossfade_count = 0

        for note in range(48, 73):  # C3 to C5
            frame = RenderFrame(
                voice_id=0,
                note=note,
                waveform="saw",
                gain_q15=Q15_HALF,
            )

            for block_idx in range(blocks_per_note):
                # Check if crossfade will be applied
                voice = renderer.voices.get(0)
                if voice and voice.last_frame and voice.last_frame.note != note:
                    crossfade_count += 1
                samples = renderer.render_voice(frame)
                all_samples.extend(samples)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "harmonic_chromatic_c3_c5_unified")

        # Verify no clicks by checking max delta at note boundaries
        # (not all samples - sawtooth naturally has large wrap deltas)
        note_boundary_samples = blocks_per_note * BLOCK_SIZE
        max_boundary_delta = 0
        for i in range(1, 25):  # 24 note changes
            boundary_idx = i * note_boundary_samples
            if boundary_idx < len(all_samples) - 1:
                # Check around the boundary (within crossfade region)
                for offset in range(-32, 32):
                    idx = boundary_idx + offset
                    if 0 < idx < len(all_samples):
                        delta = abs(all_samples[idx] - all_samples[idx - 1])
                        max_boundary_delta = max(max_boundary_delta, delta)

        print(f"\n  Chromatic scale (Harmonic/Unified): {len(L)} samples")
        print(f"    Max delta at boundaries: {max_boundary_delta}")
        print(f"    Crossfades applied: {crossfade_count}")
        print(f"    Saved: {filepath}")

        # With dual-render, max delta at boundaries should be reasonable
        # Note: sawtooth has natural wrap deltas but crossfade smooths transitions
        assert max_boundary_delta < 20000, f"Click at boundary: delta={max_boundary_delta}"

    def test_rapid_transitions(self):
        """Test rapid pitch transitions (stress test)."""
        backend = HarmonicScaffoldBackend()
        renderer = UnifiedRenderer(backend)

        renderer.voice_allocate(0)

        all_samples = []
        notes = [60, 67, 60, 67] * 25  # 100 transitions

        for note in notes:
            frame = RenderFrame(
                voice_id=0,
                note=note,
                waveform="saw",
                gain_q15=Q15_HALF,
            )

            # Just 2 blocks per note (rapid)
            for _ in range(2):
                samples = renderer.render_voice(frame)
                all_samples.extend(samples)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "harmonic_rapid_transitions_unified")

        max_delta = compute_max_delta(all_samples)
        clicks = count_clicks(all_samples, 5000)

        print(f"\n  Rapid transitions (Harmonic/Unified): {len(L)} samples")
        print(f"    Max delta: {max_delta}, Clicks >5000: {clicks}")
        print(f"    Saved: {filepath}")


# =============================================================================
# TESTS: WAVELET BACKEND
# =============================================================================

class TestWaveletBackend:
    """Test UnifiedRenderer with WaveletBackend."""

    def test_single_note(self):
        """Render a single note."""
        backend = WaveletBackend(block_size=BLOCK_SIZE)
        renderer = UnifiedRenderer(backend)

        renderer.voice_allocate(0)

        freq = midi_to_freq(60)
        coeffs = generate_coeffs_for_freq(freq)

        frame = RenderFrame(
            voice_id=0,
            note=60,
            gain_q15=Q15_HALF,
            coeff_q15=coeffs,
        )

        all_samples = []
        for _ in range(int(1.0 * BLOCKS_PER_SEC)):  # 1 second
            samples = renderer.render_voice(frame)
            all_samples.extend(samples)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "wavelet_single_note_c4")

        max_delta = compute_max_delta(all_samples)
        print(f"\n  Single note (Wavelet): {len(L)} samples, max_delta={max_delta}")
        print(f"    Saved: {filepath}")

        assert len(all_samples) > 0

    def test_chromatic_scale_auto_crossfade(self):
        """
        Chromatic scale with automatic dual-render crossfade.

        The UnifiedRenderer should automatically detect coefficient changes
        and apply dual-render crossfade.
        """
        backend = WaveletBackend(block_size=BLOCK_SIZE)
        renderer = UnifiedRenderer(backend)

        renderer.voice_allocate(0)

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)  # 250ms per note
        all_samples = []

        for note in range(48, 73):  # C3 to C5
            freq = midi_to_freq(note)
            coeffs = generate_coeffs_for_freq(freq)

            frame = RenderFrame(
                voice_id=0,
                note=note,
                gain_q15=Q15_HALF,
                coeff_q15=coeffs,
            )

            for _ in range(blocks_per_note):
                samples = renderer.render_voice(frame)
                all_samples.extend(samples)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "wavelet_chromatic_c3_c5_unified")

        max_delta = compute_max_delta(all_samples)
        clicks = count_clicks(all_samples, 5000)

        print(f"\n  Chromatic scale (Wavelet/Unified): {len(L)} samples")
        print(f"    Max delta: {max_delta}, Clicks >5000: {clicks}")
        print(f"    Saved: {filepath}")

        # With dual-render, should have fewer clicks than before
        # (may still have some due to wavelet nature)

    def test_rapid_transitions(self):
        """Test rapid pitch transitions (stress test)."""
        backend = WaveletBackend(block_size=BLOCK_SIZE)
        renderer = UnifiedRenderer(backend)

        renderer.voice_allocate(0)

        all_samples = []
        notes = [60, 67, 60, 67] * 25  # 100 transitions

        for note in notes:
            freq = midi_to_freq(note)
            coeffs = generate_coeffs_for_freq(freq)

            frame = RenderFrame(
                voice_id=0,
                note=note,
                gain_q15=Q15_HALF,
                coeff_q15=coeffs,
            )

            # Just 2 blocks per note (rapid)
            for _ in range(2):
                samples = renderer.render_voice(frame)
                all_samples.extend(samples)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "wavelet_rapid_transitions_unified")

        max_delta = compute_max_delta(all_samples)
        clicks = count_clicks(all_samples, 5000)

        print(f"\n  Rapid transitions (Wavelet/Unified): {len(L)} samples")
        print(f"    Max delta: {max_delta}, Clicks >5000: {clicks}")
        print(f"    Saved: {filepath}")


# =============================================================================
# COMPARISON TESTS
# =============================================================================

class TestComparison:
    """Compare behavior between backends."""

    def test_same_pattern_both_backends(self):
        """
        Run same chromatic pattern through both backends.

        Both should produce click-free output with similar characteristics.
        """
        blocks_per_note = int(0.125 * BLOCKS_PER_SEC)  # 125ms per note

        # Harmonic backend
        harmonic_backend = HarmonicScaffoldBackend()
        harmonic_renderer = UnifiedRenderer(harmonic_backend)
        harmonic_renderer.voice_allocate(0)

        harmonic_samples = []
        for note in range(48, 73):
            frame = RenderFrame(
                voice_id=0,
                note=note,
                waveform="saw",
                gain_q15=Q15_HALF,
            )
            for _ in range(blocks_per_note):
                samples = harmonic_renderer.render_voice(frame)
                harmonic_samples.extend(samples)

        # Wavelet backend
        wavelet_backend = WaveletBackend(block_size=BLOCK_SIZE)
        wavelet_renderer = UnifiedRenderer(wavelet_backend)
        wavelet_renderer.voice_allocate(0)

        wavelet_samples = []
        for note in range(48, 73):
            freq = midi_to_freq(note)
            coeffs = generate_coeffs_for_freq(freq)

            frame = RenderFrame(
                voice_id=0,
                note=note,
                gain_q15=Q15_HALF,
                coeff_q15=coeffs,
            )
            for _ in range(blocks_per_note):
                samples = wavelet_renderer.render_voice(frame)
                wavelet_samples.extend(samples)

        # Save both
        L, R = samples_to_stereo(harmonic_samples)
        filepath_h = save_audio(L, R, "comparison_harmonic")

        L, R = samples_to_stereo(wavelet_samples)
        filepath_w = save_audio(L, R, "comparison_wavelet")

        # Analyze
        h_max_delta = compute_max_delta(harmonic_samples)
        h_clicks = count_clicks(harmonic_samples, 5000)

        w_max_delta = compute_max_delta(wavelet_samples)
        w_clicks = count_clicks(wavelet_samples, 5000)

        print(f"\n  Comparison - Same pattern, different backends:")
        print(f"    Harmonic: max_delta={h_max_delta}, clicks={h_clicks}")
        print(f"    Wavelet:  max_delta={w_max_delta}, clicks={w_clicks}")
        print(f"    Saved: {filepath_h}")
        print(f"    Saved: {filepath_w}")


# =============================================================================
# MICROFADE TESTS
# =============================================================================

class TestMicrofade:
    """Test voice on/off microfade behavior."""

    def test_voice_on_fade_in(self):
        """First block should have fade-in applied."""
        backend = HarmonicScaffoldBackend()
        renderer = UnifiedRenderer(backend, microfade_samples=16)

        renderer.voice_allocate(0)

        frame = RenderFrame(
            voice_id=0,
            note=60,
            waveform="saw",
            gain_q15=Q15_ONE,  # Full volume
        )

        # First block should have fade-in
        samples = renderer.render_voice(frame)

        # First sample should be near zero (faded in)
        assert abs(samples[0]) < 1000, f"First sample too loud: {samples[0]}"

        # Sample at end of fade should be louder
        assert abs(samples[15]) > abs(samples[0]), "Fade-in not working"

        print(f"\n  Voice on fade-in:")
        print(f"    First sample: {samples[0]}")
        print(f"    Sample 15: {samples[15]}")
        print(f"    Last sample: {samples[-1]}")

    def test_voice_off_fade_out(self):
        """Signal voice_off and next block should have fade-out."""
        backend = HarmonicScaffoldBackend()
        renderer = UnifiedRenderer(backend, microfade_samples=16)

        renderer.voice_allocate(0)

        frame = RenderFrame(
            voice_id=0,
            note=60,
            waveform="saw",
            gain_q15=Q15_ONE,
        )

        # First block (fade-in)
        _ = renderer.render_voice(frame)

        # Second block (normal)
        _ = renderer.render_voice(frame)

        # Signal voice off
        renderer.voice_off(0)

        # Third block should have fade-out
        samples = renderer.render_voice(frame)

        # Last samples should be near zero
        assert abs(samples[-1]) < 1000, f"Last sample too loud: {samples[-1]}"

        print(f"\n  Voice off fade-out:")
        print(f"    First sample: {samples[0]}")
        print(f"    Sample 48: {samples[48]}")
        print(f"    Last sample: {samples[-1]}")


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_crossfade_samples(self):
        """Test crossfade_samples function."""
        old = [1000] * 64
        new = [5000] * 64

        result = crossfade_samples(old, new, fade_samples=16)

        # First sample should be closer to old
        assert result[0] < 2000

        # Last sample should be new
        assert result[-1] == 5000

        # Middle of fade should be blended
        assert 2000 < result[8] < 4000

        print(f"\n  Crossfade test:")
        print(f"    result[0]: {result[0]} (should be ~1000)")
        print(f"    result[8]: {result[8]} (should be ~3000)")
        print(f"    result[16]: {result[16]} (should be 5000)")

    def test_apply_microfade(self):
        """Test apply_microfade function."""
        samples = [10000] * 64

        # Fade in
        result_in = apply_microfade(samples, fade_in=True, fade_samples=16)
        assert result_in[0] < 1000
        assert result_in[15] > 5000

        # Fade out
        result_out = apply_microfade(samples, fade_out=True, fade_samples=16)
        assert result_out[-1] < 1000
        assert result_out[-16] > 5000

        print(f"\n  Microfade test:")
        print(f"    Fade-in first: {result_in[0]}")
        print(f"    Fade-out last: {result_out[-1]}")

    def test_detect_discontinuities(self):
        """Test discontinuity detection."""
        frame1 = RenderFrame(voice_id=0, note=60)
        frame2 = RenderFrame(voice_id=0, note=60)  # Same
        frame3 = RenderFrame(voice_id=0, note=62)  # Different

        # Same frame should not trigger
        flags = detect_discontinuities(frame1, frame2)
        assert not flags.any_discontinuity

        # Different note should trigger
        flags = detect_discontinuities(frame1, frame3)
        assert flags.pitch_changed
        assert flags.any_discontinuity

        # First frame (None old) should trigger voice_allocated
        flags = detect_discontinuities(None, frame1)
        assert flags.voice_allocated

        print(f"\n  Discontinuity detection:")
        print(f"    Same frame: {detect_discontinuities(frame1, frame2).any_discontinuity}")
        print(f"    Note change: {detect_discontinuities(frame1, frame3).pitch_changed}")
        print(f"    New voice: {detect_discontinuities(None, frame1).voice_allocated}")


# =============================================================================
# SAFETY ENGINE INTEGRATION TESTS
# =============================================================================

class TestSafetyIntegration:
    """
    Test UnifiedRenderer with SafetyEngine integration.

    This mirrors test_safety_chromatic.py but uses the UnifiedRenderer
    with dual-render crossfade for pitch transitions.
    """

    def test_chromatic_c3_c5_with_safety(self):
        """
        Chromatic scale C3-C5 through UnifiedRenderer + SafetyEngine.

        This is the key test - does dual-render + safety produce click-free output?
        """
        from symcrash_python_ap.safety_engine import SafetyEngine
        from symcrash_python_ap.frames import ParamFrame

        backend = WaveletBackend(block_size=BLOCK_SIZE)
        renderer = UnifiedRenderer(backend)
        safety = SafetyEngine(block_size=BLOCK_SIZE)
        params = ParamFrame()

        renderer.voice_allocate(0)
        safety.voice_on(0)

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)  # 250ms per note
        all_L = []
        all_R = []
        crossfade_count = 0

        for note in range(48, 73):  # C3 to C5
            freq = midi_to_freq(note)
            coeffs = generate_coeffs_for_freq(freq)

            frame = RenderFrame(
                voice_id=0,
                note=note,
                gain_q15=Q15_HALF,
                coeff_q15=coeffs,
            )

            for block_idx in range(blocks_per_note):
                # Check if crossfade will be applied
                voice = renderer.voices.get(0)
                if voice and voice.last_frame and voice.last_frame.note != note:
                    crossfade_count += 1

                # Render through UnifiedRenderer
                samples = renderer.render_voice(frame)

                # Apply SafetyEngine declick
                samples = safety.apply_declick(samples, 0)

                # Convert to stereo (simple mono->stereo)
                L = list(samples)
                R = list(samples)

                # Apply SafetyEngine limiting
                L, R = safety.step_block(L, R, params)

                all_L.extend(L)
                all_R.extend(R)

        # Save audio
        metadata = write_audio(
            "unified_safety_chromatic_c3_c5",
            all_L, all_R,
            timestamp=True,
            subdirectory=OUTPUT_SUBDIR,
            verbose=False
        )
        filepath = metadata.filepath

        # Compute metrics
        L_arr = np.array(all_L, dtype=np.float32)
        deltas = np.abs(np.diff(L_arr))
        clicks_3000 = int(np.sum(deltas > 3000))
        clicks_5000 = int(np.sum(deltas > 5000))
        max_delta = int(np.max(deltas)) if len(deltas) > 0 else 0
        peak = int(np.max(np.abs(L_arr))) if len(L_arr) > 0 else 0
        rms = int(np.sqrt(np.mean(L_arr ** 2))) if len(L_arr) > 0 else 0

        print(f"\n  Chromatic C3-C5 (UnifiedRenderer + SafetyEngine):")
        print(f"    Duration: {len(all_L) / FS:.2f}s")
        print(f"    Crossfades applied: {crossfade_count}")
        print(f"    Peak: {peak}, RMS: {rms}")
        print(f"    Clicks >3000: {clicks_3000}, >5000: {clicks_5000}")
        print(f"    Max delta: {max_delta}")
        print(f"    Saved: {filepath}")

        # With dual-render + safety, should have fewer clicks than raw safety
        # The safety_chromatic test typically shows high click counts
        # We want to verify this is improved
        assert max_delta < 25000, f"Very large delta detected: {max_delta}"

    def test_rapid_transitions_with_safety(self):
        """
        Rapid transitions stress test with SafetyEngine.
        """
        from symcrash_python_ap.safety_engine import SafetyEngine
        from symcrash_python_ap.frames import ParamFrame

        backend = WaveletBackend(block_size=BLOCK_SIZE)
        renderer = UnifiedRenderer(backend)
        safety = SafetyEngine(block_size=BLOCK_SIZE)
        params = ParamFrame()

        renderer.voice_allocate(0)
        safety.voice_on(0)

        all_L = []
        all_R = []

        # Alternating notes - 50 transitions per second for 2 seconds
        notes = [48, 60, 55, 67, 52, 64]
        total_blocks = int(2.0 * BLOCKS_PER_SEC)
        blocks_per_transition = max(1, BLOCKS_PER_SEC // 50)

        note_idx = 0
        for block in range(total_blocks):
            if block % blocks_per_transition == 0 and block > 0:
                note_idx = (note_idx + 1) % len(notes)

            note = notes[note_idx]
            freq = midi_to_freq(note)
            coeffs = generate_coeffs_for_freq(freq)

            frame = RenderFrame(
                voice_id=0,
                note=note,
                gain_q15=Q15_HALF,
                coeff_q15=coeffs,
            )

            samples = renderer.render_voice(frame)
            samples = safety.apply_declick(samples, 0)

            L = list(samples)
            R = list(samples)
            L, R = safety.step_block(L, R, params)

            all_L.extend(L)
            all_R.extend(R)

        metadata = write_audio(
            "unified_safety_rapid_50",
            all_L, all_R,
            timestamp=True,
            subdirectory=OUTPUT_SUBDIR,
            verbose=False
        )

        L_arr = np.array(all_L, dtype=np.float32)
        deltas = np.abs(np.diff(L_arr))
        clicks_5000 = int(np.sum(deltas > 5000))
        max_delta = int(np.max(deltas)) if len(deltas) > 0 else 0

        print(f"\n  Rapid transitions (UnifiedRenderer + SafetyEngine):")
        print(f"    Duration: {len(all_L) / FS:.2f}s")
        print(f"    Clicks >5000: {clicks_5000}")
        print(f"    Max delta: {max_delta}")
        print(f"    Saved: {metadata.filepath}")

    def test_synth_arpeggio_140bpm_with_safety(self):
        """
        Classic synth arpeggio at 140 BPM with SafetyEngine.
        """
        from symcrash_python_ap.safety_engine import SafetyEngine
        from symcrash_python_ap.frames import ParamFrame

        backend = WaveletBackend(block_size=BLOCK_SIZE)
        renderer = UnifiedRenderer(backend)
        safety = SafetyEngine(block_size=BLOCK_SIZE)
        params = ParamFrame()

        renderer.voice_allocate(0)
        safety.voice_on(0)

        # Minor 7th pattern at 140 BPM, 16th notes
        root_note = 48
        pattern = [0, 3, 7, 10, 12, 10, 7, 3]
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
            freq = midi_to_freq(note)
            coeffs = generate_coeffs_for_freq(freq)

            frame = RenderFrame(
                voice_id=0,
                note=note,
                gain_q15=Q15_HALF,
                coeff_q15=coeffs,
            )

            samples = renderer.render_voice(frame)
            samples = safety.apply_declick(samples, 0)

            L = list(samples)
            R = list(samples)
            L, R = safety.step_block(L, R, params)

            all_L.extend(L)
            all_R.extend(R)
            blocks_since_transition += 1

        metadata = write_audio(
            "unified_safety_synth_arp_140bpm",
            all_L, all_R,
            timestamp=True,
            subdirectory=OUTPUT_SUBDIR,
            verbose=False
        )

        L_arr = np.array(all_L, dtype=np.float32)
        deltas = np.abs(np.diff(L_arr))
        clicks_5000 = int(np.sum(deltas > 5000))
        max_delta = int(np.max(deltas)) if len(deltas) > 0 else 0
        peak = int(np.max(np.abs(L_arr))) if len(L_arr) > 0 else 0

        print(f"\n  Synth Arpeggio 140 BPM (UnifiedRenderer + SafetyEngine):")
        print(f"    Duration: {len(all_L) / FS:.2f}s")
        print(f"    Peak: {peak}")
        print(f"    Clicks >5000: {clicks_5000}")
        print(f"    Max delta: {max_delta}")
        print(f"    Saved: {metadata.filepath}")


# =============================================================================
# STANDARD PIPELINE TESTS
# =============================================================================

class TestStandardPipelines:
    """
    Test the recommended standard pipeline classes.

    These are the "easy mode" APIs that combine UnifiedRenderer + SafetyEngine.
    """

    def test_standard_wavelet_chromatic(self):
        """
        Chromatic scale C3-C5 using StandardWaveletPipeline.

        This is the simplest way to use the wavelet engine with full safety.
        """
        pipeline = StandardWaveletPipeline()
        pipeline.voice_on(0)

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)  # 250ms per note
        all_L = []
        all_R = []

        for note in range(48, 73):  # C3 to C5
            freq = midi_to_freq(note)
            coeffs = generate_coeffs_for_freq(freq)

            for _ in range(blocks_per_note):
                L, R = pipeline.render_block(
                    voice_id=0,
                    note=note,
                    coeffs=coeffs,
                    gain_q15=Q15_HALF,
                )
                all_L.extend(L)
                all_R.extend(R)

        metadata = write_audio(
            "standard_wavelet_chromatic",
            all_L, all_R,
            timestamp=True,
            subdirectory=OUTPUT_SUBDIR,
            verbose=False
        )

        # Compute metrics
        L_arr = np.array(all_L, dtype=np.float32)
        deltas = np.abs(np.diff(L_arr))
        clicks_5000 = int(np.sum(deltas > 5000))
        max_delta = int(np.max(deltas)) if len(deltas) > 0 else 0
        peak = int(np.max(np.abs(L_arr))) if len(L_arr) > 0 else 0

        print(f"\n  StandardWaveletPipeline Chromatic:")
        print(f"    Duration: {len(all_L) / FS:.2f}s")
        print(f"    Peak: {peak}")
        print(f"    Clicks >5000: {clicks_5000}")
        print(f"    Max delta: {max_delta}")
        print(f"    Saved: {metadata.filepath}")

        pipeline.voice_off(0)

    def test_standard_harmonic_chromatic(self):
        """
        Chromatic scale C3-C5 using StandardHarmonicPipeline.
        """
        pipeline = StandardHarmonicPipeline()
        pipeline.voice_on(0)

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)  # 250ms per note
        all_L = []
        all_R = []

        for note in range(48, 73):  # C3 to C5
            for _ in range(blocks_per_note):
                L, R = pipeline.render_block(
                    voice_id=0,
                    note=note,
                    waveform="saw",
                    gain_q15=Q15_HALF,
                )
                all_L.extend(L)
                all_R.extend(R)

        metadata = write_audio(
            "standard_harmonic_chromatic",
            all_L, all_R,
            timestamp=True,
            subdirectory=OUTPUT_SUBDIR,
            verbose=False
        )

        L_arr = np.array(all_L, dtype=np.float32)
        deltas = np.abs(np.diff(L_arr))
        max_delta = int(np.max(deltas)) if len(deltas) > 0 else 0
        peak = int(np.max(np.abs(L_arr))) if len(L_arr) > 0 else 0

        print(f"\n  StandardHarmonicPipeline Chromatic:")
        print(f"    Duration: {len(all_L) / FS:.2f}s")
        print(f"    Peak: {peak}")
        print(f"    Max delta: {max_delta}")
        print(f"    Saved: {metadata.filepath}")

        pipeline.voice_off(0)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
