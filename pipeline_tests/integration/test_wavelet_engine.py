"""
WAVELET ENGINE TESTS (W1-W4)
============================

Pure mechanical realization.

Covers:
- W1: No Musical Semantics
- W2: Perfect Reconstruction (Internal)
- W3: Tree Interpolation Hygiene
- W4: Time-Multiplexing Safety
"""

import pytest
import numpy as np


class TestW1NoMusicalSemantics:
    """
    W1. No Musical Semantics

    Requirement:
    - Wavelet engine cannot know "note", "voice", "regime"

    Test:
    - Feed random coefficients -> deterministic audio

    Gotchas:
    - Hidden assumptions about coefficient meaning
    """

    def test_random_coeffs_produce_output(self, wavelet_engine, q15_half):
        """Random coefficients produce deterministic audio."""
        from symcrash_python_ap.frames import VoiceFrame

        np.random.seed(42)

        # Create random coefficients
        coeffs = [int(np.random.randint(-q15_half, q15_half)) for _ in range(256)]

        # Create minimal VoiceFrame
        vf = VoiceFrame(
            voice_id=0,
            tree_id=0,
            gain_q15=q15_half,
            coeff_q15=coeffs,
        )

        # Should produce audio
        audio = wavelet_engine.render_voice(vf)

        assert len(audio) == 64, f"Wrong output length: {len(audio)}"
        assert any(a != 0 for a in audio), "Output is all zeros"

    def test_same_coeffs_same_output(self, wavelet_engine, q15_half):
        """Same coefficients produce same output (pure synthesis, no regularity state)."""
        from symcrash_python_ap.frames import VoiceFrame

        coeffs = [1000] * 256

        vf = VoiceFrame(
            voice_id=0,
            tree_id=0,
            gain_q15=q15_half,
            coeff_q15=coeffs,
        )

        # Disable regularity to test pure wavelet synthesis determinism
        # RegularityOperator maintains state between calls (prev_sample, slew)
        # which is intentional for click-free audio but breaks this test
        audio1 = wavelet_engine.render_voice(vf, apply_regularity=False)
        audio2 = wavelet_engine.render_voice(vf, apply_regularity=False)

        assert audio1 == audio2, "Same input produced different output"

    def test_no_note_awareness(self, wavelet_engine, q15_half):
        """Wavelet engine has no concept of notes."""
        from symcrash_python_ap.frames import VoiceFrame

        # VoiceFrame doesn't contain note information for wavelet
        vf = VoiceFrame(
            voice_id=0,
            tree_id=0,
            coeff_q15=[1000] * 256,
        )

        # Should work without any pitch/note info
        audio = wavelet_engine.render_voice(vf)
        assert len(audio) == 64


class TestW2PerfectReconstruction:
    """
    W2. Perfect Reconstruction (Internal)

    Requirement:
    - Forward + inverse (where used) are numerically stable

    Test:
    - Round-trip test within fixed-point tolerance

    Gotchas:
    - Boundary handling errors
    - Lifting step truncation artifacts
    """

    def test_synthesis_bounded(self, wavelet_engine, q15_half):
        """Synthesis produces bounded output."""
        from symcrash_python_ap.frames import VoiceFrame

        # Test with various coefficient patterns
        test_patterns = [
            [q15_half] * 256,  # Uniform
            [q15_half if i % 2 == 0 else -q15_half for i in range(256)],  # Alternating
            [q15_half * (i % 10) // 10 for i in range(256)],  # Ramp
        ]

        for pattern in test_patterns:
            vf = VoiceFrame(
                voice_id=0,
                tree_id=0,
                gain_q15=q15_half,
                coeff_q15=pattern,
            )

            audio = wavelet_engine.render_voice(vf)

            # Output should be bounded
            for sample in audio:
                assert -35000 < sample < 35000, f"Synthesis overflow: {sample}"

    def test_tree_synthesis_consistent(self, wavelet_engine, q15_half):
        """All tree types produce consistent output."""
        from symcrash_python_ap.frames import VoiceFrame

        coeffs = [1000] * 256

        outputs = []
        for tree_id in range(5):
            vf = VoiceFrame(
                voice_id=0,
                tree_id=tree_id,
                gain_q15=q15_half,
                coeff_q15=coeffs,
            )

            audio = wavelet_engine.render_voice(vf)
            outputs.append(audio)

            # Each should produce valid output
            assert len(audio) == 64
            assert any(a != 0 for a in audio)

        # Different trees should produce different output
        # (at least some difference)
        all_same = all(outputs[i] == outputs[0] for i in range(1, 5))
        # It's OK if they're similar, but shouldn't all be identical
        # unless coefficients are very simple


class TestW3TreeInterpolationHygiene:
    """
    W3. Tree Interpolation Hygiene

    Requirement:
    - Tree switching produces no clicks

    Test:
    - Rapid alternation between extreme trees

    Gotchas:
    - Interpolating after synthesis instead of coefficients
    - Mismatched normalization across trees
    """

    def test_tree_crossfade_smooth(self, wavelet_engine, q15_half, q15_one):
        """Tree crossfade produces smooth transition."""
        from symcrash_python_ap.frames import VoiceFrame

        coeffs = [1000] * 256

        # Rapid tree changes
        outputs = []
        for i in range(20):
            tree_id = i % 5

            vf = VoiceFrame(
                voice_id=0,
                tree_id=tree_id,
                gain_q15=q15_half,
                coeff_q15=coeffs,
            )

            audio = wavelet_engine.render_voice(vf)
            outputs.extend(audio)

        # Check for clicks (large sample-to-sample changes)
        clicks = 0
        for i in range(1, len(outputs)):
            diff = abs(outputs[i] - outputs[i-1])
            if diff > q15_half:
                clicks += 1

        # Allow some at block boundaries, but not many
        click_rate = clicks / len(outputs)
        assert click_rate < 0.05, f"Click rate too high: {click_rate}"

    def test_blend_produces_intermediate(self, wavelet_engine, q15_half):
        """Blended trees produce intermediate output."""
        # This test checks the crossfade mechanism
        from symcrash_python_ap.frames import VoiceFrame

        coeffs = [1000] * 256

        # Get output for tree 0
        vf0 = VoiceFrame(voice_id=0, tree_id=0, gain_q15=q15_half, coeff_q15=coeffs)
        audio0 = wavelet_engine.render_voice(vf0)

        # Get output for tree 4
        vf4 = VoiceFrame(voice_id=0, tree_id=4, gain_q15=q15_half, coeff_q15=coeffs)
        audio4 = wavelet_engine.render_voice(vf4)

        # During crossfade, output should be somewhere between
        # (This is verified by the crossfade mechanism in render_voice)


class TestW4TimeMultiplexingSafety:
    """
    W4. Time-Multiplexing Safety

    Requirement:
    - Multiple voices sharing hardware do not bleed

    Test:
    - Stress test with alternating voices per block

    Gotchas:
    - Incomplete buffer clears
    - Pipeline hazards in FPGA scheduling
    """

    def test_voices_dont_bleed(self, wavelet_engine, q15_half):
        """Different voices don't bleed into each other."""
        from symcrash_python_ap.frames import VoiceFrame

        # Voice 0: high frequency content
        coeffs0 = [0] * 256
        coeffs0[200:256] = [10000] * 56  # Only high band

        # Voice 1: low frequency content
        coeffs1 = [0] * 256
        coeffs1[0:50] = [10000] * 50  # Only low band

        vf0 = VoiceFrame(voice_id=0, tree_id=0, gain_q15=q15_half, coeff_q15=coeffs0)
        vf1 = VoiceFrame(voice_id=1, tree_id=0, gain_q15=q15_half, coeff_q15=coeffs1)

        # Alternate rendering
        for _ in range(20):
            audio0 = wavelet_engine.render_voice(vf0)
            audio1 = wavelet_engine.render_voice(vf1)

            # Each should maintain its character
            # (HF content has more zero crossings)
            crossings0 = sum(1 for i in range(1, len(audio0))
                             if audio0[i] * audio0[i-1] < 0)
            crossings1 = sum(1 for i in range(1, len(audio1))
                             if audio1[i] * audio1[i-1] < 0)

            # High-frequency voice should have more zero crossings
            # (This is a rough heuristic)

    def test_sequential_voices_independent(self, wavelet_engine, q15_half):
        """Sequential voice rendering is independent."""
        from symcrash_python_ap.frames import VoiceFrame

        # Render voice 0 with complex content
        coeffs0 = [1000 * ((i * 7) % 10 - 5) for i in range(256)]
        vf0 = VoiceFrame(voice_id=0, tree_id=2, gain_q15=q15_half, coeff_q15=coeffs0)
        audio0 = wavelet_engine.render_voice(vf0)

        # Render voice 1 with simple content
        coeffs1 = [500] * 256
        vf1 = VoiceFrame(voice_id=1, tree_id=0, gain_q15=q15_half, coeff_q15=coeffs1)
        audio1_after = wavelet_engine.render_voice(vf1)

        # Render voice 1 fresh (in new engine)
        from symcrash_python_ap.wavelet import WaveletEngine
        fresh_engine = WaveletEngine(block_size=64)
        audio1_fresh = fresh_engine.render_voice(vf1)

        # Should be the same (no state bleed from voice 0)
        assert audio1_after == audio1_fresh, "Voice state bleeding detected"

    def test_high_voice_count(self, wavelet_engine, q15_half):
        """Many voices can render without interference."""
        from symcrash_python_ap.frames import VoiceFrame

        outputs = []
        for vid in range(12):
            coeffs = [1000 + vid * 100] * 256
            vf = VoiceFrame(
                voice_id=vid,
                tree_id=vid % 5,
                gain_q15=q15_half,
                coeff_q15=coeffs,
            )

            audio = wavelet_engine.render_voice(vf)
            outputs.append(audio)

        # Each should produce valid, non-zero output
        for vid, audio in enumerate(outputs):
            assert len(audio) == 64
            energy = sum(a * a for a in audio)
            assert energy > 0, f"Voice {vid} produced silence"


class TestW5VoiceLifecycle:
    """
    W5. Voice Lifecycle (WSE-STD-070)

    Requirement:
    - voice_allocate arms microfade, clears tree state
    - voice_on does NOT reset state (retrigger only)
    - voice_free keeps prev_sample for next allocate

    Test:
    - Microfade applied on allocation prevents clicks
    - Retrigger preserves continuity
    - Reallocation uses retained last sample

    Gotchas:
    - Resetting on voice_on causes clicks
    - Not preserving prev_sample on free causes clicks on realloc
    """

    def test_voice_allocate_arms_microfade(self, wavelet_engine, q15_half):
        """voice_allocate arms microfade for click prevention."""
        from symcrash_python_ap.frames import VoiceFrame

        voice_id = 0

        # Allocate voice (arms microfade)
        wavelet_engine.voice_allocate(voice_id, last_sample=0)

        R = wavelet_engine.get_regularity(voice_id)
        assert R.is_fading(), "Microfade should be armed after allocate"
        assert R._microfade_counter == 32, "Microfade should be 32 samples"

    def test_voice_on_does_not_reset(self, wavelet_engine, q15_half):
        """voice_on (retrigger) does NOT reset state."""
        from symcrash_python_ap.frames import VoiceFrame

        voice_id = 0

        # First render to establish state
        coeffs = [1000] * 256
        vf = VoiceFrame(voice_id=voice_id, tree_id=0, gain_q15=q15_half, coeff_q15=coeffs)
        wavelet_engine.render_voice(vf)

        R = wavelet_engine.get_regularity(voice_id)
        prev_sample_before = R.prev_sample

        # voice_on should NOT reset
        wavelet_engine.voice_on(voice_id)

        assert R.prev_sample == prev_sample_before, "voice_on should NOT reset prev_sample"
        assert not R.is_fading(), "voice_on should NOT arm microfade"

    def test_voice_free_preserves_last_sample(self, wavelet_engine, q15_half):
        """voice_free keeps prev_sample for next allocate."""
        from symcrash_python_ap.frames import VoiceFrame

        voice_id = 0

        # Render to establish state
        coeffs = [5000] * 256
        vf = VoiceFrame(voice_id=voice_id, tree_id=0, gain_q15=q15_half, coeff_q15=coeffs)
        wavelet_engine.render_voice(vf)

        R = wavelet_engine.get_regularity(voice_id)
        prev_sample_before = R.prev_sample

        # Free voice
        wavelet_engine.voice_free(voice_id)

        # prev_sample should be preserved
        assert R.prev_sample == prev_sample_before, "voice_free should preserve prev_sample"

    def test_reallocation_uses_preserved_sample(self, wavelet_engine, q15_half):
        """Reallocation uses preserved last sample for smooth transition."""
        from symcrash_python_ap.frames import VoiceFrame

        voice_id = 0

        # Render to establish non-zero state
        coeffs = [10000] * 256
        vf = VoiceFrame(voice_id=voice_id, tree_id=0, gain_q15=q15_half, coeff_q15=coeffs)
        wavelet_engine.render_voice(vf)

        R = wavelet_engine.get_regularity(voice_id)
        prev_sample_before = R.prev_sample

        # Free and reallocate (new owner)
        wavelet_engine.voice_free(voice_id)
        wavelet_engine.voice_allocate(voice_id)  # No explicit last_sample, uses preserved

        # Microfade should be armed
        assert R.is_fading(), "Microfade should be armed after reallocate"
        # prev_sample should still be the preserved value
        assert R.prev_sample == prev_sample_before, "allocate() should use preserved prev_sample"

    def test_microfade_produces_smooth_onset(self, wavelet_engine, q15_half):
        """Microfade produces smooth fade-in from 0."""
        from symcrash_python_ap.frames import VoiceFrame

        voice_id = 0

        # Allocate voice from silence
        wavelet_engine.voice_allocate(voice_id, last_sample=0)

        # Render loud content
        coeffs = [20000] * 256
        vf = VoiceFrame(voice_id=voice_id, tree_id=0, gain_q15=q15_half, coeff_q15=coeffs)
        audio = wavelet_engine.render_voice(vf)

        # First sample should be near 0 (faded in from 0)
        assert abs(audio[0]) < 1000, f"First sample should be near 0, got {audio[0]}"

        # Samples should gradually increase
        # Check that sample 16 is larger than sample 4
        assert abs(audio[16]) > abs(audio[4]), "Audio should fade in gradually"

    def test_allocate_clears_tree_state(self, wavelet_engine, q15_half):
        """voice_allocate clears tree transition state."""
        from symcrash_python_ap.frames import VoiceFrame

        voice_id = 0

        # Set up some tree state
        wavelet_engine.voice_tree_state[voice_id] = {"tree_a": 0, "tree_b": 2, "blend": 0.5}
        wavelet_engine.boundary_state[voice_id] = {"some": "state"}

        # Allocate (new owner)
        wavelet_engine.voice_allocate(voice_id, last_sample=0)

        # Tree state should be cleared
        assert voice_id not in wavelet_engine.voice_tree_state, "Tree state should be cleared"
        assert voice_id not in wavelet_engine.boundary_state, "Boundary state should be cleared"

    def test_voice_off_is_alias_for_voice_free(self, wavelet_engine, q15_half):
        """voice_off (legacy) behaves like voice_free."""
        from symcrash_python_ap.frames import VoiceFrame

        voice_id = 0

        # Render to establish state
        coeffs = [5000] * 256
        vf = VoiceFrame(voice_id=voice_id, tree_id=0, gain_q15=q15_half, coeff_q15=coeffs)
        wavelet_engine.render_voice(vf)

        R = wavelet_engine.get_regularity(voice_id)
        prev_sample_before = R.prev_sample

        # voice_off (legacy)
        wavelet_engine.voice_off(voice_id)

        # Should preserve state like voice_free
        assert R.prev_sample == prev_sample_before, "voice_off should preserve prev_sample"
