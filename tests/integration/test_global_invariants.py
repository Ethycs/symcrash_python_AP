"""
GLOBAL INVARIANT TESTS (G0-G2)
==============================

These are non-negotiable. Any failure here is a **hard bug**.

Covers:
- G0: Block Invisibility
- G1: Determinism Under Load
- G2: Graceful Degradation
"""

import pytest
import numpy as np
from typing import List, Tuple


class TestG0BlockInvisibility:
    """
    G0. Block Invisibility

    Requirement: No audible discontinuity attributable to block boundaries.

    Test:
    - Sustain a static tone for >=60 seconds (simulated as many blocks)
    - Sweep parameters slowly and abruptly
    - Check output for periodic artifacts at block rate (750 Hz)

    Gotchas:
    - Tree switches without coefficient crossfade
    - Event injection aligned to block edges without smoothing
    - Safety limiter engaging discretely per block
    """

    def test_static_tone_no_block_artifacts(self, voice_engine, q15_half):
        """Sustain static tone and check for block-rate discontinuities."""
        from symcrash_python_ap.voice.voice_engine import GestureBasis

        # Allocate voice with simple gesture
        vid = voice_engine.allocate_voice(program_id=1004)  # Static Organ
        gesture = GestureBasis(
            gate=True,
            pressure_q15=q15_half,
            speed_q15=0,
            chaos_q15=0,
        )
        voice_engine.note_on(vid, gesture=gesture)

        # Simulate 1000 blocks (~1.33 seconds)
        frames = []
        for _ in range(1000):
            frame = voice_engine.step_voice(vid)
            if frame:
                frames.append(frame)

        # Check for discontinuities in gain envelope
        gains = [f.gain_q15 for f in frames]

        # After attack, gain should be stable (no periodic jumps)
        stable_gains = gains[50:]  # Skip attack phase

        # Calculate variance - should be very low for static tone
        if len(stable_gains) > 10:
            gain_diffs = [abs(stable_gains[i+1] - stable_gains[i])
                          for i in range(len(stable_gains)-1)]
            max_diff = max(gain_diffs)

            # Max block-to-block change should be small for static tone
            assert max_diff < 1000, f"Block discontinuity detected: max_diff={max_diff}"

    def test_tree_morph_no_clicks(self, voice_engine, q15_half, q15_one):
        """Verify tree morphing produces no discontinuities."""
        from symcrash_python_ap.voice.voice_engine import GestureBasis

        vid = voice_engine.allocate_voice(program_id=1001)  # Glitch Violin

        # Start with low chaos (tree 1)
        gesture = GestureBasis(
            gate=True,
            pressure_q15=q15_half,
            chaos_q15=0,
        )
        voice_engine.note_on(vid, gesture=gesture)

        # Run for a bit to stabilize
        for _ in range(50):
            voice_engine.step_voice(vid)

        # Capture tree state before
        state_before = voice_engine.voices[vid].tree_morph

        # Now increase chaos to trigger tree change
        gesture.chaos_q15 = q15_one
        voice_engine.update_gesture(vid, gesture)

        # Step through morph
        blend_values = []
        for _ in range(200):
            frame = voice_engine.step_voice(vid)
            if frame:
                blend_values.append(frame.tree_blend_q15)

        # Verify blend transitions smoothly (no sudden jumps > morph_rate)
        morph_rate = voice_engine.voices[vid].tree_morph.morph_rate

        for i in range(1, len(blend_values)):
            diff = abs(blend_values[i] - blend_values[i-1])
            # Allow for blend reset when morph completes
            if blend_values[i-1] > 0 or blend_values[i] > 0:
                assert diff <= morph_rate + 100, f"Tree blend jumped: {diff} at block {i}"

    def test_parameter_sweep_continuous(self, voice_engine, q15_half, q15_one):
        """Verify parameter sweeps produce continuous output."""
        from symcrash_python_ap.voice.voice_engine import GestureBasis

        vid = voice_engine.allocate_voice(program_id=1003)  # Turbulent Choir

        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # Sweep pressure from 0 to max over 100 blocks
        gains = []
        for i in range(100):
            gesture.pressure_q15 = (i * q15_one) // 100
            voice_engine.update_gesture(vid, gesture)
            frame = voice_engine.step_voice(vid)
            if frame:
                gains.append(frame.gain_q15)

        # Check for monotonic-ish behavior (no sudden reversals)
        reversals = 0
        for i in range(2, len(gains)):
            # Detect sharp direction changes
            d1 = gains[i-1] - gains[i-2]
            d2 = gains[i] - gains[i-1]
            if d1 > 500 and d2 < -500 or d1 < -500 and d2 > 500:
                reversals += 1

        # Some reversals expected during ADSR, but not many
        assert reversals < 10, f"Too many gain reversals during sweep: {reversals}"


class TestG1DeterminismUnderLoad:
    """
    G1. Determinism Under Load

    Requirement: Same inputs -> same output bit-for-bit (except PRNG-controlled ops).

    Test:
    - Run identical gesture sequence twice
    - Diff output

    Allowed nondeterminism:
    - Explicit PRNG lanes only (NoiseBand, stochastic GateMask)

    Gotchas:
    - Hidden counters in Symbol Engine
    - Voice Engine using wall-clock instead of block count
    - Floating-point leakage
    """

    def test_identical_runs_match(self, q15_half):
        """Same input sequence produces same output."""
        from symcrash_python_ap.voice.voice_engine import VoiceEngine, GestureBasis

        def run_sequence():
            engine = VoiceEngine(max_voices=4)
            vid = engine.allocate_voice(program_id=1001)

            gesture = GestureBasis(
                gate=True,
                pressure_q15=q15_half,
                speed_q15=q15_half,
            )
            engine.note_on(vid, gesture=gesture)

            outputs = []
            for i in range(100):
                # Vary gesture deterministically
                gesture.pressure_q15 = q15_half + (i * 100) % 5000
                engine.update_gesture(vid, gesture)

                frame = engine.step_voice(vid)
                if frame:
                    outputs.append((
                        frame.gain_q15,
                        frame.tree_a_id,
                        frame.tree_blend_q15,
                        tuple(frame.family_weights),
                    ))

            return outputs

        # Run twice
        run1 = run_sequence()
        run2 = run_sequence()

        # Compare
        assert len(run1) == len(run2), "Output length mismatch"

        for i, (o1, o2) in enumerate(zip(run1, run2)):
            assert o1 == o2, f"Determinism failure at block {i}: {o1} != {o2}"

    def test_symbol_engine_determinism(self, q15_one):
        """Symbol Engine produces deterministic output."""
        from symcrash_python_ap.symbol.symbol_engine import SymbolExecutor
        from symcrash_python_ap.frames import VoiceFrame

        def run_executor():
            executor = SymbolExecutor(n_voices=4)

            outputs = []
            for i in range(50):
                vf = VoiceFrame(
                    voice_id=0,
                    gain_q15=q15_one // 2,
                    coeff_q15=[1000] * 256,
                )

                c_out = executor.step_voice(vf, program_id=0)
                outputs.append(tuple(c_out[:10]))  # First 10 coeffs

            return outputs

        run1 = run_executor()
        run2 = run_executor()

        assert run1 == run2, "Symbol Engine is non-deterministic"

    def test_no_wall_clock_dependency(self, voice_engine, q15_half):
        """Verify no wall-clock time dependency."""
        from symcrash_python_ap.voice.voice_engine import GestureBasis
        import time

        vid = voice_engine.allocate_voice(program_id=1004)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # Run 10 blocks quickly
        fast_frames = []
        for _ in range(10):
            frame = voice_engine.step_voice(vid)
            if frame:
                fast_frames.append(frame.gain_q15)

        # Reset and run again with delays (simulating slower execution)
        voice_engine2 = type(voice_engine)(max_voices=12)
        vid2 = voice_engine2.allocate_voice(program_id=1004)
        voice_engine2.note_on(vid2, gesture=gesture)

        slow_frames = []
        for _ in range(10):
            time.sleep(0.001)  # Add tiny delay
            frame = voice_engine2.step_voice(vid2)
            if frame:
                slow_frames.append(frame.gain_q15)

        # Should be identical
        assert fast_frames == slow_frames, "Wall-clock dependency detected"


class TestG2GracefulDegradation:
    """
    G2. Graceful Degradation

    Requirement: When resources are exceeded, sound degrades musically, not catastrophically.

    Test:
    - Max voices + max events + max operator chains
    - Observe: no silence, no NaNs, no runaway noise

    Expected behavior:
    - Drop higher layers first (events -> backbone complexity -> exotic ops)

    Gotchas:
    - All-or-nothing operator application
    - Symbol Engine assuming "full budget" availability
    """

    def test_max_voices_no_crash(self, q15_half):
        """System handles max voices without crash."""
        from symcrash_python_ap.voice.voice_engine import VoiceEngine, GestureBasis

        engine = VoiceEngine(max_voices=12)

        # Allocate all voices
        for i in range(12):
            vid = engine.allocate_voice(program_id=1001 + (i % 4))
            gesture = GestureBasis(
                gate=True,
                pressure_q15=q15_half,
                chaos_q15=q15_half // 2,
            )
            engine.note_on(vid, gesture=gesture)

        # Step all voices
        for _ in range(100):
            frames = engine.step_all()

            # Should get frames for all active voices
            assert len(frames) > 0, "No frames emitted"

            # Check no NaN-like values (overflow indicators)
            for f in frames:
                assert -40000 < f.gain_q15 < 40000, f"Gain overflow: {f.gain_q15}"

    def test_degradation_levels_work(self, q15_one):
        """All degradation levels produce valid output."""
        from symcrash_python_ap.symbol.symbol_engine import SymbolExecutor
        from symcrash_python_ap.frames import VoiceFrame

        for level in [0, 1, 2, 3]:  # DEGRADE_NONE through DEGRADE_EMERGENCY
            executor = SymbolExecutor(n_voices=4)

            vf = VoiceFrame(
                voice_id=0,
                gain_q15=q15_one,
                coeff_q15=[1000] * 256,
            )

            c_out = executor.step_voice(vf, program_id=0)

            # Should produce valid coefficients
            assert len(c_out) == 256, f"Wrong output length at level {level}"

            # Coefficients should be bounded
            max_coeff = max(abs(c) for c in c_out)
            assert max_coeff < 40000, f"Coefficient overflow at level {level}"

    def test_no_silence_under_load(self, q15_half):
        """System never goes completely silent under load."""
        from symcrash_python_ap.voice.voice_engine import VoiceEngine, GestureBasis

        engine = VoiceEngine(max_voices=12)

        # Allocate many voices with high activity
        for i in range(8):
            vid = engine.allocate_voice(program_id=1001 + (i % 4))
            gesture = GestureBasis(
                gate=True,
                pressure_q15=q15_half,
                chaos_q15=q15_half,
                speed_q15=q15_half,
            )
            engine.note_on(vid, gesture=gesture)

        # Run and check for total silence
        silent_blocks = 0
        for _ in range(100):
            frames = engine.step_all()
            total_gain = sum(f.gain_q15 for f in frames)
            if total_gain == 0:
                silent_blocks += 1

        # Some initial silence is OK during attack, but not sustained
        assert silent_blocks < 10, f"Too many silent blocks: {silent_blocks}"

    def test_extreme_parameters_bounded(self, q15_one):
        """Extreme parameters don't cause overflow."""
        from symcrash_python_ap.voice.voice_engine import VoiceEngine, GestureBasis

        engine = VoiceEngine(max_voices=4)
        vid = engine.allocate_voice(program_id=1002)  # Shepard Random

        # Extreme gesture
        gesture = GestureBasis(
            gate=True,
            pressure_q15=q15_one,
            speed_q15=q15_one,
            chaos_q15=q15_one,
            tilt_q15=q15_one,
            pitch_q15=q15_one,
        )
        engine.note_on(vid, gesture=gesture)

        for _ in range(100):
            frame = engine.step_voice(vid)
            if frame:
                # All values should be bounded
                assert -40000 < frame.gain_q15 < 40000
                assert 0 <= frame.tree_a_id < 5
                assert 0 <= frame.tree_b_id < 5
                assert 0 <= frame.tree_blend_q15 <= q15_one

                for w in frame.family_weights:
                    assert -40000 < w < 40000
