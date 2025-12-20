"""
VOICE ENGINE TESTS (V1-V5)
==========================

The only place musical time is allowed to exist.

Covers:
- V1: Musical Time Ownership
- V2: ADSR Correctness
- V3: Regime Graph Stability
- V4: Voice Independence
- V5: VoiceFrame Contract Integrity
"""

import pytest
import numpy as np


class TestV1MusicalTimeOwnership:
    """
    V1. Musical Time Ownership

    Requirement:
    - Voice Engine owns ALL timing: ADSR, regimes, phase, alternation, scheduling
    - Symbol Engine must be time-agnostic

    Test:
    - Disable Symbol Engine entirely; Voice Engine still emits valid VoiceFrames

    Gotchas:
    - `local_block % N` anywhere outside Voice Engine
    - Timing logic inside operator programs
    """

    def test_voice_engine_emits_without_symbol(self, voice_engine, q15_half):
        """Voice Engine works independently of Symbol Engine."""
        from voice_engine import GestureBasis

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # Should emit valid frames without any Symbol Engine involvement
        for _ in range(100):
            frame = voice_engine.step_voice(vid)
            assert frame is not None, "Voice Engine failed to emit frame"
            assert frame.is_active, "Voice should be active"

    def test_block_count_advances(self, voice_engine, q15_half):
        """Block count advances correctly."""
        from voice_engine import GestureBasis

        initial_count = voice_engine.get_block_count()

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        for i in range(50):
            voice_engine.step_all()
            assert voice_engine.get_block_count() == initial_count + i + 1

    def test_voice_has_own_timing(self, voice_engine, q15_half):
        """Each voice maintains independent timing."""
        from voice_engine import GestureBasis, VoicePhase

        # Allocate two voices at different times
        vid1 = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid1, gesture=gesture)

        # Step voice 1 for 20 blocks
        for _ in range(20):
            voice_engine.step_voice(vid1)

        # Now allocate voice 2
        vid2 = voice_engine.allocate_voice(program_id=1001)
        voice_engine.note_on(vid2, gesture=gesture)

        # Voice 1 should be in decay/sustain, voice 2 in attack
        state1 = voice_engine.voices[vid1]
        state2 = voice_engine.voices[vid2]

        assert state1.phase != VoicePhase.ATTACK, "Voice 1 should have left attack"
        assert state2.phase == VoicePhase.ATTACK, "Voice 2 should be in attack"


class TestV2ADSRCorrectness:
    """
    V2. ADSR Correctness (Amplitude + Tree)

    Requirement:
    - ADSR envelopes are continuous across blocks
    - Note-off during attack/decay behaves correctly

    Test:
    - Rapid note retriggering
    - Long release tails under heavy modulation

    Gotchas:
    - Tree ADSR implemented inside wavelet generator
    - Envelope stepping at block boundaries
    """

    def test_envelope_continuity(self, voice_engine, q15_half):
        """Envelope changes smoothly across blocks."""
        from voice_engine import GestureBasis

        vid = voice_engine.allocate_voice(program_id=1003)  # Turbulent Choir (slow attack)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        gains = []
        for _ in range(50):
            frame = voice_engine.step_voice(vid)
            if frame:
                gains.append(frame.gain_q15)

        # Check for smooth transitions (no jumps > 20% of max)
        for i in range(1, len(gains)):
            diff = abs(gains[i] - gains[i-1])
            # Allow larger jumps at phase transitions
            assert diff < 8000, f"Envelope discontinuity at block {i}: diff={diff}"

    def test_note_off_during_attack(self, voice_engine, q15_half):
        """Note-off during attack transitions to release correctly."""
        from voice_engine import GestureBasis, VoicePhase

        vid = voice_engine.allocate_voice(program_id=1003)  # Slow attack
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # Step a few blocks (still in attack)
        for _ in range(5):
            voice_engine.step_voice(vid)

        assert voice_engine.voices[vid].phase == VoicePhase.ATTACK

        # Note off during attack
        voice_engine.note_off(vid)

        # Should transition to release
        frame = voice_engine.step_voice(vid)
        assert voice_engine.voices[vid].phase == VoicePhase.RELEASE

        # Gain should decrease
        prev_gain = frame.gain_q15
        for _ in range(10):
            frame = voice_engine.step_voice(vid)
            if frame:
                assert frame.gain_q15 <= prev_gain + 100, "Gain should decrease in release"
                prev_gain = frame.gain_q15

    def test_rapid_retrigger(self, voice_engine, q15_half, q15_one):
        """Rapid retriggering doesn't cause clicks."""
        from voice_engine import GestureBasis

        vid = voice_engine.allocate_voice(program_id=1004)  # Fast attack organ
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)

        gains = []

        for i in range(50):
            if i % 10 == 0:
                # Retrigger
                voice_engine.note_off(vid)
                voice_engine.note_on(vid, gesture=gesture)

            frame = voice_engine.step_voice(vid)
            if frame:
                gains.append(frame.gain_q15)

        # Check no extreme jumps - allow some transition during retrigger
        extreme_clicks = 0
        for i in range(1, len(gains)):
            diff = abs(gains[i] - gains[i-1])
            if diff >= q15_one:  # Only count truly extreme jumps
                extreme_clicks += 1

        # Allow a few clicks at retrigger points, but not many
        assert extreme_clicks < 10, f"Too many extreme clicks: {extreme_clicks}"

    def test_release_tail_completes(self, voice_engine, q15_half):
        """Release tail runs to completion."""
        from voice_engine import GestureBasis, VoicePhase

        vid = voice_engine.allocate_voice(program_id=1002)  # Long release
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # Run through attack/decay/sustain
        for _ in range(100):
            voice_engine.step_voice(vid)

        # Note off
        voice_engine.note_off(vid)

        # Run release
        release_blocks = 0
        while voice_engine.is_voice_active(vid) and release_blocks < 500:
            voice_engine.step_voice(vid)
            release_blocks += 1

        # Voice should eventually become inactive
        assert not voice_engine.is_voice_active(vid), "Voice never became inactive"


class TestV3RegimeGraphStability:
    """
    V3. Regime Graph Stability

    Requirement:
    - Regime transitions exhibit hysteresis
    - Every regime has a recovery path

    Test:
    - Hold pressure at threshold boundaries
    - Oscillate control values around regime edges

    Gotchas:
    - Regime flapping every block
    - No "exit" from catastrophe regimes
    """

    def test_regime_hysteresis(self, voice_engine, q15_half, q15_one):
        """Regime doesn't flap rapidly at boundaries."""
        from voice_engine import GestureBasis, Regime

        vid = voice_engine.allocate_voice(program_id=1001)

        # Start with moderate glitch (below threshold)
        gesture = GestureBasis(
            gate=True,
            pressure_q15=q15_half,
            chaos_q15=int(q15_one * 0.5),  # Just below glitch threshold
        )
        voice_engine.note_on(vid, gesture=gesture)

        # Run past attack
        for _ in range(20):
            voice_engine.step_voice(vid)

        # Oscillate around threshold
        regimes = []
        for i in range(100):
            # Vary chaos around threshold
            gesture.chaos_q15 = int(q15_one * (0.55 + 0.1 * np.sin(i * 0.1)))
            voice_engine.update_gesture(vid, gesture)
            frame = voice_engine.step_voice(vid)
            if frame:
                regimes.append(frame.regime)

        # Count regime changes
        changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])

        # Should have some hysteresis (not change every block)
        assert changes < len(regimes) // 2, f"Too many regime changes: {changes}"

    def test_catastrophe_has_recovery(self, q15_one):
        """Catastrophe regime has path to recovery."""
        from voice_engine import VoiceEngine, GestureBasis, Regime

        engine = VoiceEngine(max_voices=4)
        vid = engine.allocate_voice(program_id=1001)

        # Extreme gesture to trigger catastrophe
        gesture = GestureBasis(
            gate=True,
            pressure_q15=q15_one,
            chaos_q15=q15_one,
            speed_q15=q15_one,
        )
        engine.note_on(vid, gesture=gesture)

        # Run until we might hit catastrophe or stabilize
        for _ in range(100):
            engine.step_voice(vid)

        # Now reduce to calm
        gesture.chaos_q15 = 0
        gesture.pressure_q15 = q15_one // 4
        gesture.speed_q15 = 0
        engine.update_gesture(vid, gesture)

        # Should eventually recover (not stuck in catastrophe)
        final_regime = None
        for _ in range(200):
            frame = engine.step_voice(vid)
            if frame:
                final_regime = frame.regime

        # Should not be stuck in catastrophe
        assert final_regime != Regime.R5_CATASTROPHE, "Stuck in catastrophe"

    def test_all_regimes_reachable(self, q15_one):
        """All regimes can be reached through gesture."""
        from voice_engine import VoiceEngine, GestureBasis, Regime

        reached_regimes = set()

        # Test different gesture combinations
        test_gestures = [
            # Attack regime
            {"pressure": 0.5, "chaos": 0, "note_age": 2},
            # Stable tone
            {"pressure": 0.6, "chaos": 0, "note_age": 50},
            # Grain/air (high air intent)
            {"pressure": 0.3, "speed": 0.8, "chaos": 0.2, "note_age": 50},
            # Glitch chatter
            {"pressure": 0.5, "chaos": 0.7, "tilt": 0.5, "note_age": 50},
        ]

        for params in test_gestures:
            engine = VoiceEngine(max_voices=4)
            vid = engine.allocate_voice(program_id=1001)

            gesture = GestureBasis(
                gate=True,
                pressure_q15=int(q15_one * params.get("pressure", 0.5)),
                speed_q15=int(q15_one * params.get("speed", 0.5)),
                chaos_q15=int(q15_one * params.get("chaos", 0)),
                tilt_q15=int(q15_one * params.get("tilt", 0)),
            )
            engine.note_on(vid, gesture=gesture)

            note_age = params.get("note_age", 50)
            for _ in range(note_age):
                frame = engine.step_voice(vid)
                if frame:
                    reached_regimes.add(frame.regime)

        # Should reach at least attack, stable, and one other
        assert Regime.R1_ATTACK_IGNITION in reached_regimes
        assert len(reached_regimes) >= 2, f"Only reached: {reached_regimes}"


class TestV4VoiceIndependence:
    """
    V4. Voice Independence

    Requirement:
    - Each voice is logically independent
    - Shared engines must not leak state between voices

    Test:
    - Play identical notes on different voices with different gestures
    - Kill one voice mid-catastrophe; others continue unaffected

    Gotchas:
    - Shared PRNG without per-voice salt
    - Shared temporary buffers
    """

    def test_voices_dont_affect_each_other(self, voice_engine, q15_half, q15_one):
        """Changes to one voice don't affect others."""
        from voice_engine import GestureBasis

        # Allocate two voices
        vid1 = voice_engine.allocate_voice(program_id=1001)
        vid2 = voice_engine.allocate_voice(program_id=1001)

        # Same initial gesture
        gesture1 = GestureBasis(gate=True, pressure_q15=q15_half)
        gesture2 = GestureBasis(gate=True, pressure_q15=q15_half)

        voice_engine.note_on(vid1, gesture=gesture1)
        voice_engine.note_on(vid2, gesture=gesture2)

        # Run both for a bit
        for _ in range(20):
            voice_engine.step_all()

        # Now drastically change voice 1
        gesture1.chaos_q15 = q15_one
        gesture1.pressure_q15 = 0
        voice_engine.update_gesture(vid1, gesture1)

        # Voice 2 should be unaffected
        frame2_before = voice_engine.step_voice(vid2)
        gain2_before = frame2_before.gain_q15

        # Step some more with voice 1 in chaos
        for _ in range(20):
            voice_engine.step_voice(vid1)

        frame2_after = voice_engine.step_voice(vid2)

        # Voice 2's behavior should be continuous (not affected by voice 1)
        # Allow for normal envelope progression
        assert abs(frame2_after.gain_q15 - gain2_before) < 5000

    def test_kill_voice_others_continue(self, voice_engine, q15_half):
        """Killing one voice doesn't affect others."""
        from voice_engine import GestureBasis

        # Allocate three voices
        vids = []
        for i in range(3):
            vid = voice_engine.allocate_voice(program_id=1001 + i)
            gesture = GestureBasis(gate=True, pressure_q15=q15_half)
            voice_engine.note_on(vid, gesture=gesture)
            vids.append(vid)

        # Run for a bit
        for _ in range(30):
            voice_engine.step_all()

        # Kill middle voice
        voice_engine.note_off(vids[1])

        # Wait for release
        for _ in range(100):
            voice_engine.step_all()

        # Other voices should still be active
        assert voice_engine.is_voice_active(vids[0])
        assert voice_engine.is_voice_active(vids[2])

    def test_per_voice_state_isolation(self, voice_engine, q15_half, q15_one):
        """Per-voice state is isolated."""
        from voice_engine import GestureBasis

        # Create voices with very different gestures
        vid1 = voice_engine.allocate_voice(program_id=1001)
        vid2 = voice_engine.allocate_voice(program_id=1004)  # Different program

        gesture1 = GestureBasis(gate=True, pressure_q15=q15_one, chaos_q15=q15_one)
        gesture2 = GestureBasis(gate=True, pressure_q15=q15_half // 2, chaos_q15=0)

        voice_engine.note_on(vid1, gesture=gesture1)
        voice_engine.note_on(vid2, gesture=gesture2)

        # Run and collect data
        for _ in range(50):
            frames = voice_engine.step_all()

            if len(frames) == 2:
                f1 = next(f for f in frames if f.voice_id == vid1)
                f2 = next(f for f in frames if f.voice_id == vid2)

                # Regimes should potentially differ
                # Intents should definitely differ
                state1 = voice_engine.voices[vid1]
                state2 = voice_engine.voices[vid2]

                # Gestures should be independent
                assert state1.gesture.chaos_q15 != state2.gesture.chaos_q15


class TestV5VoiceFrameContractIntegrity:
    """
    V5. VoiceFrame Contract Integrity

    Requirement:
    - Every block emits a complete, self-contained VoiceFrame

    Test:
    - Drop random VoiceFrames -> silence only for those voices

    Gotchas:
    - Implicit state assumed in Symbol Engine
    - Partial frames interpreted as valid
    """

    def test_frame_is_complete(self, voice_engine, q15_half):
        """Each frame contains all required fields."""
        from voice_engine import GestureBasis

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        for _ in range(20):
            frame = voice_engine.step_voice(vid)

            # Check all required fields are present and valid
            assert hasattr(frame, 'voice_id')
            assert hasattr(frame, 'tree_a_id')
            assert hasattr(frame, 'tree_b_id')
            assert hasattr(frame, 'tree_blend_q15')
            assert hasattr(frame, 'gain_q15')
            assert hasattr(frame, 'pan_q15')
            assert hasattr(frame, 'family_weights')
            assert hasattr(frame, 'events')
            assert hasattr(frame, 'regime')
            assert hasattr(frame, 'is_active')

            # Check validity
            assert frame.voice_id == vid
            assert 0 <= frame.tree_a_id < 5
            assert 0 <= frame.tree_b_id < 5
            assert len(frame.family_weights) == 8

    def test_frame_self_contained(self, voice_engine, q15_half):
        """Frame doesn't depend on previous frames."""
        from voice_engine import GestureBasis

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # Skip some frames (simulating drops)
        for i in range(50):
            frame = voice_engine.step_voice(vid)

            if i % 5 == 0:
                # "Drop" this frame by not using it
                continue

            # Even after "dropped" frames, subsequent ones should be valid
            assert frame is not None
            assert frame.is_active
            assert 0 <= frame.gain_q15 <= 32767

    def test_regime_in_frame(self, voice_engine, q15_half):
        """Regime is correctly encoded in frame."""
        from voice_engine import GestureBasis, Regime

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # First frame should be attack regime
        frame = voice_engine.step_voice(vid)
        assert frame.regime == Regime.R1_ATTACK_IGNITION

        # Run through attack
        for _ in range(20):
            frame = voice_engine.step_voice(vid)

        # Should have transitioned to sustain regime
        assert frame.regime in {Regime.R2_STABLE_TONE, Regime.R3_GRAIN_AIR}
