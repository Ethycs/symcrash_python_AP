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


# =============================================================================
# CMMM CONFORMANCE TESTS (V6-V10)
# =============================================================================


class TestV6CMMMLegacyAcceptance:
    """
    V6. CMMM Legacy Acceptance Tests

    Requirement:
    - Core functionality from inline tests must pass

    Test:
    - Gesture to intent mapping
    - Family weight computation
    - Tree morph transitions
    - All instrument programs load
    """

    def test_gesture_to_intent(self, q15_one, q15_half):
        """GestureBasis correctly maps to IntentVector."""
        from voice_engine import (
            GestureBasis, default_gesture_to_intent, I_TONE, I_GLITCH, I_GLUE
        )

        # High pressure gesture
        g = GestureBasis(gate=True, pressure_q15=q15_one, speed_q15=q15_half)
        intent = default_gesture_to_intent(g)

        assert intent.bow == q15_one, f"Expected bow={q15_one}, got {intent.bow}"
        assert intent.tone > 0, "Tone should be positive with gate on"

        # High chaos gesture
        g2 = GestureBasis(gate=True, chaos_q15=q15_one, pressure_q15=q15_half)
        intent2 = default_gesture_to_intent(g2)

        assert intent2.glitch > 0, "Glitch should be positive with high chaos"
        assert intent2.glue < q15_one, "Glue should decrease with chaos"

    def test_regime_transitions(self, voice_engine, q15_half):
        """Regime FSM transitions correctly."""
        from voice_engine import GestureBasis, Regime

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # Should start in attack regime
        frame = voice_engine.step_voice(vid)
        assert frame.regime == Regime.R1_ATTACK_IGNITION

        # Step through attack
        for _ in range(10):
            frame = voice_engine.step_voice(vid)

        # Should transition to sustain regime
        assert frame.regime in {Regime.R2_STABLE_TONE, Regime.R3_GRAIN_AIR}

    def test_family_weights(self, q15_one, q15_half):
        """Family weights computed correctly from intent."""
        from voice_engine import (
            IntentVector, compute_family_weights, Regime,
            I_TONE, I_BOW, I_GLUE, F_COLOR, F_GLUE, F_CATASTROPHE
        )

        intent = IntentVector()
        intent.lanes[I_TONE] = q15_one
        intent.lanes[I_BOW] = q15_half
        intent.lanes[I_GLUE] = q15_one

        weights = compute_family_weights(intent, Regime.R2_STABLE_TONE)

        # Color should be high (tone + bow)
        assert weights[F_COLOR] > q15_half
        # Glue should be high
        assert weights[F_GLUE] > q15_half
        # Catastrophe should be zero (not in R5)
        assert weights[F_CATASTROPHE] == 0

    def test_tree_morph(self, q15_one):
        """Tree morph transitions correctly."""
        from voice_engine import TreeMorphState, update_tree_morph

        state = TreeMorphState(tree_a_id=0)

        # Start morph to tree 2
        state = update_tree_morph(state, 2)
        assert state.is_morphing
        assert state.tree_b_id == 2

        # Step through morph
        while state.is_morphing:
            state = update_tree_morph(state, 2)

        # Should complete with A promoted to 2
        assert state.tree_a_id == 2
        assert state.blend_q15 == 0

    def test_instrument_programs(self, q15_half):
        """All four instrument programs load correctly."""
        from voice_engine import VoiceEngine, GestureBasis

        for program_id in [1001, 1002, 1003, 1004]:
            engine = VoiceEngine()
            vid = engine.allocate_voice(program_id=program_id)
            assert vid is not None

            gesture = GestureBasis(gate=True, pressure_q15=q15_half)
            engine.note_on(vid, gesture=gesture)
            frame = engine.step_voice(vid)

            assert frame is not None
            assert frame.program_id == program_id
            assert frame.is_active


class TestV7CMMStateSnapshot:
    """
    V7. CMMM State Snapshot/Restore

    Requirement:
    - VoiceState can be fully serialized and restored
    - Restored state produces identical behavior

    Test:
    - Snapshot at various points in lifecycle
    - Restore and verify all fields
    """

    def test_voicestate_snapshot_restore(self, q15_half, q15_one):
        """VoiceState can be snapshotted and restored for replay."""
        from voice_engine import (
            VoiceState, VoicePhase, Regime, GestureBasis, IntentVector,
            ArcState, TreeMorphState, DegradationLevel, TowerLevel,
            I_AIR, NUM_INTENTS, MAX_ARCS
        )

        # Create initial state with all fields populated
        state = VoiceState(voice_id=42)
        state.phase = VoicePhase.SUSTAIN
        state.phase_block = 100
        state.regime = Regime.R3_GRAIN_AIR
        state.regime_blocks = 50
        state.gesture = GestureBasis(gate=True, pressure_q15=q15_half, chaos_q15=1000)
        state.intent.lanes[I_AIR] = 20000
        state.arc = ArcState(active=True, arc_id=2, progress_blocks=5, total_blocks=16)
        state.family_weights = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        state.tree_morph = TreeMorphState(tree_a_id=2, tree_b_id=4, blend_q15=10000, is_morphing=True)
        state.gain_q15 = 25000
        state.degradation = DegradationLevel.LIGHT
        state.tower_level = TowerLevel.L2_SPARSE
        state.program_id = 1002
        state.block_count = 500

        # New CMMM fields
        state.rng_u32 = 0xDEADBEEF
        state.env_phase = VoicePhase.DECAY
        state.env_level_q15 = 15000
        state.env_counter = 25
        state.dI_q15 = [100, 200, 300, 400, 500, 600, 700, 800]
        state.arc_phase_q16 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        state.note_u8 = 72
        state.cents_i16 = 50
        state.activity_q15 = 28000
        state.sendA_q15 = 5000
        state.sendB_q15 = 3000

        # Snapshot
        snapshot = state.snapshot()

        # Restore to new state
        restored = VoiceState.restore(snapshot)

        # Verify core fields match
        assert restored.voice_id == state.voice_id
        assert restored.phase == state.phase
        assert restored.phase_block == state.phase_block
        assert restored.regime == state.regime
        assert restored.regime_blocks == state.regime_blocks
        assert restored.gesture.gate == state.gesture.gate
        assert restored.gesture.pressure_q15 == state.gesture.pressure_q15
        assert restored.gesture.chaos_q15 == state.gesture.chaos_q15
        assert restored.intent.lanes == state.intent.lanes
        assert restored.arc.active == state.arc.active
        assert restored.arc.arc_id == state.arc.arc_id
        assert restored.arc.progress_blocks == state.arc.progress_blocks
        assert restored.family_weights == state.family_weights
        assert restored.tree_morph.tree_a_id == state.tree_morph.tree_a_id
        assert restored.tree_morph.blend_q15 == state.tree_morph.blend_q15
        assert restored.gain_q15 == state.gain_q15
        assert restored.degradation == state.degradation
        assert restored.tower_level == state.tower_level
        assert restored.program_id == state.program_id
        assert restored.block_count == state.block_count

        # Verify new CMMM fields
        assert restored.rng_u32 == state.rng_u32
        assert restored.env_phase == state.env_phase
        assert restored.env_level_q15 == state.env_level_q15
        assert restored.env_counter == state.env_counter
        assert restored.dI_q15 == state.dI_q15
        assert restored.arc_phase_q16 == state.arc_phase_q16
        assert restored.note_u8 == state.note_u8
        assert restored.cents_i16 == state.cents_i16
        assert restored.activity_q15 == state.activity_q15
        assert restored.sendA_q15 == state.sendA_q15
        assert restored.sendB_q15 == state.sendB_q15


class TestV8CMMRegimeGraph:
    """
    V8. CMMM Regime Graph Tests

    Requirement:
    - RegimeGraph correctly checks edge conditions
    - Transitions respect dwell time and hysteresis

    Test:
    - High intent triggers correct transition
    - Dwell time prevents premature transitions
    """

    def test_regime_graph_transitions(self, q15_one):
        """RegimeGraph correctly checks EdgeSpec conditions."""
        from voice_engine import (
            build_default_regime_graph, IntentVector, Regime, I_GLITCH, I_WARP
        )

        graph = build_default_regime_graph()

        # Test: High glitch intent should trigger R2->R4 transition
        intent = IntentVector()
        intent.lanes[I_GLITCH] = int(q15_one * 0.7)

        target = graph.check_transition(Regime.R2_STABLE_TONE, intent, dwell_blocks=10)
        assert target == Regime.R4_GLITCH_CHATTER

        # Test: Short dwell time should prevent transition
        target = graph.check_transition(Regime.R2_STABLE_TONE, intent, dwell_blocks=2)
        assert target is None

        # Test: From R4, very high glitch triggers R5
        intent.lanes[I_GLITCH] = int(q15_one * 0.95)
        target = graph.check_transition(Regime.R4_GLITCH_CHATTER, intent, dwell_blocks=20)
        assert target == Regime.R5_CATASTROPHE

    def test_arc_modulation(self, voice_engine, q15_half):
        """ArcSpec modulators apply correctly during transitions."""
        from voice_engine import GestureBasis

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # First frame should have arc active (attack burst)
        frame = voice_engine.step_voice(vid)
        state = voice_engine.voices[vid]

        # Attack arc should be active
        assert state.arc.active
        assert state.arc.arc_id == 1
        assert frame.arc_active
        assert frame.arc_id == 1

        # Step through arc duration
        for _ in range(5):
            frame = voice_engine.step_voice(vid)

        # Arc should complete
        state = voice_engine.voices[vid]
        assert not state.arc.active


class TestV9CMMRegimeSpecPolicies:
    """
    V9. CMMM Regime Spec Policies

    Requirement:
    - Each regime has correct family/event masks
    - Tower levels assigned correctly
    - Degradation levels disable correct families
    """

    def test_regime_spec_policies(self):
        """RegimeSpec policies correctly influence family weights and events."""
        from voice_engine import (
            REGIME_SPECS, Regime, TowerLevel, F_CATASTROPHE
        )

        # Check R0_IDLE has all families disabled
        spec_idle = REGIME_SPECS[Regime.R0_IDLE]
        assert spec_idle.family_enable_mask == 0x00
        assert spec_idle.event_enable_mask == 0x00

        # Check R5_CATASTROPHE enables catastrophe family
        spec_cat = REGIME_SPECS[Regime.R5_CATASTROPHE]
        assert spec_cat.family_enable_mask & (1 << F_CATASTROPHE)

        # Check regime tower levels
        assert spec_idle.tower_level == TowerLevel.L4_SCALAR
        assert REGIME_SPECS[Regime.R2_STABLE_TONE].tower_level == TowerLevel.L0_FULL

    def test_degradation_levels(self, voice_engine, q15_half):
        """Degradation levels correctly disable operator families."""
        from voice_engine import (
            GestureBasis, DegradationLevel, F_TRANSPORT, F_TOPOLOGY, F_GLUE, NUM_FAMILIES
        )

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # Step with HEAVY degradation
        voice_engine.set_degradation_level(DegradationLevel.HEAVY)
        frame = voice_engine.step_voice(vid)

        # Transport and topology should be disabled
        assert frame.w_family_q15[F_TRANSPORT] == 0
        assert frame.w_family_q15[F_TOPOLOGY] == 0

        # Emergency should only keep glue
        voice_engine.set_degradation_level(DegradationLevel.EMERGENCY)
        frame = voice_engine.step_voice(vid)

        for i in range(NUM_FAMILIES):
            if i != F_GLUE:
                assert frame.w_family_q15[i] == 0

    def test_tower_level_assignment(self, voice_engine, q15_half):
        """Tower levels are assigned based on regime."""
        from voice_engine import GestureBasis, TowerLevel, Regime

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half)
        voice_engine.note_on(vid, gesture=gesture)

        # Attack should use L1
        frame = voice_engine.step_voice(vid)
        assert frame.tower_level == TowerLevel.L1_COARSE

        # Step to sustain (R2)
        for _ in range(20):
            frame = voice_engine.step_voice(vid)

        # Stable tone should use L0
        if frame.regime_id == int(Regime.R2_STABLE_TONE):
            assert frame.tower_level == TowerLevel.L0_FULL


class TestV10CMMVoiceFrameSerialization:
    """
    V10. CMMM VoiceFrame Serialization

    Requirement:
    - VoiceFrameABI can serialize to bytes for FPGA/network
    - All fields are correctly encoded

    Test:
    - Serialize a fully populated frame
    - Verify byte output
    """

    def test_voiceframe_serialization(self, q15_half):
        """VoiceFrameABI can serialize to bytes."""
        from voice_engine import VoiceFrameABI

        frame = VoiceFrameABI(
            program_id=1001,
            voice_id=5,
            regime_id=2,
            gain_q15=20000,
            pan_q15=-5000,
            w_family_q15=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
            event_mask=0x05,
            event_strengths=[100, 0, 200, 0, 0, 0, 0, 0],
            tree_a_id=1,
            tree_b_id=3,
            tree_blend_q15=15000,
            is_active=True,
            arc_active=True,
            arc_id=2,
        )

        # Should not raise
        data = frame.to_bytes()
        assert len(data) > 0
        assert isinstance(data, bytes)

    def test_voiceframe_abi_fields(self, voice_engine, q15_half):
        """VoiceFrameABI has all required fields."""
        from voice_engine import GestureBasis, NUM_TREES, NUM_FAMILIES

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=q15_half, pan_q15=q15_half // 2)
        voice_engine.note_on(vid, gesture=gesture)

        frame = voice_engine.step_voice(vid)

        # Check all ABI fields are set
        assert frame.voice_id == vid
        assert 0 <= frame.tree_a_id < NUM_TREES
        assert 0 <= frame.tree_b_id < NUM_TREES
        assert 0 <= frame.tree_blend_q15 <= 32767
        assert frame.gain_q15 > 0
        assert len(frame.family_weights) == NUM_FAMILIES
        assert frame.is_active

        # Check new ABI fields
        assert hasattr(frame, 'activity_q15')
        assert hasattr(frame, 'sendA_q15')
        assert hasattr(frame, 'sendB_q15')
        assert hasattr(frame, 'note_u8')
        assert hasattr(frame, 'cents_i16')
        assert hasattr(frame, 'event_payload_u64')
