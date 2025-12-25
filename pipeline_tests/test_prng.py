"""
WSE-STD-040 PRNG Conformance Tests
===================================

Tests for deterministic pseudorandomness per WSE-STD-040.
"""

import pytest
from symcrash_python_ap.prng import (
    # Core PRNG
    PCG32,
    # Seeding
    hash64_mix,
    hash64_seed,
    # Per-voice state
    VoicePRNG,
    # Operator contract
    OPERATOR_PRNG_DRAWS,
    get_operator_prng_draws,
    validate_operator_draws,
    # Fixed-point conversion
    prng_to_q15,
    prng_to_q15_unsigned,
    prng_to_range,
    # Manager
    PRNGManager,
)


# =============================================================================
# PCG-32 CORE TESTS
# =============================================================================

class TestPCG32:
    """Test PCG-32 generator properties."""

    def test_deterministic_sequence(self):
        """Same seed should produce same sequence."""
        rng1 = PCG32(state=12345)
        rng2 = PCG32(state=12345)

        for _ in range(100):
            assert rng1.next_u32() == rng2.next_u32()

    def test_different_seeds_different_sequences(self):
        """Different seeds should produce different sequences."""
        rng1 = PCG32(state=12345)
        rng2 = PCG32(state=67890)

        # Very unlikely to match
        seq1 = [rng1.next_u32() for _ in range(10)]
        seq2 = [rng2.next_u32() for _ in range(10)]
        assert seq1 != seq2

    def test_different_streams_different_sequences(self):
        """Different streams should produce different sequences."""
        rng1 = PCG32(state=12345, stream=1)
        rng2 = PCG32(state=12345, stream=3)

        seq1 = [rng1.next_u32() for _ in range(10)]
        seq2 = [rng2.next_u32() for _ in range(10)]
        assert seq1 != seq2

    def test_output_range_u32(self):
        """Output should be 32-bit unsigned."""
        rng = PCG32(state=42)
        for _ in range(1000):
            val = rng.next_u32()
            assert 0 <= val <= 0xFFFFFFFF

    def test_output_range_q15(self):
        """Q15 output should be in [-16384, 16383]."""
        rng = PCG32(state=42)
        for _ in range(1000):
            val = rng.next_q15()
            assert -16384 <= val <= 16383

    def test_output_range_u8(self):
        """U8 output should be in [0, 255]."""
        rng = PCG32(state=42)
        for _ in range(1000):
            val = rng.next_u8()
            assert 0 <= val <= 255

    def test_output_range_float(self):
        """Float output should be in [0.0, 1.0)."""
        rng = PCG32(state=42)
        for _ in range(1000):
            val = rng.next_float()
            assert 0.0 <= val < 1.0

    def test_output_range_float_bipolar(self):
        """Bipolar float output should be in [-1.0, 1.0)."""
        rng = PCG32(state=42)
        for _ in range(1000):
            val = rng.next_float_bipolar()
            assert -1.0 <= val < 1.0

    def test_state_serialization(self):
        """State should be serializable and restorable."""
        rng1 = PCG32(state=12345)

        # Generate some values
        for _ in range(50):
            rng1.next_u32()

        # Save state
        saved_state = rng1.state
        saved_stream = rng1.stream

        # Generate more values
        expected = [rng1.next_u32() for _ in range(10)]

        # Create new RNG and restore state
        rng2 = PCG32(state=0, stream=saved_stream)
        rng2.state = saved_state

        # Should produce same sequence
        actual = [rng2.next_u32() for _ in range(10)]
        assert actual == expected

    def test_clone(self):
        """Clone should produce identical sequence from that point."""
        rng1 = PCG32(state=12345)

        # Generate some values
        for _ in range(50):
            rng1.next_u32()

        # Clone
        rng2 = rng1.clone()

        # Both should produce same sequence
        for _ in range(100):
            assert rng1.next_u32() == rng2.next_u32()

    def test_statistical_distribution(self):
        """Output should be reasonably distributed."""
        rng = PCG32(state=42)

        # Generate many values and check distribution
        values = [rng.next_u32() for _ in range(10000)]

        # Check we're not stuck
        assert len(set(values)) > 9000  # Should have many unique values

        # Check rough uniformity (each quartile should have ~25%)
        q1 = sum(1 for v in values if v < 0x40000000) / len(values)
        q2 = sum(1 for v in values if 0x40000000 <= v < 0x80000000) / len(values)
        q3 = sum(1 for v in values if 0x80000000 <= v < 0xC0000000) / len(values)
        q4 = sum(1 for v in values if v >= 0xC0000000) / len(values)

        assert 0.20 < q1 < 0.30
        assert 0.20 < q2 < 0.30
        assert 0.20 < q3 < 0.30
        assert 0.20 < q4 < 0.30


# =============================================================================
# HASH64 SEEDING TESTS
# =============================================================================

class TestHash64:
    """Test deterministic seeding per Section 4.3."""

    def test_hash64_deterministic(self):
        """Hash should be deterministic."""
        h1 = hash64_seed(global_seed=123, voice_id=5, note_id=60, preset_id=1)
        h2 = hash64_seed(global_seed=123, voice_id=5, note_id=60, preset_id=1)
        assert h1 == h2

    def test_hash64_different_inputs(self):
        """Different inputs should produce different seeds."""
        h1 = hash64_seed(global_seed=123, voice_id=5, note_id=60, preset_id=1)
        h2 = hash64_seed(global_seed=123, voice_id=6, note_id=60, preset_id=1)
        h3 = hash64_seed(global_seed=123, voice_id=5, note_id=61, preset_id=1)
        h4 = hash64_seed(global_seed=123, voice_id=5, note_id=60, preset_id=2)

        # All should be different
        assert len({h1, h2, h3, h4}) == 4

    def test_hash64_zero_default(self):
        """Global seed defaults to 0."""
        h1 = hash64_seed(global_seed=0, voice_id=5, note_id=60, preset_id=1)
        h2 = hash64_seed(voice_id=5, note_id=60, preset_id=1)  # Default
        assert h1 == h2

    def test_hash64_64bit_output(self):
        """Output should be 64-bit."""
        h = hash64_seed(global_seed=0xFFFFFFFFFFFFFFFF, voice_id=0xFFFF)
        assert 0 <= h <= 0xFFFFFFFFFFFFFFFF

    def test_hash64_mix_associativity(self):
        """hash64_mix should produce consistent results."""
        a, b = 12345, 67890
        h1 = hash64_mix(a, b)
        h2 = hash64_mix(a, b)
        assert h1 == h2


# =============================================================================
# PER-VOICE PRNG TESTS
# =============================================================================

class TestVoicePRNG:
    """Test per-voice PRNG state per Section 4.2."""

    def test_create_with_seeding(self):
        """VoicePRNG should seed correctly."""
        vprng1 = VoicePRNG.create(voice_id=0, global_seed=123, note_id=60, preset_id=1)
        vprng2 = VoicePRNG.create(voice_id=0, global_seed=123, note_id=60, preset_id=1)

        # Same parameters should produce same sequence
        for _ in range(10):
            assert vprng1.draw_u32() == vprng2.draw_u32()

    def test_different_voices_independent(self):
        """Different voices should have independent sequences."""
        vprng1 = VoicePRNG.create(voice_id=0, global_seed=123)
        vprng2 = VoicePRNG.create(voice_id=1, global_seed=123)

        seq1 = [vprng1.draw_u32() for _ in range(10)]
        seq2 = [vprng2.draw_u32() for _ in range(10)]
        assert seq1 != seq2

    def test_draw_tracking(self):
        """Draw counts should be tracked."""
        vprng = VoicePRNG.create(voice_id=0)

        assert vprng.draws_this_block == 0
        assert vprng.total_draws == 0

        vprng.draw_u32()
        assert vprng.draws_this_block == 1
        assert vprng.total_draws == 1

        vprng.draw_q15()
        assert vprng.draws_this_block == 2
        assert vprng.total_draws == 2

    def test_begin_block_resets_count(self):
        """begin_block should reset per-block counter."""
        vprng = VoicePRNG.create(voice_id=0)

        vprng.draw_u32()
        vprng.draw_u32()
        assert vprng.draws_this_block == 2
        assert vprng.total_draws == 2

        vprng.begin_block()

        assert vprng.draws_this_block == 0
        assert vprng.total_draws == 2  # Total preserved

    def test_frozen_does_not_advance(self):
        """Frozen PRNG should not advance (Section 5.1)."""
        vprng = VoicePRNG.create(voice_id=0, global_seed=42)

        # Draw some values
        val1 = vprng.draw_u32()
        state_before = vprng.prng.state

        # Freeze
        vprng.freeze()

        # Draws should return 0 and not advance
        assert vprng.draw_u32() == 0
        assert vprng.prng.state == state_before

        # Unfreeze
        vprng.unfreeze()

        # Now should advance again
        val2 = vprng.draw_u32()
        assert val2 != 0
        assert vprng.prng.state != state_before

    def test_draw_n(self):
        """draw_n should draw exactly N values."""
        vprng = VoicePRNG.create(voice_id=0)

        values = vprng.draw_n(5)

        assert len(values) == 5
        assert vprng.draws_this_block == 5
        assert vprng.total_draws == 5

    def test_state_serialization(self):
        """State should be serializable."""
        vprng1 = VoicePRNG.create(voice_id=0, global_seed=42)

        # Generate some values
        for _ in range(20):
            vprng1.draw_u32()

        # Save state
        state, stream = vprng1.get_state()

        # Generate expected values
        expected = [vprng1.draw_u32() for _ in range(10)]

        # Create new and restore
        vprng2 = VoicePRNG.create(voice_id=0)
        vprng2.set_state(state, stream)

        # Should produce same sequence
        actual = [vprng2.draw_u32() for _ in range(10)]
        assert actual == expected


# =============================================================================
# PRNG ADVANCEMENT RULES TESTS (Section 5)
# =============================================================================

class TestPRNGAdvancement:
    """Test PRNG advancement rules per Section 5."""

    def test_skipped_voice_not_advanced(self):
        """Skipped voices should NOT advance PRNG (Section 5.1)."""
        manager = PRNGManager(global_seed=42)

        # Allocate two voices
        v0 = manager.allocate_voice(voice_id=0)
        v1 = manager.allocate_voice(voice_id=1)

        # Service voice 0, skip voice 1
        v0_val = v0.draw_u32()
        v1_state_before = v1.prng.state

        # Voice 1 should not have advanced
        assert v1.prng.state == v1_state_before

        # Now service voice 1
        v1_val = v1.draw_u32()
        assert v1.prng.state != v1_state_before

    def test_frozen_voice_not_advanced(self):
        """Frozen voices should NOT advance PRNG."""
        manager = PRNGManager(global_seed=42)
        v0 = manager.allocate_voice(voice_id=0)

        v0.draw_u32()
        state_after_first = v0.prng.state

        manager.freeze_voice(0)

        # Try to draw while frozen
        v0.draw_u32()
        assert v0.prng.state == state_after_first  # Should not change

    def test_draw_order_determinism(self):
        """Draw order within voice should be deterministic (Section 5.2)."""
        vprng1 = VoicePRNG.create(voice_id=0, global_seed=123)
        vprng2 = VoicePRNG.create(voice_id=0, global_seed=123)

        # Simulate operator execution order
        # Op1: 2 draws, Op2: 1 draw, Op3: 3 draws
        op1_v1 = vprng1.draw_n(2)
        op2_v1 = vprng1.draw_n(1)
        op3_v1 = vprng1.draw_n(3)

        op1_v2 = vprng2.draw_n(2)
        op2_v2 = vprng2.draw_n(1)
        op3_v2 = vprng2.draw_n(3)

        assert op1_v1 == op1_v2
        assert op2_v1 == op2_v2
        assert op3_v1 == op3_v2


# =============================================================================
# OPERATOR CONTRACT TESTS (Section 7)
# =============================================================================

class TestOperatorContract:
    """Test operator PRNG contract per Section 7."""

    def test_noise_band_draws(self):
        """NoiseBand should have 2 draws per spec example."""
        assert get_operator_prng_draws('NoiseBand') == 2

    def test_quasi_rotate_draws(self):
        """QuasiRotate should have 1 draw per spec example."""
        assert get_operator_prng_draws('QuasiRotate') == 1

    def test_non_random_operators_zero_draws(self):
        """Non-random operators should have 0 draws."""
        assert get_operator_prng_draws('Identity') == 0
        assert get_operator_prng_draws('Tilt') == 0
        assert get_operator_prng_draws('FineClamp') == 0
        assert get_operator_prng_draws('NormalizeScale') == 0

    def test_unknown_operator_zero_draws(self):
        """Unknown operators should default to 0 draws."""
        assert get_operator_prng_draws('NonexistentOp') == 0

    def test_validate_correct_draws(self):
        """Validation should pass for correct draw count."""
        assert validate_operator_draws('NoiseBand', 2)
        assert validate_operator_draws('QuasiRotate', 1)
        assert validate_operator_draws('Identity', 0)

    def test_validate_incorrect_draws(self):
        """Validation should fail for incorrect draw count."""
        assert not validate_operator_draws('NoiseBand', 1)
        assert not validate_operator_draws('NoiseBand', 3)
        assert not validate_operator_draws('QuasiRotate', 0)


# =============================================================================
# FIXED-POINT CONVERSION TESTS (Section 9)
# =============================================================================

class TestFixedPointConversion:
    """Test fixed-point conversion per Section 9."""

    def test_prng_to_q15_range(self):
        """prng_to_q15 output should be in [-16384, 16383]."""
        for r in [0, 0x80000000, 0xFFFFFFFF, 0x12345678]:
            q15 = prng_to_q15(r)
            assert -16384 <= q15 <= 16383

    def test_prng_to_q15_formula(self):
        """prng_to_q15 should follow spec formula."""
        # Per spec: int16 q15 = (r >> 17) - 0x4000
        r = 0xABCD1234
        expected = (r >> 17) - 0x4000
        assert prng_to_q15(r) == expected

    def test_prng_to_q15_unsigned_range(self):
        """prng_to_q15_unsigned output should be in [0, 32767]."""
        for r in [0, 0x80000000, 0xFFFFFFFF, 0x12345678]:
            q15 = prng_to_q15_unsigned(r)
            assert 0 <= q15 <= 32767

    def test_prng_to_range(self):
        """prng_to_range should produce values in range."""
        rng = PCG32(state=42)

        for _ in range(1000):
            r = rng.next_u32()
            val = prng_to_range(r, 10, 20)
            assert 10 <= val <= 20

    def test_prng_to_range_single_value(self):
        """prng_to_range with min==max should return that value."""
        val = prng_to_range(0xABCDEF12, 42, 42)
        assert val == 42


# =============================================================================
# PRNG MANAGER TESTS
# =============================================================================

class TestPRNGManager:
    """Test PRNG manager for voice engine integration."""

    def test_allocate_voice(self):
        """Manager should allocate voice PRNG."""
        manager = PRNGManager(global_seed=123)
        vprng = manager.allocate_voice(voice_id=5, note_id=60, preset_id=1)

        assert vprng is not None
        assert vprng.voice_id == 5
        assert manager.get_voice(5) is vprng

    def test_deallocate_voice(self):
        """Manager should deallocate voice PRNG."""
        manager = PRNGManager()
        manager.allocate_voice(voice_id=5)

        assert manager.get_voice(5) is not None

        manager.deallocate_voice(5)

        assert manager.get_voice(5) is None

    def test_reallocate_voice(self):
        """Reallocating voice should reseed PRNG."""
        manager = PRNGManager(global_seed=123)

        v1 = manager.allocate_voice(voice_id=0, note_id=60)
        seq1 = [v1.draw_u32() for _ in range(5)]

        # Reallocate with different note
        v2 = manager.allocate_voice(voice_id=0, note_id=72)
        seq2 = [v2.draw_u32() for _ in range(5)]

        assert seq1 != seq2  # Different note_id = different seed

    def test_begin_block(self):
        """Manager should call begin_block on all voices."""
        manager = PRNGManager()
        v0 = manager.allocate_voice(voice_id=0)
        v1 = manager.allocate_voice(voice_id=1)

        v0.draw_u32()
        v1.draw_u32()
        v1.draw_u32()

        assert v0.draws_this_block == 1
        assert v1.draws_this_block == 2

        manager.begin_block()

        assert v0.draws_this_block == 0
        assert v1.draws_this_block == 0

    def test_freeze_unfreeze(self):
        """Manager should freeze/unfreeze voices."""
        manager = PRNGManager()
        v0 = manager.allocate_voice(voice_id=0)

        manager.freeze_voice(0)
        assert v0.frozen

        manager.unfreeze_voice(0)
        assert not v0.frozen

    def test_set_global_seed(self):
        """Setting global seed should affect future allocations."""
        manager = PRNGManager(global_seed=0)

        v0_seed0 = manager.allocate_voice(voice_id=0)
        seq0 = [v0_seed0.draw_u32() for _ in range(5)]

        manager.set_global_seed(12345)

        # Reallocate with new global seed
        v0_seed1 = manager.allocate_voice(voice_id=0)
        seq1 = [v0_seed1.draw_u32() for _ in range(5)]

        assert seq0 != seq1


# =============================================================================
# REPLAY DETERMINISM TESTS (Section 10)
# =============================================================================

class TestReplayDeterminism:
    """Test replay guarantee per Section 10."""

    def test_identical_replay(self):
        """Identical inputs should produce identical outputs."""
        # Simulate two identical "performances"
        def run_performance(global_seed, note_sequence):
            manager = PRNGManager(global_seed=global_seed)
            results = []

            for voice_id, note_id in note_sequence:
                vprng = manager.allocate_voice(
                    voice_id=voice_id,
                    note_id=note_id,
                    preset_id=1,
                )

                # Simulate block processing
                manager.begin_block()
                block_results = [vprng.draw_u32() for _ in range(10)]
                results.append(block_results)

            return results

        note_seq = [(0, 60), (1, 64), (0, 67), (2, 72)]

        results1 = run_performance(global_seed=123, note_sequence=note_seq)
        results2 = run_performance(global_seed=123, note_sequence=note_seq)

        assert results1 == results2

    def test_reordering_does_not_affect_results(self):
        """Voice reordering should not affect individual voice results."""
        manager = PRNGManager(global_seed=42)

        # Allocate voices
        v0 = manager.allocate_voice(voice_id=0, note_id=60)
        v1 = manager.allocate_voice(voice_id=1, note_id=64)

        # Process in order 0, 1
        manager.begin_block()
        v0_results_order1 = [v0.draw_u32() for _ in range(5)]
        v1_results_order1 = [v1.draw_u32() for _ in range(5)]

        # Reset and process in order 1, 0
        manager = PRNGManager(global_seed=42)
        v0 = manager.allocate_voice(voice_id=0, note_id=60)
        v1 = manager.allocate_voice(voice_id=1, note_id=64)

        manager.begin_block()
        v1_results_order2 = [v1.draw_u32() for _ in range(5)]
        v0_results_order2 = [v0.draw_u32() for _ in range(5)]

        # Each voice should produce same results regardless of order
        assert v0_results_order1 == v0_results_order2
        assert v1_results_order1 == v1_results_order2
