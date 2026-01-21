"""
Tests for Envelope Suite
========================

Validates all envelope generators and followers in the envelope_suite module.
"""

import pytest
import math

from symcrash_python_ap.envelope_suite import (
    ADSR, AR, Decay, EnvelopeFollower, Slew, LFO,
    EnvelopePhase, LFOShape,
    BLOCKS_PER_SECOND, ms_to_blocks, hz_to_phase_inc,
)
from symcrash_python_ap.primitives import Q15_ONE


class TestConstants:
    """Test utility functions and constants."""

    def test_blocks_per_second(self):
        """Verify block rate matches 48kHz/64."""
        assert BLOCKS_PER_SECOND == 750

    def test_ms_to_blocks(self):
        """Test millisecond to block conversion."""
        # 1000ms = 750 blocks
        assert ms_to_blocks(1000) == 750
        # 1ms = 1 block (minimum)
        assert ms_to_blocks(1) >= 1
        # 0ms = 1 block (minimum)
        assert ms_to_blocks(0) == 1

    def test_hz_to_phase_inc(self):
        """Test Hz to phase increment conversion."""
        # 1 Hz should complete one cycle in 750 blocks
        inc = hz_to_phase_inc(1.0)
        # After 750 blocks, phase should wrap ~65536
        total_phase = inc * 750
        assert 60000 < total_phase < 70000  # ~65536 with rounding


class TestADSR:
    """Test ADSR envelope generator."""

    def test_idle_returns_zero(self):
        """Idle envelope returns zero."""
        env = ADSR()
        assert env.tick() == 0
        assert not env.is_active

    def test_trigger_starts_attack(self):
        """Trigger starts attack phase."""
        env = ADSR(attack_ms=10)
        env.trigger()
        assert env.is_active
        assert env.phase == EnvelopePhase.ATTACK

    def test_attack_rises_to_peak(self):
        """Attack phase rises to Q15_ONE."""
        env = ADSR(attack_ms=10, decay_ms=100, sustain=0.5)
        env.trigger()

        # Run through attack
        values = []
        for _ in range(ms_to_blocks(15)):
            values.append(env.tick())

        # Should have risen to peak
        assert max(values) == Q15_ONE
        # Should be monotonically increasing during attack
        attack_values = values[:ms_to_blocks(10)]
        for i in range(1, len(attack_values)):
            assert attack_values[i] >= attack_values[i - 1]

    def test_decay_falls_to_sustain(self):
        """Decay phase falls to sustain level."""
        env = ADSR(attack_ms=1, decay_ms=50, sustain=0.5, release_ms=100)
        env.trigger()

        # Run through attack + decay
        for _ in range(ms_to_blocks(60)):
            env.tick()

        # Should be at sustain level
        assert env.phase == EnvelopePhase.SUSTAIN
        assert abs(env.value - int(0.5 * Q15_ONE)) < 100

    def test_sustain_holds(self):
        """Sustain phase holds level indefinitely."""
        env = ADSR(attack_ms=1, decay_ms=1, sustain=0.7, release_ms=100)
        env.trigger()

        # Run to sustain
        for _ in range(ms_to_blocks(10)):
            env.tick()

        # Check sustain holds
        sustain_level = int(0.7 * Q15_ONE)
        for _ in range(100):
            val = env.tick()
            assert abs(val - sustain_level) < 100

    def test_release_falls_to_zero(self):
        """Release phase falls to zero."""
        env = ADSR(attack_ms=1, decay_ms=1, sustain=0.5, release_ms=50)
        env.trigger()

        # Run to sustain
        for _ in range(ms_to_blocks(10)):
            env.tick()

        # Release
        env.release()
        assert env.is_releasing

        # Run through release
        for _ in range(ms_to_blocks(60)):
            env.tick()

        # Should be idle at zero
        assert not env.is_active
        assert env.tick() == 0

    def test_retrigger_from_current_level(self):
        """Retriggering starts from current level, not zero."""
        env = ADSR(attack_ms=10, decay_ms=100, sustain=0.5, release_ms=100)
        env.trigger()

        # Run partway through
        for _ in range(5):
            env.tick()

        level_before = env.value

        # Retrigger
        env.trigger()

        # Should continue from current level
        assert env.value == level_before

    def test_reset(self):
        """Reset returns envelope to idle."""
        env = ADSR()
        env.trigger()
        for _ in range(10):
            env.tick()

        env.reset()
        assert not env.is_active
        assert env.tick() == 0


class TestAR:
    """Test AR (attack-release) envelope."""

    def test_ar_basic_cycle(self):
        """AR envelope rises then falls."""
        env = AR(attack_ms=10, release_ms=20)
        env.trigger()

        # Collect values
        values = []
        while env.is_active:
            values.append(env.tick())
            if len(values) > 1000:
                break

        # Should have peaked at Q15_ONE
        assert max(values) == Q15_ONE

        # Should end at zero
        assert values[-1] == 0

    def test_ar_no_sustain(self):
        """AR immediately enters release after attack."""
        env = AR(attack_ms=5, release_ms=10)
        env.trigger()

        # Run through attack
        for _ in range(ms_to_blocks(6)):
            env.tick()

        # Should be in release, not sustain
        assert env._phase == EnvelopePhase.RELEASE


class TestDecay:
    """Test Decay (one-shot exponential) envelope."""

    def test_decay_starts_at_peak(self):
        """Decay starts at Q15_ONE."""
        env = Decay(decay_ms=100)
        env.trigger()
        assert env.value == Q15_ONE

    def test_decay_falls_exponentially(self):
        """Decay follows exponential curve."""
        env = Decay(decay_ms=100)
        env.trigger()

        values = []
        for _ in range(ms_to_blocks(300)):
            values.append(env.tick())

        # Should be monotonically decreasing
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

        # Should approach zero
        assert values[-1] < Q15_ONE // 10

    def test_decay_floor(self):
        """Decay respects floor level."""
        env = Decay(decay_ms=50, floor=0.1)
        env.trigger()

        # Run until done
        for _ in range(ms_to_blocks(500)):
            env.tick()

        # Should be at floor
        floor_q15 = int(0.1 * Q15_ONE)
        assert abs(env.value - floor_q15) < 100


class TestEnvelopeFollower:
    """Test EnvelopeFollower."""

    def test_follower_tracks_rising_input(self):
        """Follower rises when input increases."""
        follower = EnvelopeFollower(attack_ms=5, release_ms=100)

        # Start with zero
        for _ in range(10):
            follower.tick(0)

        # Apply high input
        for _ in range(ms_to_blocks(10)):
            follower.tick(Q15_ONE)

        # Should have risen
        assert follower.value > Q15_ONE // 2

    def test_follower_tracks_falling_input(self):
        """Follower falls when input decreases."""
        follower = EnvelopeFollower(attack_ms=1, release_ms=50)

        # Bring to peak
        for _ in range(10):
            follower.tick(Q15_ONE)

        # Apply zero input
        initial = follower.value
        for _ in range(ms_to_blocks(100)):
            follower.tick(0)

        # Should have fallen
        assert follower.value < initial // 2

    def test_follower_asymmetric(self):
        """Attack is faster than release."""
        follower = EnvelopeFollower(attack_ms=5, release_ms=100)

        # Rise
        rise_count = 0
        for _ in range(1000):
            follower.tick(Q15_ONE)
            rise_count += 1
            if follower.value > Q15_ONE * 0.9:
                break

        # Fall
        follower.tick(Q15_ONE)  # Ensure at peak
        fall_count = 0
        for _ in range(1000):
            follower.tick(0)
            fall_count += 1
            if follower.value < Q15_ONE * 0.1:
                break

        # Release should take longer than attack
        assert fall_count > rise_count

    def test_follower_handles_negative_input(self):
        """Follower rectifies negative input."""
        follower = EnvelopeFollower(attack_ms=1, release_ms=10)

        for _ in range(10):
            follower.tick(-Q15_ONE)

        # Should track magnitude
        assert follower.value > Q15_ONE // 2


class TestSlew:
    """Test Slew limiter."""

    def test_slew_limits_rise(self):
        """Slew limits rate of rise."""
        slew = Slew(rise_ms=50, fall_ms=50, mode="LINEAR")

        # Jump to target
        for _ in range(5):
            slew.tick(Q15_ONE)

        # Should not have reached target yet
        assert slew.value < Q15_ONE

    def test_slew_limits_fall(self):
        """Slew limits rate of fall."""
        slew = Slew(rise_ms=50, fall_ms=50, mode="LINEAR")

        # Set to high
        slew.set_immediate(Q15_ONE)

        # Jump to zero
        for _ in range(5):
            slew.tick(0)

        # Should not have reached zero yet
        assert slew.value > 0

    def test_slew_reaches_target(self):
        """Slew eventually reaches target."""
        slew = Slew(rise_ms=10, fall_ms=10, mode="LINEAR")

        for _ in range(ms_to_blocks(20)):
            slew.tick(Q15_ONE)

        assert slew.value == Q15_ONE

    def test_slew_immediate_set(self):
        """set_immediate bypasses slew."""
        slew = Slew(rise_ms=1000, fall_ms=1000)
        slew.set_immediate(12345)
        assert slew.value == 12345

    def test_slew_exponential_mode(self):
        """Exponential mode uses one-pole filter."""
        slew = Slew(rise_ms=50, fall_ms=50, mode="EXP")

        values = []
        for _ in range(ms_to_blocks(200)):  # Run longer for exponential settling
            values.append(slew.tick(Q15_ONE))

        # Should approach target (exponential takes ~3 time constants to reach 95%)
        assert values[-1] > Q15_ONE * 0.8


class TestLFO:
    """Test LFO oscillator."""

    def test_lfo_sine_range(self):
        """Sine LFO output is in valid range."""
        lfo = LFO(rate_hz=10.0, shape=LFOShape.SINE)

        for _ in range(1000):
            val = lfo.tick()
            assert -32768 <= val <= 32767

    def test_lfo_unipolar(self):
        """Unipolar mode outputs 0 to 32767."""
        lfo = LFO(rate_hz=10.0, shape=LFOShape.SINE, unipolar=True)

        for _ in range(1000):
            val = lfo.tick()
            assert 0 <= val <= 32767

    def test_lfo_rate(self):
        """LFO completes cycles at correct rate."""
        lfo = LFO(rate_hz=1.0, shape=LFOShape.SINE)

        # Run for exactly 750 blocks (1 second) and track phase
        initial_phase = lfo.phase
        for _ in range(750):
            lfo.tick()

        # After 1 second at 1 Hz, phase should have wrapped once
        # (be close to initial phase)
        final_phase = lfo.phase
        # Allow for some rounding error
        assert abs(final_phase - initial_phase) < 0.1 or abs(final_phase - initial_phase - 1.0) < 0.1

    def test_lfo_shapes(self):
        """All LFO shapes produce output."""
        for shape in LFOShape:
            lfo = LFO(rate_hz=5.0, shape=shape)
            values = [lfo.tick() for _ in range(200)]  # Run longer to ensure variation

            # Should produce varying output (S&H may need time to change)
            unique_vals = len(set(values))
            assert unique_vals > 1, f"{shape} produced constant output: {values[:10]}"

    def test_lfo_sync(self):
        """Sync resets phase."""
        lfo = LFO(rate_hz=10.0, shape=LFOShape.SAW_UP)

        # Run partway
        for _ in range(50):
            lfo.tick()

        # Sync to zero
        lfo.sync(0.0)
        assert lfo.phase < 0.1

    def test_lfo_phase_offset(self):
        """Phase offset applies at initialization."""
        lfo1 = LFO(rate_hz=1.0, shape=LFOShape.SINE, phase_offset=0.0)
        lfo2 = LFO(rate_hz=1.0, shape=LFOShape.SINE, phase_offset=0.5)

        val1 = lfo1.tick()
        val2 = lfo2.tick()

        # 180 degree offset should produce opposite values
        # (approximately, accounting for phase increment)
        assert abs(val1 + val2) < 1000

    def test_lfo_sample_hold(self):
        """Sample & hold produces stepped random values."""
        lfo = LFO(rate_hz=10.0, shape=LFOShape.SAMPLE_HOLD)

        # Collect values
        values = []
        for _ in range(100):
            values.append(lfo.tick())

        # Should have some variety
        assert len(set(values)) > 1

        # Values should hold between updates
        # (at 10 Hz, ~75 blocks per cycle at 750 Hz block rate)


class TestIntegration:
    """Integration tests for envelope suite."""

    def test_adsr_with_slew(self):
        """ADSR output through slew produces smooth result."""
        env = ADSR(attack_ms=10, decay_ms=50, sustain=0.5, release_ms=100)
        slew = Slew(rise_ms=2, fall_ms=2, mode="LINEAR")

        env.trigger()
        values = []
        for _ in range(ms_to_blocks(200)):
            raw = env.tick()
            smoothed = slew.tick(raw)
            values.append(smoothed)

        # Should be relatively smooth (linear slew limits step size)
        large_jumps = 0
        for i in range(1, len(values)):
            delta = abs(values[i] - values[i - 1])
            if delta > Q15_ONE // 5:
                large_jumps += 1

        # Allow very few large jumps (slew should smooth most)
        assert large_jumps < len(values) // 10

    def test_lfo_modulating_follower(self):
        """LFO can modulate envelope follower parameters."""
        lfo = LFO(rate_hz=2.0, shape=LFOShape.SINE, unipolar=True)
        follower = EnvelopeFollower(attack_ms=5, release_ms=100)

        # Simulate modulated following
        for _ in range(200):
            mod = lfo.tick()
            # Use LFO to modulate input amplitude
            input_level = (Q15_ONE * mod) >> 15
            follower.tick(input_level)

        # Follower should track LFO-modulated input
        assert 0 <= follower.value <= Q15_ONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
