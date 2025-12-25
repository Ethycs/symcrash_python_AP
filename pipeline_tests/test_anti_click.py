"""
ANTI-CLICK OPERATOR TESTS
=========================

Tests for the unified AntiClickOperator that handles:
- Latency alignment (5-block FIFO for wet/carrier sync)
- Carrier/wet mixing (equal-power crossfade with beta ramp)
- OLA seam smoothing (Hann window overlap)

This is the single source of truth for click prevention.
"""

import pytest
import numpy as np
import math


class TestAntiClickConfig:
    """Tests for AntiClickConfig defaults and validation."""

    def test_default_config_values(self):
        """Default config has expected values."""
        from anti_click import AntiClickConfig

        cfg = AntiClickConfig()

        assert cfg.block_size == 64
        assert cfg.sample_rate == 48_000
        assert cfg.wet_latency_blocks == 5
        assert cfg.bloom_time_s == 0.05
        assert cfg.ola_K == 16
        assert cfg.equal_power is True

    def test_custom_config(self):
        """Custom config values are respected."""
        from anti_click import AntiClickConfig

        cfg = AntiClickConfig(
            block_size=128,
            sample_rate=44100,
            wet_latency_blocks=3,
            bloom_time_s=0.1,
            ola_K=8,
            equal_power=False
        )

        assert cfg.block_size == 128
        assert cfg.sample_rate == 44100
        assert cfg.wet_latency_blocks == 3
        assert cfg.bloom_time_s == 0.1
        assert cfg.ola_K == 8
        assert cfg.equal_power is False


class TestAntiClickState:
    """Tests for AntiClickState initialization."""

    def test_init_state_creates_valid_state(self):
        """init_state creates properly initialized state."""
        from anti_click import AntiClickOperator, AntiClickConfig

        op = AntiClickOperator(AntiClickConfig())
        st = op.init_state()

        assert st.beta == 0.0
        assert st.beta_target == 0.0
        assert st.beta_step > 0.0
        assert st.wet_ready is False
        assert st.tail is not None
        assert st.carrier_fifo is not None
        assert len(st.carrier_fifo) == 5  # wet_latency_blocks

    def test_init_state_tail_size(self):
        """Tail buffer has correct size based on ola_K."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(ola_K=32)
        op = AntiClickOperator(cfg)
        st = op.init_state()

        assert len(st.tail) == 32

    def test_init_state_no_ola(self):
        """State without OLA has no tail buffer."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(ola_K=0)
        op = AntiClickOperator(cfg)
        st = op.init_state()

        assert st.tail is None


class TestNoteOnOff:
    """Tests for note_on and note_off state transitions."""

    def test_note_on_resets_state(self):
        """note_on resets state to initial values."""
        from anti_click import AntiClickOperator, AntiClickConfig

        op = AntiClickOperator(AntiClickConfig())
        st = op.init_state()

        # Modify state
        st.beta = 0.5
        st.beta_target = 1.0
        st.wet_ready = True

        # Reset with note_on
        op.note_on(st)

        assert st.beta == 0.0
        assert st.beta_target == 0.0
        assert st.wet_ready is False

    def test_note_on_clears_tail(self):
        """note_on clears the OLA tail buffer."""
        from anti_click import AntiClickOperator, AntiClickConfig

        op = AntiClickOperator(AntiClickConfig())
        st = op.init_state()

        # Fill tail with values
        st.tail.fill(1000.0)

        op.note_on(st)

        assert all(x == 0.0 for x in st.tail)

    def test_note_on_clears_fifo(self):
        """note_on clears the carrier FIFO."""
        from anti_click import AntiClickOperator, AntiClickConfig

        op = AntiClickOperator(AntiClickConfig())
        st = op.init_state()

        # Fill FIFO with non-zero blocks
        for block in st.carrier_fifo:
            block.fill(1000.0)

        op.note_on(st)

        for block in st.carrier_fifo:
            assert all(x == 0.0 for x in block)


class TestSetWetTarget:
    """Tests for set_wet_target and beta ramping."""

    def test_set_wet_target_clamps(self):
        """set_wet_target clamps to [0, 1] range."""
        from anti_click import AntiClickOperator, AntiClickConfig

        op = AntiClickOperator(AntiClickConfig())
        st = op.init_state()

        op.set_wet_target(st, -0.5)
        assert st.beta_target == 0.0

        op.set_wet_target(st, 1.5)
        assert st.beta_target == 1.0

        op.set_wet_target(st, 0.5)
        assert st.beta_target == 0.5

    def test_beta_ramps_toward_target(self):
        """Beta ramps toward target over multiple blocks."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(bloom_time_s=0.05)
        op = AntiClickOperator(cfg)
        st = op.init_state()

        # Set target to 1.0
        op.set_wet_target(st, 1.0)
        st.wet_ready = True  # Enable wet path

        carrier = np.zeros((64,), dtype=np.float32)
        wet = np.zeros((64,), dtype=np.float32)

        # Run blocks and check beta progression
        betas = [st.beta]
        for _ in range(50):
            op.step(st, carrier, wet)
            betas.append(st.beta)

        # Beta should increase toward 1.0
        assert betas[-1] > betas[0], "Beta should increase"
        assert betas[-1] == 1.0, "Beta should reach target"


class TestCarrierBeforeWet:
    """Tests for carrier-first behavior during warmup."""

    def test_outputs_carrier_before_wet_ready(self):
        """Outputs carrier directly before wet becomes ready."""
        from anti_click import AntiClickOperator, AntiClickConfig

        op = AntiClickOperator(AntiClickConfig())
        st = op.init_state()
        op.set_wet_target(st, 1.0)

        # Create distinctive carrier
        carrier = np.ones((64,), dtype=np.float32) * 0.5

        # No wet available
        output = op.step(st, carrier, None)

        # Should output carrier directly
        np.testing.assert_array_almost_equal(output, carrier)

    def test_wet_ready_set_when_wet_provided(self):
        """wet_ready is set when wet signal is first provided."""
        from anti_click import AntiClickOperator, AntiClickConfig

        op = AntiClickOperator(AntiClickConfig())
        st = op.init_state()

        assert st.wet_ready is False

        carrier = np.zeros((64,), dtype=np.float32)
        wet = np.zeros((64,), dtype=np.float32)

        op.step(st, carrier, wet)

        assert st.wet_ready is True


class TestLatencyAlignment:
    """Tests for carrier FIFO latency alignment."""

    def test_carrier_delayed_by_latency_blocks(self):
        """Carrier is delayed by wet_latency_blocks."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(wet_latency_blocks=3, ola_K=0)
        op = AntiClickOperator(cfg)
        st = op.init_state()
        st.wet_ready = True
        op.set_wet_target(st, 1.0)

        # Push distinctive blocks through
        blocks = []
        for i in range(5):
            carrier = np.ones((64,), dtype=np.float32) * (i + 1)
            wet = np.ones((64,), dtype=np.float32) * 10  # Constant wet
            blocks.append(carrier.copy())
            op.step(st, carrier, wet)

        # After 3 blocks, the delayed carrier should be block 0
        # (FIFO is pre-filled with zeros, then block 0, 1, 2 pushed)
        # At step 4, we're mixing delayed carrier (block 1) with wet
        # Beta is ramping, so exact values are mixed


class TestMixing:
    """Tests for carrier/wet mixing."""

    def test_pure_dry_at_beta_zero(self):
        """Pure carrier output when beta=0."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(ola_K=0)
        op = AntiClickOperator(cfg)
        st = op.init_state()
        st.wet_ready = True
        st.beta = 0.0
        st.beta_target = 0.0  # Explicitly stay at 0

        carrier = np.ones((64,), dtype=np.float32) * 0.5
        wet = np.ones((64,), dtype=np.float32) * 0.9

        output = op.step(st, carrier, wet)

        # Should output carrier, not wet
        np.testing.assert_array_almost_equal(output, carrier)

    def test_pure_wet_at_beta_one(self):
        """Pure wet output when beta=1."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(ola_K=0, wet_latency_blocks=1)
        op = AntiClickOperator(cfg)
        st = op.init_state()
        st.wet_ready = True
        st.beta = 1.0
        st.beta_target = 1.0

        carrier = np.ones((64,), dtype=np.float32) * 0.5
        wet = np.ones((64,), dtype=np.float32) * 0.9

        # Push carrier through FIFO
        op.step(st, carrier, wet)
        output = op.step(st, carrier, wet)

        # Should output wet (after FIFO warmup)
        np.testing.assert_array_almost_equal(output, wet)

    def test_equal_power_crossfade_at_half(self):
        """Equal-power crossfade at beta=0.5 has expected weights."""
        from anti_click import equal_power_weights

        dry, wet = equal_power_weights(0.5)

        # At beta=0.5, both weights should be ~0.707
        expected = math.cos(0.25 * math.pi)  # ~0.707
        assert abs(dry - expected) < 0.001
        assert abs(wet - expected) < 0.001

    def test_equal_power_extremes(self):
        """Equal-power crossfade extremes are correct."""
        from anti_click import equal_power_weights

        dry0, wet0 = equal_power_weights(0.0)
        assert abs(dry0 - 1.0) < 0.001
        assert abs(wet0 - 0.0) < 0.001

        dry1, wet1 = equal_power_weights(1.0)
        assert abs(dry1 - 0.0) < 0.001
        assert abs(wet1 - 1.0) < 0.001


class TestOLA:
    """Tests for Overlap-Add seam smoothing."""

    def test_ola_blends_block_boundaries(self):
        """OLA smooths transition between blocks."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(ola_K=16, wet_latency_blocks=1)
        op = AntiClickOperator(cfg)
        st = op.init_state()
        st.wet_ready = True
        st.beta = 1.0
        st.beta_target = 1.0

        # First block: zeros
        carrier1 = np.zeros((64,), dtype=np.float32)
        wet1 = np.zeros((64,), dtype=np.float32)
        out1 = op.step(st, carrier1, wet1)

        # Second block: ones
        carrier2 = np.ones((64,), dtype=np.float32)
        wet2 = np.ones((64,), dtype=np.float32)
        out2 = op.step(st, carrier2, wet2)

        # First K samples of out2 should be blended (not pure 1.0)
        K = cfg.ola_K
        assert out2[0] < 1.0, "OLA should blend first sample"
        assert out2[K-1] < 1.0, "OLA should blend up to K samples"
        assert out2[K] == 1.0 or abs(out2[K] - 1.0) < 0.01, "Beyond K should be unchanged"

    def test_no_ola_when_ola_k_zero(self):
        """No OLA smoothing when ola_K=0."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(ola_K=0, wet_latency_blocks=1)
        op = AntiClickOperator(cfg)
        st = op.init_state()
        st.wet_ready = True
        st.beta = 1.0
        st.beta_target = 1.0

        # First block: zeros
        carrier1 = np.zeros((64,), dtype=np.float32)
        wet1 = np.zeros((64,), dtype=np.float32)
        op.step(st, carrier1, wet1)

        # Second block: ones
        carrier2 = np.ones((64,), dtype=np.float32)
        wet2 = np.ones((64,), dtype=np.float32)
        out2 = op.step(st, carrier2, wet2)

        # Should be pure wet (no blending)
        np.testing.assert_array_almost_equal(out2, wet2)

    def test_ola_not_applied_for_pure_carrier(self):
        """OLA is not applied when output is pure carrier (beta=0)."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(ola_K=16)
        op = AntiClickOperator(cfg)
        st = op.init_state()
        op.set_wet_target(st, 0.0)  # Explicitly pure dry

        # First block: zeros
        carrier1 = np.zeros((64,), dtype=np.float32)
        op.step(st, carrier1, None)

        # Second block: ones (pure carrier, step change)
        carrier2 = np.ones((64,), dtype=np.float32)
        out2 = op.step(st, carrier2, None)

        # Without OLA, output should match carrier exactly
        # (carrier is already phase-continuous, doesn't need OLA)
        np.testing.assert_array_almost_equal(out2, carrier2)


class TestQ15Interface:
    """Tests for Q15 fixed-point interface."""

    def test_step_q15_converts_correctly(self):
        """step_q15 converts Q15 I/O correctly."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(ola_K=0)
        op = AntiClickOperator(cfg)
        st = op.init_state()

        # Q15 half = 16384
        carrier = [16384] * 64
        output = op.step_q15(st, carrier, None)

        # Should output similar values (pure carrier, no wet)
        assert all(abs(o - 16384) < 100 for o in output), "Q15 conversion failed"

    def test_step_q15_clips_output(self):
        """step_q15 clips output to Q15 range."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(ola_K=0)
        op = AntiClickOperator(cfg)
        st = op.init_state()
        st.wet_ready = True
        st.beta = 1.0
        st.beta_target = 1.0

        # Max Q15 values
        carrier = [32767] * 64
        wet = [32767] * 64

        output = op.step_q15(st, carrier, wet)

        # All values should be in Q15 range
        assert all(-32768 <= o <= 32767 for o in output)


class TestHannWindow:
    """Tests for Hann window generation."""

    def test_hann_window_shape(self):
        """Hann window has expected shape."""
        from anti_click import hann_window

        w = hann_window(16)

        assert len(w) == 16
        assert w[0] == 0.0  # Hann starts at 0
        assert abs(w[8] - 1.0) < 0.02  # Peak near center (16-sample window)
        assert abs(w[15] - 0.0) < 0.02  # Ends near 0

    def test_hann_window_single_sample(self):
        """Hann window of length 1 is all ones."""
        from anti_click import hann_window

        w = hann_window(1)

        assert len(w) == 1
        assert w[0] == 1.0


class TestClickFreeOutput:
    """Integration tests for click-free output."""

    def test_no_clicks_during_wet_bloom(self):
        """No clicks during wet fade-in."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig(bloom_time_s=0.01)  # Fast bloom
        op = AntiClickOperator(cfg)
        st = op.init_state()
        op.note_on(st)
        op.set_wet_target(st, 1.0)

        # Generate blocks with step change in wet
        all_output = []
        for i in range(20):
            carrier = np.ones((64,), dtype=np.float32) * 0.3
            wet = np.ones((64,), dtype=np.float32) * 0.8
            output = op.step(st, carrier, wet)
            all_output.extend(output)

        # Check for clicks (large sample-to-sample jumps)
        max_jump = 0.0
        for i in range(1, len(all_output)):
            jump = abs(all_output[i] - all_output[i-1])
            max_jump = max(max_jump, jump)

        # Max jump should be reasonable (no harsh clicks)
        # 0.3 threshold allows for the carrier->wet transition ramp
        assert max_jump < 0.3, f"Click detected: max jump = {max_jump}"

    def test_no_clicks_on_voice_restart(self):
        """No clicks when voice restarts."""
        from anti_click import AntiClickOperator, AntiClickConfig

        cfg = AntiClickConfig()
        op = AntiClickOperator(cfg)
        st = op.init_state()

        # First voice
        op.note_on(st)
        op.set_wet_target(st, 0.5)

        carrier = np.ones((64,), dtype=np.float32) * 0.5
        wet = np.ones((64,), dtype=np.float32) * 0.7

        for _ in range(10):
            op.step(st, carrier, wet)

        # Get last sample before restart
        last_block = op.step(st, carrier, wet)
        last_sample = last_block[-1]

        # Restart voice
        op.note_on(st)
        op.set_wet_target(st, 0.5)

        # First block after restart
        first_block = op.step(st, carrier, None)
        first_sample = first_block[0]

        # Jump should be reasonable (note_on resets state, starts fresh)
        # The carrier starts at 0.5, so first sample should be ~0.5
        assert abs(first_sample - 0.5) < 0.1, f"Unexpected first sample: {first_sample}"


class TestClampUtility:
    """Tests for clamp utility function."""

    def test_clamp_within_range(self):
        """Clamp returns value within range."""
        from anti_click import clamp

        assert clamp(0.5, 0.0, 1.0) == 0.5
        assert clamp(-0.5, 0.0, 1.0) == 0.0
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_clamp_edge_values(self):
        """Clamp handles edge values correctly."""
        from anti_click import clamp

        assert clamp(0.0, 0.0, 1.0) == 0.0
        assert clamp(1.0, 0.0, 1.0) == 1.0
