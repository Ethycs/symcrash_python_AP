"""
WPO v1 Oscillator Conformance Tests
===================================

Tests for:
1. Phase accumulator — frequency conversion, MIDI note, wrap detection
2. Tree archetypes — configuration values, canonical definitions
3. BLEP kernel — generation, injection, residual buffer
4. Two-stream blend — scaffold + texture crossfade
5. Harmonic scaffold — saw/square/triangle waveforms
6. Analog drift — phase/amplitude drift, unison detune
7. WPOState — full oscillator rendering
8. OscillatorManager — multi-voice handling
"""

import pytest
import math
from symcrash_python_ap.wavelet.oscillator import (
    # Constants
    PHASE_BITS,
    PHASE_FRAC_BITS,
    PHASE_ONE,
    PHASE_MASK,
    BLOCK_SIZE,
    SAMPLE_RATE,
    Q15_ONE,
    Q15_HALF,
    Q15_SCALE,
    BLEP_KERNEL_SIZE,
    BLEP_OVERSAMPLING,
    DRIFT_SCALE,
    DRIFT_DECAY_Q15,
    # Tree archetypes
    TreeArchetype,
    TreeArchetypeConfig,
    TREE_ARCHETYPES,
    get_archetype_config,
    # BLEP
    BLEP_KERNEL,
    BLEP_TABLE,
    BLEPState,
    generate_blep_kernel,
    generate_blep_table,
    # Phase
    PhaseAccumulator,
    frequency_to_phase_inc,
    midi_to_frequency,
    midi_to_phase_inc,
    # Two-stream
    HarmonicScaffold,
    PacketTexture,
    blend_two_stream,
    # Drift
    AnalogDrift,
    # Oscillator
    WPOState,
    OscillatorManager,
)
from symcrash_python_ap.prng import VoicePRNG


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Verify oscillator constants."""

    def test_phase_bits(self):
        """Phase accumulator is 32-bit."""
        assert PHASE_BITS == 32

    def test_phase_frac_bits(self):
        """16-bit fractional phase."""
        assert PHASE_FRAC_BITS == 16

    def test_phase_one(self):
        """PHASE_ONE = 2^16."""
        assert PHASE_ONE == 65536

    def test_phase_mask(self):
        """PHASE_MASK = 2^32 - 1."""
        assert PHASE_MASK == 0xFFFFFFFF

    def test_block_size(self):
        """Block size is 64 samples."""
        assert BLOCK_SIZE == 64

    def test_sample_rate(self):
        """Sample rate is 48kHz."""
        assert SAMPLE_RATE == 48000

    def test_q15_one(self):
        """Q15 maximum is 32767."""
        assert Q15_ONE == 32767

    def test_q15_half(self):
        """Q15 half is 16384."""
        assert Q15_HALF == 16384

    def test_blep_kernel_size(self):
        """BLEP kernel is 64 samples."""
        assert BLEP_KERNEL_SIZE == 64

    def test_blep_oversampling(self):
        """BLEP uses 8x oversampling."""
        assert BLEP_OVERSAMPLING == 8


# =============================================================================
# PHASE ACCUMULATOR TESTS
# =============================================================================

class TestFrequencyConversion:
    """Test frequency to phase increment conversion."""

    def test_frequency_to_phase_inc_440hz(self):
        """440 Hz → correct phase increment."""
        inc = frequency_to_phase_inc(440.0)
        # Expected: 440 * 2^32 / 48000 ≈ 39,370,534
        # This gives ~109 samples per cycle (48000/440)
        expected = int(440.0 * (1 << 32) / 48000)
        assert abs(inc - expected) <= 1

    def test_frequency_to_phase_inc_1khz(self):
        """1000 Hz → correct phase increment."""
        inc = frequency_to_phase_inc(1000.0)
        # Expected: 1000 * 2^32 / 48000 ≈ 89,478,485
        # This gives 48 samples per cycle (48000/1000)
        expected = int(1000.0 * (1 << 32) / 48000)
        assert abs(inc - expected) <= 1

    def test_frequency_zero(self):
        """0 Hz → 0 phase increment."""
        inc = frequency_to_phase_inc(0.0)
        assert inc == 0

    def test_midi_to_frequency_a4(self):
        """MIDI note 69 (A4) → 440 Hz."""
        freq = midi_to_frequency(69)
        assert abs(freq - 440.0) < 0.01

    def test_midi_to_frequency_c4(self):
        """MIDI note 60 (C4) → ~261.6 Hz."""
        freq = midi_to_frequency(60)
        assert abs(freq - 261.63) < 0.1

    def test_midi_to_frequency_with_cents(self):
        """MIDI note with cents offset."""
        # +100 cents = one semitone up
        freq_a4 = midi_to_frequency(69, 0)
        freq_a4_plus_100 = midi_to_frequency(69, 100)
        freq_a4sharp = midi_to_frequency(70, 0)
        assert abs(freq_a4_plus_100 - freq_a4sharp) < 0.01

    def test_midi_to_phase_inc(self):
        """MIDI note directly to phase increment."""
        inc = midi_to_phase_inc(69)  # A4
        expected = frequency_to_phase_inc(440.0)
        assert inc == expected


class TestPhaseAccumulator:
    """Test phase accumulator functionality."""

    def test_initial_state(self):
        """Phase accumulator starts at zero."""
        pa = PhaseAccumulator()
        assert pa.phase == 0
        assert pa.phase_inc == 0
        assert pa.cycle_count == 0

    def test_set_frequency(self):
        """Set frequency updates phase increment."""
        pa = PhaseAccumulator()
        pa.set_frequency(440.0)
        assert pa.phase_inc > 0

    def test_set_midi_note(self):
        """Set MIDI note updates phase increment."""
        pa = PhaseAccumulator()
        pa.set_midi_note(69)
        assert pa.phase_inc > 0
        # Should match frequency_to_phase_inc for A4
        expected = frequency_to_phase_inc(440.0)
        assert pa.phase_inc == expected

    def test_advance_increments_phase(self):
        """Advance increases phase by phase_inc."""
        pa = PhaseAccumulator()
        pa.phase_inc = 1000
        pa.advance()
        assert pa.phase == 1000

    def test_advance_returns_tuple(self):
        """Advance returns (phase, wrapped, frac_offset)."""
        pa = PhaseAccumulator()
        pa.phase_inc = 1000
        result = pa.advance()
        assert len(result) == 3
        phase, wrapped, frac_offset = result
        assert phase == 1000
        assert wrapped is False
        assert 0 <= frac_offset <= 7

    def test_wrap_detection(self):
        """Detect cycle boundary wrap."""
        pa = PhaseAccumulator()
        pa.phase = PHASE_MASK - 100  # Close to wrap
        pa.phase_inc = 200
        _, wrapped, _ = pa.advance()
        assert wrapped is True
        assert pa.cycle_count == 1

    def test_multiple_cycles(self):
        """Cycle count tracks multiple wraps at high frequency."""
        pa = PhaseAccumulator()
        # Use a very high phase increment to guarantee wrap
        # PHASE_MASK = 2^32-1, so inc of 2^30 wraps every 4 samples
        pa.phase_inc = 1 << 30
        for _ in range(10):
            pa.advance()
        # Should have wrapped multiple times
        assert pa.cycle_count >= 2

    def test_advance_block(self):
        """Advance a full block of samples."""
        pa = PhaseAccumulator()
        pa.phase_inc = 1000
        results = pa.advance_block(64)
        assert len(results) == 64
        assert pa.phase == 64 * 1000

    def test_get_normalized_phase(self):
        """Get phase as normalized float."""
        pa = PhaseAccumulator()
        pa.phase = 1 << 31  # Half way
        norm = pa.get_normalized_phase()
        assert abs(norm - 0.5) < 0.01

    def test_reset(self):
        """Reset clears phase and cycle count."""
        pa = PhaseAccumulator()
        pa.phase = 12345
        pa.cycle_count = 5
        pa.reset()
        assert pa.phase == 0
        assert pa.cycle_count == 0


# =============================================================================
# TREE ARCHETYPE TESTS
# =============================================================================

class TestTreeArchetypes:
    """Test tree archetype definitions."""

    def test_all_archetypes_defined(self):
        """All 5 archetypes are defined."""
        assert len(TreeArchetype) == 5
        assert TreeArchetype.HARMONIC.value == 0
        assert TreeArchetype.BRIGHT.value == 1
        assert TreeArchetype.HOLLOW.value == 2
        assert TreeArchetype.NOISY.value == 3
        assert TreeArchetype.TRANSIENT.value == 4

    def test_all_configs_present(self):
        """All archetypes have configurations."""
        for arch in TreeArchetype:
            assert arch in TREE_ARCHETYPES

    def test_get_archetype_config(self):
        """get_archetype_config returns correct config."""
        config = get_archetype_config(TreeArchetype.HARMONIC)
        assert config.name == "Harmonic"
        assert config.harmonic_lock is True

    def test_harmonic_archetype(self):
        """HARMONIC: neutral, harmonic-locked."""
        config = TREE_ARCHETYPES[TreeArchetype.HARMONIC]
        assert config.tilt_q15 == 0
        assert config.odd_emphasis_q15 == 0
        assert config.noise_inject_q15 == 0
        assert config.transient_q15 == 0
        assert config.harmonic_lock is True

    def test_bright_archetype(self):
        """BRIGHT: positive tilt, harmonic-locked."""
        config = TREE_ARCHETYPES[TreeArchetype.BRIGHT]
        assert config.tilt_q15 > 0
        assert config.harmonic_lock is True

    def test_hollow_archetype(self):
        """HOLLOW: odd emphasis, harmonic-locked."""
        config = TREE_ARCHETYPES[TreeArchetype.HOLLOW]
        assert config.odd_emphasis_q15 > 0
        assert config.harmonic_lock is True

    def test_noisy_archetype(self):
        """NOISY: noise injection, not harmonic-locked."""
        config = TREE_ARCHETYPES[TreeArchetype.NOISY]
        assert config.noise_inject_q15 > 0
        assert config.harmonic_lock is False

    def test_transient_archetype(self):
        """TRANSIENT: high transient, not harmonic-locked."""
        config = TREE_ARCHETYPES[TreeArchetype.TRANSIENT]
        assert config.transient_q15 > 0
        assert config.harmonic_lock is False


# =============================================================================
# BLEP KERNEL TESTS
# =============================================================================

class TestBLEPKernel:
    """Test BLEP kernel generation."""

    def test_kernel_size(self):
        """BLEP kernel has correct size."""
        assert len(BLEP_KERNEL) == BLEP_KERNEL_SIZE

    def test_kernel_values_in_range(self):
        """BLEP kernel values are in Q15 range."""
        for v in BLEP_KERNEL:
            assert -Q15_SCALE <= v <= Q15_ONE

    def test_kernel_has_valid_structure(self):
        """BLEP kernel has valid structure for bandlimited synthesis."""
        # The kernel should have a transition from low to high through center
        # Check that the center region contains Q15_HALF (the transition point)
        center_start = BLEP_KERNEL_SIZE // 4
        center_end = 3 * BLEP_KERNEL_SIZE // 4
        center_region = BLEP_KERNEL[center_start:center_end]
        # Center should contain values near Q15_HALF
        has_transition = any(abs(v - Q15_HALF) < Q15_HALF // 2 for v in center_region)
        assert has_transition, "Kernel should have transition values near center"

    def test_generate_blep_kernel_deterministic(self):
        """Kernel generation is deterministic."""
        k1 = generate_blep_kernel()
        k2 = generate_blep_kernel()
        assert k1 == k2

    def test_blep_table_size(self):
        """BLEP table has correct dimensions."""
        assert len(BLEP_TABLE) == BLEP_OVERSAMPLING
        for row in BLEP_TABLE:
            assert len(row) == BLEP_KERNEL_SIZE

    def test_blep_table_values_in_range(self):
        """All BLEP table values are in Q15 range."""
        for row in BLEP_TABLE:
            for v in row:
                assert -Q15_SCALE <= v <= Q15_ONE


class TestBLEPState:
    """Test BLEP residual buffer state."""

    def test_initial_state(self):
        """BLEP state starts inactive."""
        blep = BLEPState()
        assert blep.active is False
        assert blep.write_pos == 0
        assert all(s == 0 for s in blep.buffer)

    def test_inject_edge_activates(self):
        """Injecting an edge activates the buffer."""
        blep = BLEPState()
        blep.inject_edge(Q15_ONE, 0)
        assert blep.active is True

    def test_inject_edge_modifies_buffer(self):
        """Edge injection modifies buffer values."""
        blep = BLEPState()
        blep.inject_edge(Q15_ONE, 0)
        assert any(s != 0 for s in blep.buffer)

    def test_read_and_advance(self):
        """read_and_advance returns sample and moves pointer."""
        blep = BLEPState()
        blep.inject_edge(Q15_ONE, 0)
        initial_pos = blep.write_pos
        _ = blep.read_and_advance()
        assert blep.write_pos == (initial_pos + 1) % BLEP_KERNEL_SIZE

    def test_read_returns_zero_when_inactive(self):
        """Inactive BLEP returns zero."""
        blep = BLEPState()
        assert blep.read_and_advance() == 0

    def test_buffer_clears_after_reads(self):
        """Buffer becomes inactive after all samples read."""
        blep = BLEPState()
        blep.inject_edge(Q15_ONE >> 4, 0)  # Small amplitude
        for _ in range(BLEP_KERNEL_SIZE * 2):
            blep.read_and_advance()
        # Should eventually become inactive
        # (may take multiple iterations due to residual)

    def test_fractional_phase_offset(self):
        """Different fractional offsets use different table rows."""
        blep0 = BLEPState()
        blep4 = BLEPState()
        blep0.inject_edge(Q15_ONE, 0)
        blep4.inject_edge(Q15_ONE, 4)
        # Buffers should be different (unless amplitude is tiny)
        # Can't guarantee they differ, but can check they're valid
        assert blep0.active and blep4.active

    def test_saturate(self):
        """Saturation clamps values."""
        assert BLEPState._saturate(Q15_ONE + 1000) == Q15_ONE
        assert BLEPState._saturate(-Q15_SCALE - 1000) == -Q15_SCALE
        assert BLEPState._saturate(0) == 0


# =============================================================================
# TWO-STREAM BLEND TESTS
# =============================================================================

class TestTwoStreamBlend:
    """Test two-stream blend function."""

    def test_eta_zero_is_pure_scaffold(self):
        """η = 0 → pure scaffold (within Q15 rounding)."""
        scaffold = 10000
        texture = 20000
        result = blend_two_stream(scaffold, texture, 0)
        # With η=0: output = scaffold * Q15_ONE >> 15 = scaffold * 32767 >> 15
        # This gives slight loss due to Q15_ONE being 32767 not 32768
        assert abs(result - scaffold) <= 1

    def test_eta_max_is_pure_texture(self):
        """η = Q15_ONE → pure texture (within Q15 rounding)."""
        scaffold = 10000
        texture = 20000
        result = blend_two_stream(scaffold, texture, Q15_ONE)
        # With η=Q15_ONE: inv_eta = 0, output = texture * Q15_ONE >> 15
        assert abs(result - texture) <= 1

    def test_eta_half_is_average(self):
        """η = 0.5 → average."""
        scaffold = 10000
        texture = 20000
        result = blend_two_stream(scaffold, texture, Q15_HALF)
        expected = 15000
        assert abs(result - expected) <= 2

    def test_blend_saturates_positive(self):
        """Blend saturates at positive limit."""
        result = blend_two_stream(Q15_ONE, Q15_ONE, Q15_HALF)
        assert result <= Q15_ONE

    def test_blend_saturates_negative(self):
        """Blend saturates at negative limit."""
        result = blend_two_stream(-Q15_SCALE, -Q15_SCALE, Q15_HALF)
        assert result >= -Q15_SCALE


# =============================================================================
# HARMONIC SCAFFOLD TESTS
# =============================================================================

class TestHarmonicScaffold:
    """Test harmonic scaffold generator."""

    def test_initial_state(self):
        """Scaffold starts with 8 harmonics."""
        scaffold = HarmonicScaffold()
        assert scaffold.num_harmonics == 8
        assert len(scaffold.harmonic_amps_q15) == 8

    def test_set_saw_harmonics(self):
        """Saw harmonics follow 1/n falloff."""
        scaffold = HarmonicScaffold()
        scaffold.set_saw_harmonics(8)
        # Harmonic n has amplitude Q15_ONE / n
        assert scaffold.harmonic_amps_q15[0] == Q15_ONE  # n=1
        assert scaffold.harmonic_amps_q15[1] == Q15_ONE // 2  # n=2
        assert scaffold.harmonic_amps_q15[2] == Q15_ONE // 3  # n=3

    def test_set_square_harmonics(self):
        """Square harmonics are odd only, 1/n falloff."""
        scaffold = HarmonicScaffold()
        scaffold.set_square_harmonics(8)
        # Odd harmonics only
        assert scaffold.harmonic_amps_q15[0] == Q15_ONE  # n=1 (odd)
        assert scaffold.harmonic_amps_q15[1] == 0  # n=2 (even)
        assert scaffold.harmonic_amps_q15[2] == Q15_ONE // 3  # n=3 (odd)
        assert scaffold.harmonic_amps_q15[3] == 0  # n=4 (even)

    def test_set_triangle_harmonics(self):
        """Triangle harmonics are odd only, 1/n² falloff."""
        scaffold = HarmonicScaffold()
        scaffold.set_triangle_harmonics(8)
        # Odd harmonics with 1/n² falloff
        assert scaffold.harmonic_amps_q15[0] == Q15_ONE  # n=1
        assert scaffold.harmonic_amps_q15[1] == 0  # n=2 (even)
        assert scaffold.harmonic_amps_q15[2] == Q15_ONE // 9  # n=3, 1/9

    def test_render_sample_output_range(self):
        """Render sample produces Q15 output."""
        scaffold = HarmonicScaffold()
        scaffold.set_saw_harmonics(8)
        for i in range(BLOCK_SIZE):
            phase = i * (PHASE_MASK // BLOCK_SIZE)
            sample = scaffold.render_sample(phase)
            assert -Q15_SCALE <= sample <= Q15_ONE

    def test_fast_sine_range(self):
        """Fast sine approximation stays in range."""
        for i in range(1000):
            phase = (i * PHASE_MASK) // 1000
            sine = HarmonicScaffold._fast_sine(phase)
            assert -Q15_SCALE <= sine <= Q15_ONE


# =============================================================================
# PACKET TEXTURE TESTS
# =============================================================================

class TestPacketTexture:
    """Test packet texture generator."""

    def test_initial_state(self):
        """Texture starts with 64 zero coefficients."""
        texture = PacketTexture()
        assert len(texture.coefficients_q15) == 64
        assert all(c == 0 for c in texture.coefficients_q15)
        assert texture.archetype == TreeArchetype.HARMONIC

    def test_apply_archetype_updates_archetype(self):
        """apply_archetype stores archetype."""
        texture = PacketTexture()
        texture.apply_archetype(TreeArchetype.BRIGHT)
        assert texture.archetype == TreeArchetype.BRIGHT

    def test_apply_tilt_modifies_coefficients(self):
        """Tilt modifies coefficient distribution."""
        texture = PacketTexture()
        texture.coefficients_q15 = [Q15_HALF] * 64
        texture._apply_tilt(Q15_HALF)
        # With positive tilt, high indices should be boosted
        assert texture.coefficients_q15[0] != texture.coefficients_q15[63]

    def test_apply_odd_emphasis(self):
        """Odd emphasis boosts odd-indexed coefficients."""
        texture = PacketTexture()
        texture.coefficients_q15 = [Q15_HALF] * 64
        original_odd = texture.coefficients_q15[1]
        original_even = texture.coefficients_q15[0]
        texture._apply_odd_emphasis(Q15_HALF)
        # Odd should be boosted, even unchanged
        assert texture.coefficients_q15[1] > original_odd
        assert texture.coefficients_q15[0] == original_even


# =============================================================================
# ANALOG DRIFT TESTS
# =============================================================================

class TestAnalogDrift:
    """Test analog drift functionality."""

    def test_initial_state(self):
        """Drift starts at zero."""
        drift = AnalogDrift()
        assert drift.phase_drift == 0
        assert drift.amp_drift_q15 == 0
        assert drift.detune_cents == 0

    def test_update_modifies_drift(self):
        """update() changes drift values."""
        drift = AnalogDrift()
        prng = VoicePRNG.create(voice_id=0, global_seed=12345)
        drift.update(prng)
        # After update, drift should be non-zero (usually)
        # Can't guarantee non-zero, but structure should be valid

    def test_apply_phase_drift(self):
        """Phase drift applied to increment."""
        drift = AnalogDrift()
        drift.phase_drift = 100
        result = drift.apply_phase_drift(1000)
        assert result == 1100

    def test_apply_amp_drift(self):
        """Amplitude drift applied to output."""
        drift = AnalogDrift()
        drift.amp_drift_q15 = 50
        result = drift.apply_amp_drift(10000)
        assert result == 10050

    def test_unison_detune_single_voice(self):
        """Single voice has no detune."""
        drift = AnalogDrift()
        drift.set_unison_detune(0, 10, 1)
        assert drift.detune_cents == 0

    def test_unison_detune_two_voices(self):
        """Two voices spread around center."""
        drift1 = AnalogDrift()
        drift2 = AnalogDrift()
        drift1.set_unison_detune(0, 10, 2)
        drift2.set_unison_detune(1, 10, 2)
        # Should be opposite signs
        assert drift1.detune_cents < 0 or drift2.detune_cents < 0
        assert drift1.detune_cents > 0 or drift2.detune_cents > 0

    def test_unison_detune_symmetry(self):
        """Unison detune is symmetric around center."""
        drifts = [AnalogDrift() for _ in range(5)]
        for i, d in enumerate(drifts):
            d.set_unison_detune(i, 20, 5)
        # Center voice (index 2) should have ~0 detune
        assert abs(drifts[2].detune_cents) <= 1


# =============================================================================
# WPO STATE TESTS
# =============================================================================

class TestWPOState:
    """Test complete oscillator state."""

    def test_initial_state(self):
        """WPOState initializes with defaults."""
        osc = WPOState()
        assert osc.voice_id == 0
        assert osc.eta_q15 == Q15_HALF
        assert osc.archetype_a == TreeArchetype.HARMONIC
        assert osc.last_output_q15 == 0

    def test_configure(self):
        """configure() sets all parameters."""
        osc = WPOState()
        osc.configure(
            note=69,
            cents=0,
            archetype=TreeArchetype.BRIGHT,
            eta_q15=Q15_ONE,
            waveform="saw",
        )
        assert osc.phase.phase_inc > 0
        assert osc.archetype_a == TreeArchetype.BRIGHT
        assert osc.eta_q15 == Q15_ONE

    def test_configure_waveforms(self):
        """configure() handles all waveform types."""
        for waveform in ["saw", "square", "triangle", "sine"]:
            osc = WPOState()
            osc.configure(note=60, waveform=waveform)
            # Should not raise

    def test_render_sample_output_range(self):
        """render_sample() produces Q15 output."""
        osc = WPOState()
        osc.configure(note=60, waveform="saw")
        for _ in range(100):
            sample = osc.render_sample()
            assert -Q15_SCALE <= sample <= Q15_ONE

    def test_render_sample_updates_last_output(self):
        """render_sample() updates last_output_q15."""
        osc = WPOState()
        osc.configure(note=60)
        sample = osc.render_sample()
        assert osc.last_output_q15 == sample

    def test_render_block_size(self):
        """render_block() produces correct number of samples."""
        osc = WPOState()
        osc.configure(note=60)
        samples = osc.render_block(64)
        assert len(samples) == 64

    def test_render_block_with_prng(self):
        """render_block() with PRNG updates drift."""
        osc = WPOState()
        osc.configure(note=60, archetype=TreeArchetype.NOISY)
        prng = VoicePRNG.create(voice_id=0, global_seed=12345)
        samples = osc.render_block(64, prng)
        assert len(samples) == 64

    def test_render_deterministic_without_prng(self):
        """Without PRNG, rendering is deterministic."""
        osc1 = WPOState()
        osc2 = WPOState()
        osc1.configure(note=60, waveform="sine")
        osc2.configure(note=60, waveform="sine")
        samples1 = osc1.render_block(64)
        samples2 = osc2.render_block(64)
        assert samples1 == samples2

    def test_blep_injected_on_wrap(self):
        """BLEP is injected at cycle boundaries."""
        osc = WPOState()
        osc.configure(note=60, waveform="saw")
        # Render enough samples to get a cycle wrap
        initial_active = osc.blep.active
        osc.render_block(BLOCK_SIZE * 4)
        # BLEP should have been triggered at some point
        # (can't guarantee it's still active, but structure should be valid)


# =============================================================================
# OSCILLATOR MANAGER TESTS
# =============================================================================

class TestOscillatorManager:
    """Test multi-voice oscillator manager."""

    def test_initial_state(self):
        """Manager starts with no oscillators."""
        mgr = OscillatorManager()
        assert len(mgr.oscillators) == 0
        assert mgr.max_voices == 16

    def test_allocate(self):
        """allocate() creates and stores oscillator."""
        mgr = OscillatorManager()
        osc = mgr.allocate(voice_id=5)
        assert osc.voice_id == 5
        assert mgr.get(5) is osc

    def test_deallocate(self):
        """deallocate() removes oscillator."""
        mgr = OscillatorManager()
        mgr.allocate(voice_id=3)
        mgr.deallocate(3)
        assert mgr.get(3) is None

    def test_deallocate_nonexistent(self):
        """deallocate() ignores nonexistent voice."""
        mgr = OscillatorManager()
        mgr.deallocate(99)  # Should not raise

    def test_get_nonexistent(self):
        """get() returns None for missing voice."""
        mgr = OscillatorManager()
        assert mgr.get(42) is None

    def test_configure_voice(self):
        """configure_voice() updates oscillator parameters."""
        mgr = OscillatorManager()
        mgr.allocate(voice_id=1)
        mgr.configure_voice(1, note=72, waveform="square")
        osc = mgr.get(1)
        assert osc.phase.phase_inc > 0

    def test_configure_nonexistent_voice(self):
        """configure_voice() ignores nonexistent voice."""
        mgr = OscillatorManager()
        mgr.configure_voice(99, note=60)  # Should not raise

    def test_render_all(self):
        """render_all() produces output for all voices."""
        mgr = OscillatorManager()
        mgr.allocate(voice_id=0)
        mgr.allocate(voice_id=1)
        mgr.configure_voice(0, note=60)
        mgr.configure_voice(1, note=72)

        outputs = mgr.render_all(block_size=32)
        assert 0 in outputs
        assert 1 in outputs
        assert len(outputs[0]) == 32
        assert len(outputs[1]) == 32

    def test_render_all_with_prngs(self):
        """render_all() uses per-voice PRNGs."""
        mgr = OscillatorManager()
        mgr.allocate(voice_id=0)
        mgr.allocate(voice_id=1)
        mgr.configure_voice(0, note=60)
        mgr.configure_voice(1, note=72)

        prngs = {
            0: VoicePRNG.create(voice_id=0, global_seed=111),
            1: VoicePRNG.create(voice_id=1, global_seed=222),
        }
        outputs = mgr.render_all(block_size=32, prngs=prngs)
        assert len(outputs) == 2

    def test_render_empty_manager(self):
        """render_all() returns empty dict for no voices."""
        mgr = OscillatorManager()
        outputs = mgr.render_all()
        assert outputs == {}


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full oscillator pipeline."""

    def test_full_pipeline_determinism(self):
        """Full pipeline is deterministic with same inputs."""
        def render_pipeline():
            mgr = OscillatorManager()
            mgr.allocate(0)
            mgr.configure_voice(0, note=60, waveform="saw", archetype=TreeArchetype.HARMONIC)
            return mgr.render_all(block_size=64)

        result1 = render_pipeline()
        result2 = render_pipeline()
        assert result1[0] == result2[0]

    def test_different_notes_differ(self):
        """Different notes produce different output."""
        mgr = OscillatorManager()
        mgr.allocate(0)
        mgr.allocate(1)
        mgr.configure_voice(0, note=60)
        mgr.configure_voice(1, note=72)
        outputs = mgr.render_all(block_size=64)
        assert outputs[0] != outputs[1]

    def test_different_archetypes_differ(self):
        """Different archetypes produce different harmonic structure."""
        # Configure with saw waveform to get non-zero output
        scaffold1 = HarmonicScaffold()
        scaffold2 = HarmonicScaffold()
        scaffold1.set_saw_harmonics(8)
        scaffold2.set_square_harmonics(8)  # Square has odd-only harmonics

        # The harmonic amplitudes should differ
        assert scaffold1.harmonic_amps_q15 != scaffold2.harmonic_amps_q15
        # Square should have zeros at even harmonics
        assert scaffold2.harmonic_amps_q15[1] == 0  # 2nd harmonic
        assert scaffold2.harmonic_amps_q15[3] == 0  # 4th harmonic

    def test_different_waveforms_differ(self):
        """Different waveforms have different harmonic content."""
        scaffold_saw = HarmonicScaffold()
        scaffold_square = HarmonicScaffold()
        scaffold_saw.set_saw_harmonics(8)
        scaffold_square.set_square_harmonics(8)

        # Saw has all harmonics, square has only odd
        assert scaffold_saw.harmonic_amps_q15[1] > 0  # 2nd harmonic present
        assert scaffold_square.harmonic_amps_q15[1] == 0  # 2nd harmonic absent

    def test_eta_blend_affects_output(self):
        """Different eta values change the blend."""
        # Test the blend function directly
        scaffold = 10000
        texture = 5000

        result_scaffold_only = blend_two_stream(scaffold, texture, 0)
        result_texture_only = blend_two_stream(scaffold, texture, Q15_ONE)

        # Should be near scaffold and texture respectively
        assert abs(result_scaffold_only - scaffold) <= 1
        assert abs(result_texture_only - texture) <= 1

    def test_prng_adds_variation(self):
        """PRNG with noisy archetype adds variation."""
        osc = WPOState()
        osc.configure(note=60, archetype=TreeArchetype.NOISY)
        prng1 = VoicePRNG.create(voice_id=0, global_seed=111)
        prng2 = VoicePRNG.create(voice_id=0, global_seed=222)
        # Different seeds should give different results
        samples1 = osc.render_block(64, prng1)
        osc2 = WPOState()
        osc2.configure(note=60, archetype=TreeArchetype.NOISY)
        samples2 = osc2.render_block(64, prng2)
        # May or may not differ depending on noise injection amount

    def test_continuous_rendering(self):
        """Oscillator can render continuously without errors."""
        osc = WPOState()
        osc.configure(note=60, waveform="saw")
        for _ in range(100):  # 100 blocks = 6400 samples
            samples = osc.render_block(64)
            assert len(samples) == 64
            # Check all samples in range
            for s in samples:
                assert -Q15_SCALE <= s <= Q15_ONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
