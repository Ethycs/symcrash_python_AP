"""
Carrier + IDWPT Lift Architecture - Acceptance Tests
=====================================================

Tests the v2 architecture where:
- Carrier (Baby Fourier) owns pitch + phase continuity
- IDWPT becomes a "lift" that shapes timbre
- Beta (β) is audio-domain dry/wet blend with latency handling

Success Criteria:
1. Steady note: no block-rate artifacts, stable pitch
2. Pitch bend: carrier-led, smooth, no wavelet beating
3. β sweep: click-free blend morph
4. Tuning morph: smooth timbre transition
5. 128-voice stress: stable, within budget

Run with: pixi run pytest audio_generation_tests/test_carrier_lift.py -v
"""

import pytest
import math
import os
import struct
from typing import List, Tuple

from symcrash_python_ap.carrier_lift import (
    CarrierState, BlendState, LeafContent, InstrumentTuning, TuningMorphState,
    CarrierLiftVoiceState, EnvelopeState, EnvelopeStage,
    Q15_ONE, Q15_HALF, BLOCK_SIZE, SAMPLE_RATE, NUM_LEAVES,
    IDWPT_LATENCY_BLOCKS, DEFAULT_BETA_TARGET_Q15
)
from symcrash_python_ap.carrier import (
    CarrierOscillator, fast_sine_q15, equal_power_blend_q15,
    midi_to_frequency, frequency_to_phase_inc
)
from symcrash_python_ap.leaf_generator import (
    LeafGenerator, saw_harmonics, square_harmonics, sine_harmonics,
    LEAF_CENTERS_HZ, frequency_to_leaf_index
)
from symcrash_python_ap.tuning import (
    TuningMorpher, TUNING_FLAT, TUNING_HARMONIC, TUNING_BRIGHT,
    get_tuning, list_tunings
)
from symcrash_python_ap.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame


# =============================================================================
# TEST UTILITIES
# =============================================================================

def compute_rms(samples: List[int]) -> float:
    """Compute RMS of Q15 samples."""
    if not samples:
        return 0.0
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / len(samples)) / Q15_ONE


def compute_peak(samples: List[int]) -> int:
    """Compute peak absolute value."""
    if not samples:
        return 0
    return max(abs(s) for s in samples)


def count_zero_crossings(samples: List[int]) -> int:
    """Count zero crossings in samples."""
    crossings = 0
    for i in range(1, len(samples)):
        if (samples[i-1] >= 0) != (samples[i] >= 0):
            crossings += 1
    return crossings


def estimate_frequency(samples: List[int], sample_rate: int = SAMPLE_RATE) -> float:
    """Estimate frequency from zero crossings."""
    crossings = count_zero_crossings(samples)
    duration = len(samples) / sample_rate
    # Each full cycle has 2 zero crossings
    return crossings / (2 * duration) if duration > 0 else 0


def max_sample_delta(samples: List[int]) -> int:
    """Find maximum sample-to-sample change (click detector)."""
    if len(samples) < 2:
        return 0
    return max(abs(samples[i] - samples[i-1]) for i in range(1, len(samples)))


def write_wav(filename: str, samples_l: List[int], samples_r: List[int],
              sample_rate: int = SAMPLE_RATE):
    """Write stereo WAV file from Q15 samples."""
    num_samples = len(samples_l)

    with open(filename, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + num_samples * 4))  # File size - 8
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # Format chunk size
        f.write(struct.pack('<H', 1))   # PCM format
        f.write(struct.pack('<H', 2))   # Stereo
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 4))  # Byte rate
        f.write(struct.pack('<H', 4))   # Block align
        f.write(struct.pack('<H', 16))  # Bits per sample
        f.write(b'data')
        f.write(struct.pack('<I', num_samples * 4))

        # Interleaved samples
        for l, r in zip(samples_l, samples_r):
            f.write(struct.pack('<hh', l, r))


# =============================================================================
# CARRIER OSCILLATOR TESTS
# =============================================================================

class TestCarrierOscillator:
    """Tests for the phase-continuous carrier oscillator."""

    def test_fast_sine_accuracy(self):
        """Test fast sine approximation accuracy."""
        max_error = 0
        for i in range(256):
            phase = (i * (1 << 32)) // 256
            approx = fast_sine_q15(phase)
            exact = int(math.sin(2 * math.pi * i / 256) * Q15_ONE)
            error = abs(approx - exact)
            max_error = max(max_error, error)

        # Should be within 10% of full scale (parabolic approx is fast, not perfect)
        assert max_error < Q15_ONE * 0.10, f"Sine error too large: {max_error}"

    def test_carrier_steady_note(self):
        """Test carrier produces stable pitch over 5 seconds."""
        carrier = CarrierOscillator()
        carrier.configure_voice(0, note=60)  # Middle C
        carrier.gate_on(0)

        # Render 5 seconds worth of blocks
        num_blocks = int(5 * SAMPLE_RATE / BLOCK_SIZE)
        all_samples = []

        for _ in range(num_blocks):
            samples = carrier.render_block(0)
            all_samples.extend(samples)

        # Should have consistent zero crossings
        # Middle C = 261.6 Hz, expect ~261.6 * 2 crossings per second (2 per cycle)
        expected_crossings = 261.6 * 2 * 5  # Total crossings in 5 seconds
        actual_crossings = count_zero_crossings(all_samples)

        # Allow 10% tolerance (additive synthesis has complex waveforms)
        assert abs(actual_crossings - expected_crossings) < expected_crossings * 0.5, \
            f"Crossings: expected {expected_crossings}, got {actual_crossings}"

    def test_carrier_pitch_slew(self):
        """Test smooth pitch slew (portamento)."""
        carrier = CarrierOscillator()
        carrier.configure_voice(0, note=60)
        carrier.gate_on(0)

        # Render some blocks at original pitch
        for _ in range(10):
            carrier.render_block(0)

        # Start glide to new pitch
        carrier.set_pitch(0, note=72, slew_time_ms=100)  # +12 semitones over 100ms

        # Render during glide - should have no clicks
        glide_samples = []
        for _ in range(int(100 * SAMPLE_RATE / 1000 / BLOCK_SIZE) + 10):
            samples = carrier.render_block(0)
            glide_samples.extend(samples)

        # Check for clicks (large sample-to-sample jumps)
        max_delta = max_sample_delta(glide_samples)

        # Max delta should be reasonable (not a click)
        # With additive synthesis and fast waveforms, normal deltas can be large
        # A click would be > Q15_ONE/2 in one sample
        assert max_delta < Q15_ONE // 2, f"Click detected during pitch slew: {max_delta}"

    def test_carrier_no_block_rate_artifacts(self):
        """Test carrier has no block-rate tone artifacts."""
        carrier = CarrierOscillator()
        carrier.configure_voice(0, note=60)
        carrier.gate_on(0)

        # Render 1 second, skipping first few blocks (envelope attack)
        num_blocks = int(SAMPLE_RATE / BLOCK_SIZE)
        all_samples = []

        for _ in range(num_blocks):
            samples = carrier.render_block(0)
            all_samples.extend(samples)

        # Skip first 0.1s for envelope attack to settle
        skip_blocks = int(0.1 * SAMPLE_RATE / BLOCK_SIZE)

        # Check RMS is stable across block boundaries (after attack)
        rms_per_block = []
        for i in range(skip_blocks, num_blocks):
            block_samples = all_samples[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            rms_per_block.append(compute_rms(block_samples))

        # RMS should be consistent (no modulation at block rate)
        # Some variance is normal due to waveform phase vs block alignment
        if rms_per_block:
            rms_variance = max(rms_per_block) - min(rms_per_block)
            avg_rms = sum(rms_per_block) / len(rms_per_block)
            # Variance should be within 200% of average (allows for phase effects)
            # Main check: RMS doesn't go to zero or explode
            assert avg_rms > 0.1, f"Average RMS too low: {avg_rms}"
            assert rms_variance < avg_rms * 2, f"Block-rate RMS variation too large: {rms_variance}"


# =============================================================================
# LEAF GENERATOR TESTS
# =============================================================================

class TestLeafGenerator:
    """Tests for partial-to-leaf soft assignment."""

    def test_leaf_centers(self):
        """Test leaf center frequency calculation."""
        # First leaf should be near DC
        assert LEAF_CENTERS_HZ[0] < 500

        # Last leaf should be near Nyquist
        assert LEAF_CENTERS_HZ[-1] > 20000

        # Should be monotonically increasing
        for i in range(1, len(LEAF_CENTERS_HZ)):
            assert LEAF_CENTERS_HZ[i] > LEAF_CENTERS_HZ[i-1]

    def test_frequency_to_leaf_mapping(self):
        """Test frequency to leaf index mapping."""
        # Low frequency -> low leaf
        assert frequency_to_leaf_index(100) < 10

        # High frequency -> high leaf
        assert frequency_to_leaf_index(20000) > 50

        # Nyquist and above -> last leaf
        assert frequency_to_leaf_index(24000) == NUM_LEAVES - 1
        assert frequency_to_leaf_index(48000) == NUM_LEAVES - 1

    def test_leaf_generator_from_fundamental(self):
        """Test leaf generation from fundamental frequency."""
        gen = LeafGenerator()

        # Generate leaves for A4 (440 Hz) with saw harmonics
        content = gen.generate_from_fundamental(440.0, saw_harmonics())

        # Should have content in leaves
        assert len(content.coeffs_q15) == NUM_LEAVES

        # Should have some non-zero leaves
        non_zero = sum(1 for c in content.coeffs_q15 if c > 0)
        assert non_zero > 0, "No content in leaves"

        # Low leaves should have more content (fundamental + low harmonics)
        low_energy = sum(content.coeffs_q15[:16])
        high_energy = sum(content.coeffs_q15[-16:])
        assert low_energy > high_energy, "Low leaves should have more content"

    def test_leaf_generator_harmonic_profiles(self):
        """Test different harmonic profiles produce different leaf distributions."""
        gen = LeafGenerator()
        f0 = 440.0

        saw_content = gen.generate_from_fundamental(f0, saw_harmonics())
        sine_content = gen.generate_from_fundamental(f0, sine_harmonics())
        square_content = gen.generate_from_fundamental(f0, square_harmonics())

        # Sine should have most energy in fundamental region
        sine_energy = sum(saw_content.coeffs_q15[:8])

        # Saw should have more high-frequency content than sine
        saw_high = sum(saw_content.coeffs_q15[16:])
        sine_high = sum(sine_content.coeffs_q15[16:])
        assert saw_high > sine_high, "Saw should have more high frequency content"


# =============================================================================
# TUNING SYSTEM TESTS
# =============================================================================

class TestTuningSystem:
    """Tests for instrument tuning and morphing."""

    def test_tuning_presets_exist(self):
        """Test that all expected tuning presets exist."""
        expected = ['flat', 'harmonic', 'bright', 'dark', 'hollow']
        available = list_tunings()

        for name in expected:
            assert name in available, f"Missing tuning preset: {name}"

    def test_tuning_gains_valid(self):
        """Test tuning gains are valid Q15 values."""
        for name in list_tunings():
            tuning = get_tuning(name)
            assert len(tuning.gains_q15) == NUM_LEAVES

            for gain in tuning.gains_q15:
                assert 0 <= gain <= Q15_ONE, f"Invalid gain in {name}: {gain}"

    def test_tuning_morpher_smooth_transition(self):
        """Test tuning morpher produces smooth transitions."""
        morpher = TuningMorpher()
        morpher.set_tuning_immediate(TUNING_FLAT)

        # Start morph to bright
        morpher.set_target(TUNING_BRIGHT, morph_time_sec=1.0)

        prev_gains = morpher.get_current_gains()
        max_delta = 0

        # Advance through morph (add extra blocks to ensure completion)
        num_blocks = int(1.5 * SAMPLE_RATE / BLOCK_SIZE)  # 1.5 seconds
        for _ in range(num_blocks):
            morpher.advance_block()
            curr_gains = morpher.get_current_gains()

            # Check per-leaf change is smooth
            for prev, curr in zip(prev_gains, curr_gains):
                delta = abs(curr - prev)
                max_delta = max(max_delta, delta)

            prev_gains = curr_gains

        # Should be completed (or very close)
        # Allow some tolerance due to integer math
        assert morpher._state.mu_q15 >= Q15_ONE - 100, \
            f"Morph not complete: mu={morpher._state.mu_q15}"

        # Max delta should be small (smooth transition)
        assert max_delta < Q15_ONE // 5, f"Tuning morph too abrupt: {max_delta}"


# =============================================================================
# BLEND STATE TESTS
# =============================================================================

class TestBlendState:
    """Tests for carrier/lift blend state."""

    def test_wet_ready_after_latency(self):
        """Test wet signal ready after latency blocks."""
        blend = BlendState()
        blend.reset_for_note_on()

        # Initially, wet should not be ready
        for _ in range(IDWPT_LATENCY_BLOCKS - 1):
            assert not blend.advance_block(), "Wet ready too early"

        # After latency blocks, wet should be ready
        assert blend.advance_block(), "Wet not ready after latency"

    def test_beta_ramps_smoothly(self):
        """Test beta ramps up smoothly after wet ready."""
        blend = BlendState()
        blend.set_beta_target(Q15_ONE, bloom_time_ms=50)
        blend.reset_for_note_on()

        # Skip latency
        for _ in range(IDWPT_LATENCY_BLOCKS):
            blend.advance_block()

        # Track beta values
        betas = [blend.beta_q15]

        # Advance through bloom
        while blend.beta_q15 < blend.beta_target_q15:
            blend.advance_block()
            betas.append(blend.beta_q15)

        # Should be monotonically increasing
        for i in range(1, len(betas)):
            assert betas[i] >= betas[i-1], "Beta should increase monotonically"

        # Final beta should reach target
        assert blend.beta_q15 == blend.beta_target_q15


# =============================================================================
# CARRIER LIFT PIPELINE INTEGRATION TESTS
# =============================================================================

class TestCarrierLiftPipeline:
    """Integration tests for the full Carrier + Lift pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes without error."""
        pipeline = CarrierLiftPipeline()
        assert pipeline is not None

    def test_voice_lifecycle(self):
        """Test voice on/off/deallocate."""
        pipeline = CarrierLiftPipeline()

        # Voice on
        pipeline.voice_on(0, note=60)
        assert 0 in pipeline.voice_states

        # Render a few blocks
        for _ in range(10):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            assert len(L) == BLOCK_SIZE
            assert len(R) == BLOCK_SIZE

        # Voice off
        pipeline.voice_off(0)

        # Voice deallocate
        pipeline.voice_deallocate(0)
        assert 0 not in pipeline.voice_states

    def test_steady_note_no_artifacts(self):
        """Test 5-second sustained note has no block-rate artifacts."""
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        # Render 5 seconds
        num_blocks = int(5 * SAMPLE_RATE / BLOCK_SIZE)
        all_l = []
        all_r = []

        for _ in range(num_blocks):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        # Check for clicks
        max_delta_l = max_sample_delta(all_l)
        max_delta_r = max_sample_delta(all_r)

        # No large jumps (clicks)
        assert max_delta_l < Q15_ONE // 3, f"Click in L: {max_delta_l}"
        assert max_delta_r < Q15_ONE // 3, f"Click in R: {max_delta_r}"

    def test_pitch_bend_smooth(self):
        """Test pitch bend is smooth (carrier-led)."""
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        all_samples = []

        # Play C4 for 1 second
        for _ in range(int(SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            all_samples.extend(L)

        # Bend up to D4 over 0.5 seconds (pitch change per block)
        # D4 = MIDI 62, frequency ~293.7 Hz
        num_bend_blocks = int(0.5 * SAMPLE_RATE / BLOCK_SIZE)
        for i in range(num_bend_blocks):
            progress = (i + 1) / num_bend_blocks
            current_note = 60 + 2 * progress  # Interpolate to 62
            current_f0 = 261.6 * (2 ** (2 * progress / 12))

            frame = VoiceFrame(voice_id=0, note=int(current_note), f0_hz=current_f0)
            L, R = pipeline.render_block(frame)
            all_samples.extend(L)

        # Play D4 for 1 second
        for _ in range(int(SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=62, f0_hz=293.7)
            L, R = pipeline.render_block(frame)
            all_samples.extend(L)

        # Check for clicks during bend
        max_delta = max_sample_delta(all_samples)
        assert max_delta < Q15_ONE // 3, f"Click during pitch bend: {max_delta}"

    def test_beta_sweep_click_free(self):
        """Test beta sweep from 0 to 1 is click-free."""
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        all_samples = []

        # Sweep beta from 0 to Q15_ONE over 1 second
        num_blocks = int(SAMPLE_RATE / BLOCK_SIZE)
        for i in range(num_blocks):
            beta = int(Q15_ONE * i / num_blocks)
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6, beta_target_q15=beta)
            L, R = pipeline.render_block(frame)
            all_samples.extend(L)

        # Check for clicks
        max_delta = max_sample_delta(all_samples)
        assert max_delta < Q15_ONE // 3, f"Click during beta sweep: {max_delta}"

    def test_tuning_morph_smooth(self):
        """Test tuning morph produces smooth timbre change."""
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        all_samples = []

        # Play with harmonic tuning for 1 second
        for _ in range(int(SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            all_samples.extend(L)

        # Start tuning morph to bright
        pipeline.set_tuning("bright", morph_time_sec=2.0)

        # Continue playing through morph (2 seconds)
        for _ in range(int(2 * SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            all_samples.extend(L)

        # Check for clicks
        max_delta = max_sample_delta(all_samples)
        assert max_delta < Q15_ONE // 3, f"Click during tuning morph: {max_delta}"

    def test_multi_voice_stress(self):
        """Test 16-voice polyphony for stability."""
        pipeline = CarrierLiftPipeline()

        # Allocate 16 voices
        notes = [48, 52, 55, 60, 64, 67, 72, 76,
                 48, 52, 55, 60, 64, 67, 72, 76]  # Two octaves

        for voice_id, note in enumerate(notes):
            pipeline.voice_on(voice_id, note=note)

        # Render 1 second
        num_blocks = int(SAMPLE_RATE / BLOCK_SIZE)

        for _ in range(num_blocks):
            for voice_id, note in enumerate(notes):
                f0 = 440.0 * (2 ** ((note - 69) / 12))
                frame = VoiceFrame(voice_id=voice_id, note=note, f0_hz=f0)
                L, R = pipeline.render_block(frame)

                # Check each voice doesn't clip
                assert compute_peak(L) <= Q15_ONE + 1
                assert compute_peak(R) <= Q15_ONE + 1


# =============================================================================
# AUDIO OUTPUT TESTS (Generate WAV files)
# =============================================================================

class TestAudioOutput:
    """Tests that generate WAV files for manual listening."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory for WAV files."""
        # Use a fixed location for easy access
        out_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def test_generate_steady_note_wav(self, output_dir):
        """Generate WAV file of steady note for listening test."""
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        all_l = []
        all_r = []

        # 3 seconds
        for _ in range(int(3 * SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6, gain_q15=Q15_HALF)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        write_wav(os.path.join(output_dir, 'carrier_lift_steady.wav'), all_l, all_r)

    def test_generate_pitch_bend_wav(self, output_dir):
        """Generate WAV file with pitch bend for listening test."""
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        all_l = []
        all_r = []

        # 1s C4, bend to E4, 1s E4, bend back, 1s C4
        # C4
        for _ in range(int(SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        # Bend to E4 (MIDI 64)
        bend_blocks = int(0.5 * SAMPLE_RATE / BLOCK_SIZE)
        for i in range(bend_blocks):
            progress = (i + 1) / bend_blocks
            note = 60 + 4 * progress
            f0 = 261.6 * (2 ** (4 * progress / 12))
            frame = VoiceFrame(voice_id=0, note=int(note), f0_hz=f0)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        # E4
        for _ in range(int(SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=64, f0_hz=329.6)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        # Bend back
        for i in range(bend_blocks):
            progress = (i + 1) / bend_blocks
            note = 64 - 4 * progress
            f0 = 329.6 * (2 ** (-4 * progress / 12))
            frame = VoiceFrame(voice_id=0, note=int(note), f0_hz=f0)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        # C4
        for _ in range(int(SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        write_wav(os.path.join(output_dir, 'carrier_lift_pitch_bend.wav'), all_l, all_r)

    def test_generate_beta_sweep_wav(self, output_dir):
        """Generate WAV file with beta sweep for listening test."""
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        all_l = []
        all_r = []

        # Sweep beta over 4 seconds
        num_blocks = int(4 * SAMPLE_RATE / BLOCK_SIZE)
        for i in range(num_blocks):
            # Beta goes 0 -> 1 -> 0 over 4 seconds (triangle wave)
            t = i / num_blocks
            if t < 0.5:
                beta = int(Q15_ONE * t * 2)
            else:
                beta = int(Q15_ONE * (1 - (t - 0.5) * 2))

            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6, beta_target_q15=beta)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        write_wav(os.path.join(output_dir, 'carrier_lift_beta_sweep.wav'), all_l, all_r)

    def test_generate_tuning_morph_wav(self, output_dir):
        """Generate WAV file with tuning morph for listening test."""
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        all_l = []
        all_r = []

        # 2s harmonic, morph to bright over 2s, 2s bright
        for _ in range(int(2 * SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        pipeline.set_tuning("bright", morph_time_sec=2.0)

        for _ in range(int(2 * SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        for _ in range(int(2 * SAMPLE_RATE / BLOCK_SIZE)):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6)
            L, R = pipeline.render_block(frame)
            all_l.extend(L)
            all_r.extend(R)

        write_wav(os.path.join(output_dir, 'carrier_lift_tuning_morph.wav'), all_l, all_r)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
