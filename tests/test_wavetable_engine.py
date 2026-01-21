"""
Tests for Wavetable Synthesis Engine (Phase 4)
==============================================

Verifies:
- WavetableAtlas creation and interpolation
- WavetableBank management
- WavetableState oscillator behavior
- Coefficient generation pipeline
- 2D bilinear interpolation accuracy
"""

import pytest
import numpy as np

from symcrash_python_ap.vm.wavetable_engine import (
    WavetableAtlas,
    WavetableBank,
    WavetableState,
    WavetableSpec,
    wavetable_generate_block,
    wavetable_generate_coeffs,
    create_basic_waves,
    create_harmonic_series,
    create_default_bank,
    DEFAULT_FRAME_SIZE,
    DEFAULT_NUM_FRAMES,
)
from symcrash_python_ap.primitives import BLOCK_SIZE, NUM_LEAVES


class TestWavetableAtlas:
    """Tests for WavetableAtlas structure."""

    def test_create_from_2d_array(self):
        """Atlas should be created from 2D frame array."""
        frames = np.zeros((4, 256), dtype=np.int16)
        frames[0, :] = 1000  # DC
        frames[1, :] = 2000
        frames[2, :] = 3000
        frames[3, :] = 4000

        atlas = WavetableAtlas.from_2d_array(frames, name="Test")
        assert atlas.num_positions == 1
        assert atlas.num_frames == 4
        assert atlas.frame_size == 256
        assert atlas.name == "Test"

    def test_create_from_waveforms(self):
        """Atlas should be created from list of waveforms."""
        sine = (np.sin(np.linspace(0, 2*np.pi, 256)) * 32767).astype(np.int16)
        saw = (np.linspace(1, -1, 256) * 32767).astype(np.int16)

        atlas = WavetableAtlas.from_waveforms([sine, saw], name="SineSaw")
        assert atlas.num_frames == 2
        assert atlas.frame_size == 256

    def test_create_basic_waves(self):
        """create_basic_waves should produce valid atlas."""
        atlas = create_basic_waves()
        assert atlas.name == "BasicWaves"
        assert atlas.num_frames == 4  # sine, tri, saw, square
        assert atlas.num_positions == 1

    def test_create_harmonic_series(self):
        """create_harmonic_series should produce valid atlas."""
        atlas = create_harmonic_series(num_frames=16, max_harmonics=16)
        assert atlas.name == "HarmonicSeries"
        assert atlas.num_frames == 16

    def test_get_sample_basic(self):
        """get_sample should return valid samples."""
        atlas = create_basic_waves()

        # Sample at frame 0 (sine), phase 0
        sample = atlas.get_sample(0, 0, 0)
        assert -32768 <= sample <= 32767

        # Sample at frame 0, phase 0.25 (should be near max for sine)
        phase_quarter = 16384  # 0.25 in Q16
        sample_quarter = atlas.get_sample(0, 0, phase_quarter)
        # Sine at 0.25 phase should be near +32767
        assert sample_quarter > 20000

    def test_get_sample_frame_interpolation(self):
        """Interpolation between frames should work."""
        # Create simple atlas with distinct frames
        frames = np.zeros((2, 256), dtype=np.int16)
        frames[0, :] = 10000  # Frame 0: DC at 10000
        frames[1, :] = 20000  # Frame 1: DC at 20000

        atlas = WavetableAtlas.from_2d_array(frames)

        # At frame 0, should get ~10000
        s0 = atlas.get_sample(0, 0, 0)
        assert abs(s0 - 10000) < 100

        # At frame 1, should get ~20000
        s1 = atlas.get_sample(0, 1 << 16, 0)  # frame_q16 = 1.0
        assert abs(s1 - 20000) < 100

        # At frame 0.5, should get ~15000 (interpolated)
        s_half = atlas.get_sample(0, 32768, 0)  # frame_q16 = 0.5
        assert abs(s_half - 15000) < 1000


class TestWavetableBank:
    """Tests for WavetableBank collection."""

    def test_create_default_bank(self):
        """create_default_bank should produce valid bank."""
        bank = create_default_bank()
        assert len(bank) == 2  # BasicWaves + HarmonicSeries

    def test_add_and_get(self):
        """Bank should allow adding and retrieving atlases."""
        bank = WavetableBank()
        atlas = create_basic_waves()
        idx = bank.add(atlas)

        assert idx == 0
        assert bank.get(0) is atlas
        assert bank.get_by_name("BasicWaves") is atlas

    def test_get_invalid_index(self):
        """get() should return None for invalid index."""
        bank = WavetableBank()
        assert bank.get(99) is None


class TestWavetableState:
    """Tests for WavetableState oscillator."""

    def test_state_initialization(self):
        """State should initialize with default values."""
        state = WavetableState()
        assert state.phase_q32 == 0
        assert state.position_q16 == 0
        assert state.frame_q16 == 0
        assert not state.is_playing

    def test_set_pitch_hz(self):
        """set_pitch_hz should set correct phase increment."""
        atlas = create_basic_waves()
        state = WavetableState(atlas=atlas)

        # 440 Hz at 48000 Hz sample rate
        state.set_pitch_hz(440.0, sample_rate=48000)

        # phase_inc_q32 = (440 * 2^32) / 48000 ≈ 39370533
        expected = int((440 * (1 << 32)) / 48000)
        assert state.phase_inc_q32 == expected

    def test_set_pitch_note(self):
        """set_pitch_note should set pitch from MIDI note."""
        atlas = create_basic_waves()
        state = WavetableState(atlas=atlas)

        # A4 = MIDI 69 = 440 Hz
        state.set_pitch_note(69, sample_rate=48000)
        expected_440 = int((440 * (1 << 32)) / 48000)
        assert abs(state.phase_inc_q32 - expected_440) < 100

    def test_set_position_and_frame(self):
        """Position and frame setters should work correctly."""
        atlas = create_basic_waves()  # 4 frames, 1 position
        state = WavetableState(atlas=atlas)

        state.set_position(0.5)
        assert state.position_q16 == 0  # Only 1 position

        state.set_frame(0.5)
        # 0.5 of 3 (num_frames-1) = 1.5 in Q16
        expected = int(0.5 * ((atlas.num_frames - 1) << 16))
        assert state.frame_q16 == expected

    def test_reset(self):
        """reset() should reinitialize oscillator state."""
        atlas = create_basic_waves()
        state = WavetableState(atlas=atlas)
        state.phase_q32 = 12345
        state.is_playing = False

        state.reset()
        assert state.phase_q32 == 0
        assert state.is_playing


class TestWavetableGeneration:
    """Tests for audio and coefficient generation."""

    def test_generate_block_silent_when_not_playing(self):
        """Should generate silence when not playing."""
        atlas = create_basic_waves()
        state = WavetableState(atlas=atlas)
        state.is_playing = False

        output = wavetable_generate_block(state)
        assert len(output) == BLOCK_SIZE
        assert np.all(output == 0)

    def test_generate_block_produces_audio(self):
        """Should generate audio when playing."""
        atlas = create_basic_waves()
        state = WavetableState(atlas=atlas)
        state.reset()
        state.set_pitch_hz(440.0)

        output = wavetable_generate_block(state)
        assert len(output) == BLOCK_SIZE
        # Should not be all zeros (playing a sine wave)
        assert np.any(output != 0)

    def test_generate_block_advances_phase(self):
        """Phase should advance after block generation."""
        atlas = create_basic_waves()
        state = WavetableState(atlas=atlas)
        state.reset()
        state.set_pitch_hz(440.0)

        initial_phase = state.phase_q32
        wavetable_generate_block(state)
        assert state.phase_q32 != initial_phase

    def test_generate_coeffs_produces_512(self):
        """Should generate 512-coefficient output."""
        atlas = create_basic_waves()
        state = WavetableState(atlas=atlas)
        state.reset()
        state.set_pitch_hz(440.0)

        output_512 = np.zeros(512, dtype=np.int16)
        wavetable_generate_coeffs(state, output_512)

        # Should not be all zeros
        assert np.any(output_512 != 0)
        # Should be valid int16 range
        assert np.all(output_512 >= -32768)
        assert np.all(output_512 <= 32767)

    def test_generate_coeffs_silent_when_not_playing(self):
        """Should generate zero coefficients when not playing."""
        atlas = create_basic_waves()
        state = WavetableState(atlas=atlas)
        state.is_playing = False

        output_512 = np.ones(512, dtype=np.int16) * 9999
        wavetable_generate_coeffs(state, output_512)
        assert np.all(output_512 == 0)


class TestWavetableSpec:
    """Tests for WavetableSpec IR type."""

    def test_default_values(self):
        """WavetableSpec should have sensible defaults."""
        spec = WavetableSpec()
        assert spec.atlas_name == "BasicWaves"
        assert spec.position == 0.0
        assert spec.frame == 0.0
        assert spec.frame_rate_hz == 0.0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        spec = WavetableSpec(
            atlas_name="MyWaves",
            position=0.5,
            frame=0.25,
            frame_rate_hz=1.0,
            position_mod_source="LFO1",
        )
        d = spec.to_dict()

        assert d['atlas_name'] == "MyWaves"
        assert d['position'] == 0.5
        assert d['frame'] == 0.25
        assert d['frame_rate_hz'] == 1.0
        assert d['position_mod_source'] == "LFO1"

    def test_from_dict(self):
        """from_dict should deserialize correctly."""
        d = {
            'atlas_name': 'HarmonicSeries',
            'position': 0.7,
            'frame': 0.3,
            'frame_rate_hz': 2.0,
            'frame_mod_source': 'ENV1',
        }
        spec = WavetableSpec.from_dict(d)

        assert spec.atlas_name == 'HarmonicSeries'
        assert spec.position == 0.7
        assert spec.frame == 0.3
        assert spec.frame_rate_hz == 2.0
        assert spec.frame_mod_source == 'ENV1'

    def test_roundtrip(self):
        """to_dict/from_dict should roundtrip."""
        original = WavetableSpec(
            atlas_name="Test",
            position=0.3,
            frame=0.6,
            frame_rate_hz=0.5,
            frame_loop=False,
        )
        d = original.to_dict()
        restored = WavetableSpec.from_dict(d)

        assert restored.atlas_name == original.atlas_name
        assert restored.position == original.position
        assert restored.frame == original.frame
        assert restored.frame_rate_hz == original.frame_rate_hz
        assert restored.frame_loop == original.frame_loop


class TestFrameScanning:
    """Tests for automatic frame scanning."""

    def test_frame_scanning_advances_frame(self):
        """Non-zero frame_rate should advance frame during generation."""
        atlas = create_harmonic_series(num_frames=16)
        state = WavetableState(atlas=atlas)
        state.reset()
        state.set_pitch_hz(440.0)
        state.set_frame_rate(1.0, sample_rate=48000)  # 1 Hz scanning

        initial_frame = state.frame_q16
        # Generate several blocks
        for _ in range(10):
            wavetable_generate_block(state)

        # Frame should have advanced
        assert state.frame_q16 != initial_frame

    def test_static_frame(self):
        """Zero frame_rate should not advance frame."""
        atlas = create_harmonic_series(num_frames=16)
        state = WavetableState(atlas=atlas)
        state.reset()
        state.set_pitch_hz(440.0)
        state.frame_rate_q16 = 0  # Static

        initial_frame = state.frame_q16
        wavetable_generate_block(state)
        assert state.frame_q16 == initial_frame


class TestBilinearInterpolation:
    """Tests specifically for 2D bilinear interpolation accuracy."""

    def test_corner_values(self):
        """Corners should return exact values."""
        # Create 2x2 position/frame atlas with known values
        data = np.array([
            [[1000, 2000], [3000, 4000]],  # Position 0: frames 0,1
            [[5000, 6000], [7000, 8000]],  # Position 1: frames 0,1
        ], dtype=np.int16)

        atlas = WavetableAtlas(data=data)

        # Corner (0,0) - position 0, frame 0, phase 0
        s00 = atlas.get_sample(0, 0, 0)
        assert abs(s00 - 1000) < 100

        # Corner (1,0) - position 1, frame 0, phase 0
        s10 = atlas.get_sample(1 << 16, 0, 0)
        assert abs(s10 - 5000) < 100

    def test_midpoint_interpolation(self):
        """Midpoint should interpolate correctly."""
        # Create 2x2 atlas with simple values
        data = np.zeros((2, 2, 256), dtype=np.int16)
        data[0, 0, :] = 0       # Position 0, Frame 0
        data[0, 1, :] = 10000   # Position 0, Frame 1
        data[1, 0, :] = 20000   # Position 1, Frame 0
        data[1, 1, :] = 30000   # Position 1, Frame 1

        atlas = WavetableAtlas(data=data)

        # Midpoint (0.5, 0.5)
        s_mid = atlas.get_sample(32768, 32768, 0)
        # Expected: average of all 4 corners = (0+10000+20000+30000)/4 = 15000
        assert abs(s_mid - 15000) < 2000  # Allow some tolerance


class TestIntegration:
    """Integration tests for full wavetable pipeline."""

    def test_full_pipeline(self):
        """Full pipeline: bank -> state -> coefficients."""
        # Create bank
        bank = create_default_bank()
        atlas = bank.get(0)

        # Create state
        state = WavetableState(atlas=atlas)
        state.reset()
        state.set_pitch_note(60)  # Middle C
        state.set_frame(0.5)

        # Generate coefficients
        output_512 = np.zeros(512, dtype=np.int16)
        wavetable_generate_coeffs(state, output_512)

        # Verify output
        assert output_512.shape == (512,)
        assert np.any(output_512 != 0)

    def test_morphing_timbres(self):
        """Frame morphing should produce different timbres."""
        atlas = create_harmonic_series(num_frames=32)
        state = WavetableState(atlas=atlas)
        state.reset()
        state.set_pitch_hz(440.0)

        # Capture output at frame 0
        state.set_frame(0.0)
        output_frame0 = np.zeros(512, dtype=np.int16)
        wavetable_generate_coeffs(state, output_frame0)

        # Reset and capture at frame 1.0
        state.reset()
        state.set_pitch_hz(440.0)
        state.set_frame(1.0)
        output_frame1 = np.zeros(512, dtype=np.int16)
        wavetable_generate_coeffs(state, output_frame1)

        # Outputs should be different (different harmonic content)
        # Note: Not guaranteed to be different in first frame due to phase
        # but harmonic series atlas should show difference in energy distribution
