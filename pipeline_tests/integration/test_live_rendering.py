"""
Live Rendering Tests
====================

Tests the live rendering pattern used in pico2_live_synth where VoiceFrame
objects are created fresh each block.

This validates:
- State persistence across blocks in UnifiedOutputStage
- Correct handling of new VoiceFrame instances per block
- No discontinuities or artifacts from frame recreation
- Clean output over extended rendering (2+ seconds)
"""

import pytest
import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.audio_io import write_audio

Q15_ONE = 32767
BLOCK_SIZE = 64
SAMPLE_RATE = 48000


def render_live_pattern(
    note=60,
    f0_hz=261.63,
    beta_target_q15=0,
    gain_q15=24576,
    duration_sec=2.0,
):
    """
    Render using live pattern: new VoiceFrame each block.

    This mimics pico2_live_synth behavior where frames are constructed
    per-block rather than reused.

    Returns:
        tuple: (L_samples, R_samples) as np.ndarray
    """
    stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
    stage.voice_on(0, note, 0)

    num_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)
    all_L = []
    all_R = []

    for _ in range(num_blocks):
        # Create NEW VoiceFrame each block (like pico2_live_synth)
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            cents=0,
            f0_hz=f0_hz,
            gain_q15=gain_q15,
            pan_q15=0,
            tree_a_id=0,
            tree_b_id=0,
            tree_blend_q15=0,
            is_active=True,
            beta_target_q15=beta_target_q15,
            partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
        )

        L, R = stage.render_block(frame)
        all_L.extend(L)
        all_R.extend(R)

    return np.array(all_L, dtype=np.int16), np.array(all_R, dtype=np.int16)


def detect_spikes(samples, threshold_sigma=3):
    """Detect spikes using second-order difference."""
    d2 = np.diff(np.diff(samples.astype(np.int32)))
    threshold = d2.std() * threshold_sigma
    spike_indices = np.where(np.abs(d2) > threshold)[0]

    return {
        "spike_count": len(spike_indices),
        "max_d2": int(d2.max()),
        "min_d2": int(d2.min()),
        "threshold": int(threshold),
        "spike_indices": spike_indices.tolist()[:10],
    }


class TestLivePatternContinuity:
    """Test live rendering pattern creates continuous output."""

    def test_live_pattern_no_spikes(self):
        """Live pattern (new frame each block) should have no spikes."""
        L, R = render_live_pattern(duration_sec=2.0)
        result = detect_spikes(L)

        assert result["spike_count"] == 0, (
            f"Found {result['spike_count']} spikes. "
            f"Max d2={result['max_d2']}, indices={result['spike_indices']}"
        )

    def test_stereo_channels_match(self):
        """For center pan, L and R should be identical."""
        L, R = render_live_pattern(duration_sec=1.0)
        max_diff = np.max(np.abs(L.astype(np.int32) - R.astype(np.int32)))
        assert max_diff <= 1, f"L/R mismatch: max diff = {max_diff}"


class TestLivePatternWithWet:
    """Test live pattern with wet (IDWPT) path enabled."""

    def test_live_pattern_50_percent_wet(self):
        """50% wet should render continuously."""
        L, R = render_live_pattern(beta_target_q15=16384, duration_sec=1.0)
        result = detect_spikes(L, threshold_sigma=5)

        assert result["spike_count"] < 10, (
            f"Too many spikes with 50% wet: {result['spike_count']}"
        )


@pytest.mark.manual
def test_generate_live_pattern_files():
    """Generate live pattern test files (not run in CI)."""
    L, R = render_live_pattern(beta_target_q15=0, duration_sec=2.0)
    meta = write_audio("live_pattern_test", L, R, timestamp=True, subdirectory="live_tests")
    print(f"Wrote: {meta.filepath}")
