"""
Pytest Configuration for Audio Generation Tests
================================================

Shared fixtures and configuration for all test modules.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add audio_out to path for test_config imports
sys.path.insert(0, str(Path(__file__).parent.parent / "audio_out"))

# Import common modules
from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame


# Constants
SAMPLE_RATE = 48000
BLOCK_SIZE = 64
Q15_ONE = 32767
Q15_HALF = 16384


@pytest.fixture
def pipeline():
    """Create a fresh CarrierLiftPipeline instance."""
    return CarrierLiftPipeline()


@pytest.fixture
def pipeline_with_voice():
    """Create a pipeline with voice 0 initialized at middle C."""
    p = CarrierLiftPipeline()
    p.voice_on(0, note=60)
    return p


@pytest.fixture
def basic_frame():
    """Create a basic VoiceFrame for testing."""
    return VoiceFrame(
        voice_id=0,
        note=60,
        f0_hz=261.63,
        gain_q15=Q15_HALF,
        beta_target_q15=0,  # Pure dry
    )


@pytest.fixture
def wet_frame():
    """Create a VoiceFrame with wet path enabled."""
    return VoiceFrame(
        voice_id=0,
        note=60,
        f0_hz=261.63,
        gain_q15=Q15_HALF,
        beta_target_q15=Q15_HALF,  # 50% wet
    )


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "audio: marks tests that generate audio files"
    )
