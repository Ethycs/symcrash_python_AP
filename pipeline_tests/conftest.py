"""
Pytest Configuration and Fixtures
=================================

Shared fixtures for pipeline tests.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src" / "symcrash_python_ap"
sys.path.insert(0, str(src_path))


@pytest.fixture
def block_size():
    """Standard block size."""
    return 64


@pytest.fixture
def sample_rate():
    """Standard sample rate."""
    return 48000.0


@pytest.fixture
def q15_one():
    """Q1.15 maximum positive value."""
    return 32767


@pytest.fixture
def q15_half():
    """Q1.15 half value."""
    return 16384


@pytest.fixture
def voice_engine():
    """Fresh Voice Engine instance."""
    from symcrash_python_ap.voice import VoiceEngine
    return VoiceEngine(max_voices=12)


@pytest.fixture
def symbol_executor():
    """Fresh Symbol Executor instance."""
    from symcrash_python_ap.symbol import SymbolExecutor
    return SymbolExecutor(n_voices=12)


@pytest.fixture
def safety_engine():
    """Fresh Safety Engine instance."""
    from symcrash_python_ap.safety import SafetyEngine
    return SafetyEngine(block_size=64)


@pytest.fixture
def wavelet_engine():
    """Fresh Wavelet Engine instance."""
    from symcrash_python_ap.wavelet import WaveletEngine
    return WaveletEngine(block_size=64)
