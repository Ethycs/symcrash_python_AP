"""Actual queued DMA, ownership, payload RAM and CCS decoder, including fallback."""
import pytest
from tools.check_hdl_ccs_stream import CONFIGURATIONS, Simulator, SimulatorUnavailable, check


@pytest.fixture(scope='module')
def simulator():
    try:
        return Simulator()
    except SimulatorUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('configuration', CONFIGURATIONS)
def test_dma_to_audio_exact_layout_release_prefetch_fallback_and_reset(simulator, configuration):
    check(*configuration, simulator=simulator)
