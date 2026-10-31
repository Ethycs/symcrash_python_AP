"""One shared DMA deadline across queueing, service and owner callbacks."""
import pytest
from tools.check_hdl_dma_window import WINDOWS, STREAMS, Simulator, SimulatorUnavailable, check_window, check_stream


@pytest.fixture(scope='module')
def simulator():
    try:
        return Simulator()
    except SimulatorUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('limit', WINDOWS)
def test_shared_window_boundaries_full_queue_rejection_and_reset(simulator, limit):
    check_window(limit, simulator)


@pytest.mark.parametrize('configuration', STREAMS)
def test_shared_window_publication_fallback_sealing_retirement_and_reset(simulator, configuration):
    check_stream(*configuration, simulator=simulator)
