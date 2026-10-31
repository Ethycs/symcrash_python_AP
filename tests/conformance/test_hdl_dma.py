"""Run the DMA engine RTL; Python generates independent payloads and bounds."""
import pytest

from tools.check_hdl_dma import Simulator, SimulatorUnavailable, check


@pytest.fixture(scope='module')
def simulator():
    try:
        return Simulator()
    except SimulatorUnavailable as exc:
        pytest.skip(str(exc))


def test_dma_all_lengths_directions_addresses_payloads_and_stalls(simulator):
    check(simulator=simulator)


@pytest.mark.parametrize('window', [1, 2, 3, 4, 8])
def test_dma_deadline_edges_and_completion_backpressure(simulator, window):
    check(False, window, simulator)
