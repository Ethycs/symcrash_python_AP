"""Priority scheduler plus actual DMA burst RTL, compared with an arrival-list oracle."""
import pytest

from tools.check_hdl_dma_priority import CONFIGURATIONS, Simulator, SimulatorUnavailable, check


@pytest.fixture(scope='module')
def simulator():
    try:
        return Simulator()
    except SimulatorUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('configuration', CONFIGURATIONS)
def test_priority_dma_order_pool_backpressure_reset_and_payloads(simulator, configuration):
    check(*configuration, simulator=simulator)
