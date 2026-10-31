"""Coefficient buffer publication/leases with the real queued DMA path."""
import pytest
from tools.check_hdl_coeff_buffers import CONFIGURATIONS, Simulator, SimulatorUnavailable, check


@pytest.fixture(scope='module')
def simulator():
    try:
        return Simulator()
    except SimulatorUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('configuration', CONFIGURATIONS)
def test_ownership_deadlines_errors_reset_and_two_block_prefetch(simulator, configuration):
    check(*configuration, simulator=simulator)
