"""Current CCS RTL, exercised through an actual Verilog simulator."""
import pytest

from tools.check_hdl_idwpt import Simulator, SimulatorUnavailable, check, check_rounding, check_headroom


@pytest.fixture(scope='module')
def simulator():
    try:
        return Simulator()
    except SimulatorUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('slots', [2, 4, 8])
@pytest.mark.parametrize('stalls', [False, True])
def test_hdl_arithmetic_handshakes_reset_and_cycles(simulator, slots, stalls):
    check(slots, 2, stalls, simulator=simulator)


@pytest.mark.parametrize('headroom', [0, 1, 3, 15])
def test_hdl_preserves_headroom_contract(simulator, headroom):
    check(4, headroom, True, exhaustive=False, simulator=simulator)


def test_hdl_rounding_exhausts_proven_lifting_sum_ranges(simulator):
    check_rounding(simulator)


def test_hdl_headroom_preserves_both_saturation_boundaries_exhaustively(simulator):
    check_headroom(simulator)
