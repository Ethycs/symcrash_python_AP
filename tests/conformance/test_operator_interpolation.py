"""Independent interpolation invariants, beyond Python/JIT agreement."""
import numpy as np
import pytest

from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.vm.decode import BackboneEntry
from symcrash_python_ap.vm.lanes import dispatch_coeff_op
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.tables import COEFF_LEN, NODE_LEN


@pytest.fixture(params=[False, True], ids=['python', 'jit'])
def backend(request):
    if request.param and not hot.NUMBA_AVAILABLE:
        pytest.skip('numba unavailable')
    old = hot.jit_enabled()
    hot.set_jit_enabled(request.param)
    yield
    hot.set_jit_enabled(old)


@pytest.mark.parametrize('op_id', [3, 12], ids=['IntraMix', 'GluePair'])
def test_zero_strength_is_identity(backend, op_id):
    state = VMState()
    original = np.random.default_rng(42).integers(-32768, 32768, COEFF_LEN, dtype=np.int16)
    state.coeff_state = original.copy()
    dispatch_coeff_op(state, BackboneEntry(op_id, strength_q15=0), 0xFFFFFFFF)
    np.testing.assert_array_equal(state.coeff_state, original)


@pytest.mark.parametrize('op_id', [3, 12])
@pytest.mark.parametrize('strength', [1, 16384, 32767])
def test_constant_coefficients_are_a_fixed_point(backend, op_id, strength):
    state = VMState()
    state.coeff_state[:] = -12345
    dispatch_coeff_op(state, BackboneEntry(op_id, strength_q15=strength), 0xFFFFFFFF)
    assert np.all(state.coeff_state == -12345)


@pytest.mark.parametrize('strength,fraction', [(0, 0), (16384, .5), (32767, 1)])
def test_intramix_interpolates_toward_weighted_average(backend, strength, fraction):
    state = VMState()
    state.coeff_state[:] = 8000
    state.coeff_state[1] = -8000
    dispatch_coeff_op(state, BackboneEntry(3, strength_q15=strength), 1)
    # [8000, -8000, 8000] under [1/4, 1/2, 1/4] averages to zero.
    assert abs(int(state.coeff_state[1]) - (-8000 * (1 - fraction))) <= 1
    assert np.all(state.coeff_state[NODE_LEN:] == 8000)  # Inactive nodes unchanged.


@pytest.mark.parametrize('strength,fraction', [(0, 0), (16384, .5), (32767, 1)])
def test_glue_interpolates_pair_toward_mean(backend, strength, fraction):
    state = VMState()
    state.coeff_state[:] = 500
    state.coeff_state[0], state.coeff_state[NODE_LEN] = 24000, -8000
    dispatch_coeff_op(state, BackboneEntry(12, strength_q15=strength), 3)
    expected = [24000 * (1 - fraction) + 8000 * fraction,
                -8000 * (1 - fraction) + 8000 * fraction]
    np.testing.assert_allclose(state.coeff_state[[0, NODE_LEN]], expected, atol=1, rtol=0)
    assert np.all(state.coeff_state[2 * NODE_LEN:] == 500)


@pytest.mark.parametrize('value', [-32768, -24000, 24000, 32767])
@pytest.mark.parametrize('strength,fraction', [(0, 0), (16384, .5), (32767, 1)])
def test_gate_mask_uses_widened_product_without_wraparound(backend, value, strength, fraction):
    import warnings

    state = VMState()
    state.coeff_state[:] = value
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        dispatch_coeff_op(state, BackboneEntry(16, strength_q15=strength), 3)
    # Sturmian nodes zero/one are respectively closed/open.
    # Q15 unity is one LSB below 1.0, and the final shift truncates: together
    # those reductions contribute less than two LSBs at the signed extrema.
    np.testing.assert_allclose(state.coeff_state[:NODE_LEN], value * (1 - fraction), atol=2, rtol=0)
    assert np.all(state.coeff_state[NODE_LEN:] == value)
