"""One physical decoder must preserve numerical results across runtime K changes."""
import pytest
from tools import check_hdl_idwpt_runtime as checks


@pytest.fixture(scope='module')
def simulator():
    try:
        return checks.Simulator()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('headroom', [0, 2, 15])
@pytest.mark.parametrize('stalls', [False, True])
def test_runtime_modes_resets_and_exact_audio(headroom, stalls, simulator):
    report = checks.check(headroom, stalls, False, simulator)
    assert report['exact_samples'] == 6912
    assert 'resets=55 rejected=16' in report['summary']
