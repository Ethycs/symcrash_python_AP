"""All-input validation proof, with counterexamples for two boundary mistakes."""
import pytest
from tools.check_hdl_dma_bounds import SOURCE, Tool, prove


@pytest.fixture(scope='module')
def yosys():
    try:
        return Tool('yosys')
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_actual_dma_bounds_match_full_precision_reference(yosys):
    assert prove(tool=yosys)['result'] == 'PASS'


@pytest.mark.parametrize('old,new', [
    ('(|ram_end[32:28])', '(|ram_end[31:28])'),  # Miss wrap beyond 32 bits.
    ('(bram_end[16] && (|bram_end[15:0]))', 'bram_end[16]'),  # Reject exact endpoint.
])
def test_formal_gate_rejects_overflow_and_exact_endpoint_regressions(tmp_path, yosys, old, new):
    original = SOURCE.read_text(encoding='utf-8')
    assert original.count(old) == 1
    mutant = tmp_path / 'dma_burst.v'
    mutant.write_text(original.replace(old, new), encoding='utf-8')
    with pytest.raises(RuntimeError, match='bounds equivalence failed'):
        prove(mutant, tmp_path / 'evidence', yosys)
    assert not list((tmp_path / 'evidence').glob('*.json'))
