"""Exhaustive control-cone equivalence, with observable-error negative controls."""
import pytest
from tools.check_hdl_control import SOURCES, Tool, prove


@pytest.fixture(scope='module')
def yosys():
    try:
        return Tool('yosys')
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('kind', list(SOURCES))
def test_actual_control_cone(kind, yosys):
    assert prove(kind, tool=yosys)['result'] == 'PASS'


@pytest.mark.parametrize('kind,old,new', [
    ('owner', 'next_state=expire ? DISCARD_FILL : FILLING', 'next_state=expire ? FILLING : FILLING'),
    ('payload_write', 'payload_address<=dma_bram_address;', "payload_address<=dma_bram_address ^ 16'd16;"),
    ('payload_read', 'payload[{lease_bank,read_index[ADDRESS_BITS-1:0]}];',
                     "payload[{lease_bank,read_index[ADDRESS_BITS-1:0]}] ^ 128'd1;"),
])
def test_gate_rejects_transition_and_qualified_payload_errors(kind, old, new, tmp_path, yosys):
    original = SOURCES[kind].read_text(encoding='utf-8')
    assert original.count(old) == 1
    mutant = tmp_path / SOURCES[kind].name
    mutant.write_text(original.replace(old, new), encoding='utf-8')
    with pytest.raises(RuntimeError, match='control proof failed'):
        prove(kind, mutant, tmp_path / 'evidence', yosys)
    assert not list((tmp_path / 'evidence').glob('*.json'))
