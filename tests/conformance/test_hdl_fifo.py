"""Mapped FIFO state/data integrity, including actual netlist corruption."""
from copy import deepcopy
import pytest
from tools.check_hdl_fifo import SOURCE, Tool, extract, synthesize, prove


@pytest.fixture(scope='module')
def mapping():
    try:
        tool = Tool('yosys')
    except RuntimeError as exc:
        pytest.skip(str(exc))
    mapped, metadata, stem = synthesize(tool=tool)
    return mapped, metadata, stem, tool


def test_all_states_and_ports_of_mapped_fifo(mapping):
    mapped, metadata, stem, tool = mapping
    assert prove(mapped, metadata, stem, tool=tool)['result'] == 'PASS'


@pytest.mark.parametrize('damage', ['write_data', 'read_data'])
def test_mapped_corruption_is_rejected(mapping, damage, tmp_path):
    mapped, metadata, stem, tool = mapping
    broken = deepcopy(mapped)
    if damage == 'write_data':
        cell = next(c for c in broken['cells'].values() if c['type'] == 'TRELLIS_FF')
        cell['connections']['DI'] = ['0']
    else:
        broken['ports']['current_word']['bits'][0] = '0'
    with pytest.raises(RuntimeError, match='Mapped FIFO transition proof failed'):
        prove(broken, metadata, stem, directory=tmp_path, tool=tool)
    assert not list(tmp_path.glob('*_proof.json'))


def test_changed_storage_contract_requires_new_proof_model():
    text = SOURCE.read_text(encoding='utf-8')
    with pytest.raises(AssertionError, match='outside this proof model'):
        extract(text.replace('words [0:1]', 'words [0:3]'))
