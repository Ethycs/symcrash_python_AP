"""All-input admission/matching proofs and actual physical-state mutations."""
import json
from pathlib import Path
import re

import pytest
from tools import check_hdl_owner_pipeline as checks


@pytest.fixture(scope='module')
def yosys():
    try:
        return checks.Tool('yosys')
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_actual_admission_and_captured_matching(yosys):
    result = checks.prove('prematch', tool=yosys)
    assert result['result'] == result['admission']['result'] == 'PASS'


@pytest.mark.parametrize('old,new', [
    ('free_mask[1] || !tag_matches[1]', 'free_mask[1] || tag_matches[1]'),
    ('!free_mask[0] && free_mask[1]', 'free_mask[1]'),
])
def test_reject_duplicate_and_bank_priority_errors(tmp_path, yosys, old, new):
    original = checks.OWNER.read_text(encoding='utf-8')
    assert original.count(old) == 1
    mutant = tmp_path/'coeff_buffer_owner.v'
    mutant.write_text(original.replace(old, new), encoding='utf-8')
    with pytest.raises(RuntimeError, match='admission proof failed'):
        checks.prove('admission', owner=mutant, directory=tmp_path, tool=yosys)
    assert not list(tmp_path.glob('*inventory.json'))


def test_reject_wrong_descriptor_chunk(tmp_path, yosys):
    original = checks.WINDOW.read_text(encoding='utf-8')
    old = 'next_callback_descriptor[15:0]==match_descriptors[15:0]'
    assert original.count(old) == 1
    mutant = tmp_path/'dma_window.v'
    mutant.write_text(original.replace(old, 'next_callback_descriptor[15:0]==match_descriptors[79:64]'), encoding='utf-8')
    with pytest.raises(RuntimeError, match='prematch proof failed'):
        checks.prove('prematch', window=mutant, directory=tmp_path, tool=yosys)
    assert not list(tmp_path.glob('owner_prematch_*_inventory.json'))


def test_reject_changed_captured_match_reduction():
    original = checks.WINDOW.read_text(encoding='utf-8')
    old = '&callback_matches[7:4]'
    assert original.count(old) == 1
    with pytest.raises(AssertionError):
        checks.prematch_module(checks.OWNER.read_text(encoding='utf-8'), original.replace(old, '|callback_matches[7:4]'))


def test_reject_completion_on_new_waiting_claim(tmp_path, yosys):
    original = checks.OWNER.read_text(encoding='utf-8')
    old = 'WAITING: if(bank_submit) next_state=expire'
    assert original.count(old) == 1
    mutant = tmp_path/'coeff_buffer_owner.v'
    mutant.write_text(original.replace(old, 'WAITING: if(finished) next_state=READY; else if(bank_submit) next_state=expire'), encoding='utf-8')
    with pytest.raises(RuntimeError, match='prematch proof failed'):
        checks.prove('prematch', owner=mutant, directory=tmp_path, tool=yosys)
    assert not list(tmp_path.glob('owner_prematch_*_inventory.json'))


@pytest.fixture(scope='module', params=[(2, 1), (8, 0)], ids=['k2_captured', 'k8_direct'])
def encoded(request, tmp_path_factory, yosys):
    k, mode = request.param
    return checks.recode(directory=tmp_path_factory.mktemp('owner-encoding'), k=k, matched=mode, tool=yosys)


def test_actual_physical_encoding(encoded):
    assert encoded['result'] == 'PASS'


def test_reject_corrupted_physical_reset(encoded, yosys):
    directory = Path(encoded['directory'])
    netlist = json.loads((directory/'after.json').read_text())
    targets = [c for c in netlist['modules']['coeff_buffer_owner']['cells'].values()
               if c['type'] in ('$adff', '$adffe') and int(c['parameters']['WIDTH'], 2) == 7]
    assert len(targets) == 2
    cell = targets[0]
    value = int(cell['parameters']['ARST_VALUE'], 2)
    assert value and not value & (value-1)
    cell['parameters']['ARST_VALUE'] = f'{((value << 1) & 127) | (value >> 6):07b}'
    mutant = directory/'wrong_reset.json'
    mutant.write_text(json.dumps(netlist)+'\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='encoding proof failed'):
        checks.encoding_equivalence(directory, yosys, mutant)
    assert re.search(r'ERROR: Found [1-9][0-9]* unproven \$equiv cells',
                     (directory/'wrong_reset_equivalence.log').read_text())
