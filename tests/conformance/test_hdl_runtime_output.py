"""Canonical output coordinates and meaningful corruption controls."""
import pytest
from tools import check_hdl_runtime_output as checks


@pytest.fixture(scope='module')
def yosys():
    try:
        return checks.Tool('yosys')
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_all_input_output_coordinates(yosys):
    assert checks.prove(tool=yosys)['result'] == 'PASS'


@pytest.mark.parametrize('old,new', [
    ('slot_group[0]==output_part[0]', 'slot_group[0]!=output_part[0]'),
    ('final_samples[2*(output_part/2)]', 'final_samples[output_part/2]'),
    ('words[output_index[3:1]]', 'words[output_index[2:0]]'),
    ('rst_n && update_valid[4] && level==5 && selected', 'update_valid[4] && level==5 && selected'),
])
def test_reject_wrong_slot_pair_read_address_and_reset_write(tmp_path, yosys, old, new):
    text = checks.SOURCE.read_text()
    assert text.count(old) == 1
    mutant = tmp_path/checks.SOURCE.name
    mutant.write_text(text.replace(old, new), encoding='utf-8')
    with pytest.raises(RuntimeError, match='Runtime output proof failed'):
        checks.prove(mutant, tmp_path, yosys)
    assert not list(tmp_path.glob('*inventory.json'))


def test_reject_changed_lifting_datapath():
    text = checks.SOURCE.read_text()
    old = 'quotient = value >>> 2;'
    assert text.count(old) == 1
    with pytest.raises(AssertionError, match='Reused datapath changed'):
        checks.preserved_datapath(text.replace(old, 'quotient = value >>> 1;'), checks.STATIC.read_text())
