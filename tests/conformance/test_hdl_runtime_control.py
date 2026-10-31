"""Mode/group invariants must survive commands, reset and arbitrary completion inputs."""
import pytest
from tools import check_hdl_runtime_control as checks


@pytest.fixture(scope='module')
def yosys():
    try:
        return checks.Tool('yosys')
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_runtime_command_group_policy_and_legality(yosys):
    assert checks.prove(tool=yosys)['result'] == 'PASS'


@pytest.mark.parametrize('old,new', [
    ('if(frame_k==3) frame_error<=1;', 'if(frame_k==2) frame_error<=1;'),
    ("slot_group<=slot_group+1'b1;", "slot_group<=slot_group+2'd2;"),
    ('if(slot_group==final_group)', 'if(slot_group==0)'),
])
def test_reject_reserved_mode_group_skip_and_early_output(tmp_path, yosys, old, new):
    text = checks.SOURCE.read_text()
    assert text.count(old) == 1
    mutant = tmp_path/checks.SOURCE.name
    mutant.write_text(text.replace(old, new), encoding='utf-8')
    with pytest.raises(RuntimeError, match='Runtime control proof failed'):
        checks.prove(mutant, tmp_path, yosys)
    assert not list(tmp_path.glob('*inventory.json'))
