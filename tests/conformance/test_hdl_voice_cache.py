"""Voice-state isolation, full-page preservation, failures and proof controls."""
import pytest
from tools import check_hdl_voice_cache as cache
from tools import check_hdl_voice_cache_control as control
from tools import check_hdl_voice_cache_ports as ports


@pytest.fixture(scope='module')
def simulator():
    try:
        return cache.Simulator()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope='module')
def yosys():
    try:
        return control.Tool('yosys')
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_voice_cache_with_real_window(tmp_path, simulator):
    row = cache.check(100, True, directory=tmp_path, simulator=simulator)
    assert row['counts']['reads'] == 2690
    assert row['counts']['readonly_reads'] == 128
    assert row['counts']['owner_conflicts'] == 2
    assert row['counts']['writes'] == 259
    assert row['counts']['max_window'] < 40000
    assert row['counts']['payload_resets'] == 4


def test_voice_cache_control_induction(tmp_path, yosys):
    row = control.prove(control.SOURCE.read_text(encoding='utf-8'), tmp_path, yosys)
    assert row['result'] == 'PASS'


def test_native_cache_ownership_induction(tmp_path, yosys):
    row = ports.prove(cache.SOURCE.read_text(encoding='utf-8'), tmp_path, yosys)
    assert row['result'] == 'PASS'


@pytest.mark.parametrize('old,new', [
    ('!dirty_mask[prefetch_slot] && !(lease_valid && lease_slot==prefetch_slot)', '!dirty_mask[prefetch_slot]'),
    ('wire retry_eligible=dirty_mask[retry_slot]', "wire retry_eligible=1'b1"),
    ('if(memory_write) lease_dirty<=1;', 'if(memory_write) lease_dirty<=0;'),
    ('!dirty && !(lease_valid && lease_slot==slot)', '!(lease_valid && lease_slot==slot)'),
])
def test_voice_cache_proof_rejects_unsafe_control(tmp_path, yosys, old, new):
    text = control.SOURCE.read_text(encoding='utf-8')
    assert text.count(old) == 1
    with pytest.raises(RuntimeError, match='Voice cache control proof failed'):
        control.prove(text.replace(old, new), tmp_path, yosys)
    assert 'proof did fail' in (tmp_path / 'control.log').read_text(encoding='utf-8')


@pytest.mark.parametrize('old,new', [
    ('voice_id[slot]==acquire_voice_id', 'voice_id[slot][7:0]==acquire_voice_id[7:0]'),
    ('pages[dma_bram_address[14:4]]<=dma_bram_write_data;', "pages[dma_bram_address[14:4]]<={64'b0,dma_bram_write_data[63:0]};"),
    ('if(dma_store) begin if(success) dirty<=0;end', 'if(dma_store) begin dirty<=0;end'),
])
def test_voice_cache_simulation_rejects_data_loss(tmp_path, simulator, old, new):
    source = cache.SOURCE
    text = source.read_text(encoding='utf-8')
    assert text.count(old) == 1
    mutated = tmp_path / source.name
    mutated.write_text(text.replace(old, new), encoding='utf-8')
    with pytest.raises((AssertionError, RuntimeError), match='FATAL'):
        cache.check(0, False, source=mutated, directory=tmp_path, simulator=simulator)


@pytest.mark.parametrize('old,new,message', [
    ('wire memory_write=state_accept && dsp_write;',
     'wire memory_write=(state_accept && dsp_write) || read_accept;', 'Read-only lease wrote state'),
    ('wire response_ready=lease_readonly ? read_response_ready : dsp_response_ready;',
     'wire response_ready=read_response_ready || dsp_response_ready;',
     'Wrong client consumed readonly response or changed page'),
    ('lease_readonly<=acquire_readonly;', 'lease_readonly<=!acquire_readonly;',
     'Read lease class followed live acquisition input'),
    ('wire release_lease=state_release || read_release;',
     'wire release_lease=state_release || read_release || (read_release_valid && lease_valid);',
     'DSP unexpectedly blocked'),
])
def test_cache_rejects_non_owner_operations(tmp_path, simulator, old, new, message):
    text = cache.SOURCE.read_text(encoding='utf-8')
    assert text.count(old) == 1
    mutated = tmp_path / cache.SOURCE.name
    mutated.write_text(text.replace(old, new), encoding='utf-8')
    with pytest.raises((AssertionError, RuntimeError), match=message):
        cache.check(0, False, source=mutated, directory=tmp_path, simulator=simulator)
