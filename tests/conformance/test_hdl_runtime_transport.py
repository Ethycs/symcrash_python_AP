"""Actual mixed-mode audio and source-bound transport proof corruption checks."""
import pytest
from tools import check_hdl_runtime_transport as transport
from tools import check_hdl_runtime_transport_proofs as proofs


@pytest.fixture(scope='module')
def yosys():
    try:return proofs.Tool('yosys')
    except RuntimeError as exc:pytest.skip(str(exc))


@pytest.mark.parametrize('kind',['stream','window'])
def test_mixed_runtime_transport_with_stalls(kind):
    try:simulator=transport.Simulator()
    except RuntimeError as exc:pytest.skip(str(exc))
    row=transport.check(kind,2,True,False,simulator)
    assert row['exact_samples']==(10432 if kind=='stream' else 7808)


def test_runtime_transport_compositional_proofs(yosys):
    report=proofs.check()
    assert all(report[k]['result']=='PASS' for k in ('admission','gather','policy','prematch'))


@pytest.mark.parametrize('kind,old,new',[
    ('admission','wire valid_mode=prefetch_k!=3;','wire valid_mode=prefetch_k!=2;'),
    ('admission',"(prefetch_k==1) ? 8'd32 : 8'd64","(prefetch_k==1) ? 8'd16 : 8'd64"),
    ('gather','{chunk[0],group_index[0]}','{group_index[0],chunk[0]}'),
    ('gather',': group_index;',": 2'd0;"),
])
def test_reject_mode_length_and_gather_corruption(tmp_path,monkeypatch,yosys,kind,old,new):
    source=proofs.owner if kind=='admission' else proofs.stream
    text=source.read_text(encoding='utf-8');assert text.count(old)==1
    text=text.replace(old,new);(tmp_path/source.name).write_text(text,encoding='utf-8')
    monkeypatch.setattr(proofs,'d',tmp_path)
    builder=proofs.admission_module if kind=='admission' else proofs.gather_module
    with pytest.raises(RuntimeError,match=kind+' proof failed'):
        proofs.prove(kind,builder(text),yosys)


@pytest.mark.parametrize('old,new',[
    ('localparam FRAME_BYTES = 1024;','localparam FRAME_BYTES = 512;'),
    ('mode[bank] <= prefetch_k;',"mode[bank] <= 2'd0;"),
])
def test_reject_changed_bank_stride_and_mode_capture(old,new):
    text=proofs.owner.read_text(encoding='utf-8');assert text.count(old)==1
    with pytest.raises(AssertionError):proofs.admission_module(text.replace(old,new))
