"""Runtime routing must bind every RTL dependency, numeric profile and mapping log."""
import json
import pytest
from tools import build_hdl_runtime_transport as build


@pytest.mark.parametrize('kind',['stream','window'])
@pytest.mark.parametrize('damage',['source','netlist','missing','base','headroom','window','name','mapping','log','valid'])
def test_runtime_transport_evidence(tmp_path,monkeypatch,kind,damage):
    for name in build.source_names(kind):(tmp_path/name).write_text('module original;endmodule\n')
    monkeypatch.setattr(build,'HDL',tmp_path);monkeypatch.setattr(build,'d',tmp_path)
    key=build.fingerprint(kind);stem=f'{build.top(kind)}_{key[:12]}_abc9'
    netlist,log=(tmp_path/(stem+s) for s in ('_ecp5.json','_ecp5.log'))
    netlist.write_text('{}\n');log.write_text('ABC: + &verify\nABC: Networks are equivalent.\n')
    report={'sources':build.hashes(kind),'fingerprint':key,'headroom':2,'base':0x8000,'mapping':build.MAPPING,
        'window_cycles':40000 if kind=='window' else None,'netlist':netlist.name,'netlist_sha256':build.digest(netlist),**build.evidence(log)}
    if damage=='source':(tmp_path/build.source_names(kind)[0]).write_text('module changed;endmodule\n')
    elif damage=='netlist':netlist.write_text('{"changed":true}\n')
    elif damage=='missing':netlist.unlink()
    elif damage=='base':report['base']=0x9000
    elif damage=='headroom':report['headroom']=0
    elif damage=='window':report['window_cycles']=80000
    elif damage=='name':report['netlist']='../unrelated.json'
    elif damage=='mapping':report['mapping']='abc'
    elif damage=='log':log.write_text('ABC: Aborted (core dumped)\n')
    (tmp_path/f'runtime_transport_{kind}_synthesis.json').write_text(json.dumps(report))
    def unexpected(*args,**kwargs):pytest.fail('Stale runtime transport evidence reached routing')
    monkeypatch.setattr(build,'Tool',unexpected)
    if damage=='valid':assert build.inventory(kind)==report
    else:
        with pytest.raises(RuntimeError,match='stale or incompatible runtime transport'):build.route(kind)
