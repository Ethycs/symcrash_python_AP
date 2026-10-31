"""Routing must reject changed source, numeric configuration and mapping evidence."""
import json
import pytest
from tools import build_hdl_idwpt_runtime as build


@pytest.mark.parametrize('damage', ['source', 'netlist', 'missing', 'name', 'headroom', 'slots',
                                  'lanes', 'mapping', 'log', 'valid'])
def test_runtime_routing_requires_matching_evidence(tmp_path, monkeypatch, damage):
    source = tmp_path/'ccs_idwpt_runtime.v'
    source.write_text('module original; endmodule\n')
    key = build.digest(source)
    stem = f'idwpt_runtime_{key[:12]}_{build.NETLIST_TAG}'
    netlist, log = (tmp_path/(stem+suffix) for suffix in ('_ecp5.json', '_ecp5.log'))
    netlist.write_text('{}\n')
    log.write_text('ABC: + &verify\nABC: Networks are equivalent.\n')
    report = {'source_sha256': key, 'headroom': 2, 'mapping': build.MAPPING,
              'supported_slots': [2, 4, 8], 'input_lanes': 2,
              'netlist': netlist.name, 'netlist_sha256': build.digest(netlist), **build.evidence(log)}
    if damage == 'source': source.write_text('module changed; endmodule\n')
    elif damage == 'netlist': netlist.write_text('{"changed":true}\n')
    elif damage == 'missing': netlist.unlink()
    elif damage == 'name': report['netlist'] = '../unrelated.json'
    elif damage == 'headroom': report['headroom'] = 0
    elif damage == 'slots': report['supported_slots'] = [2, 4]
    elif damage == 'lanes': report['input_lanes'] = 4
    elif damage == 'mapping': report['mapping'] = 'abc'
    elif damage == 'log': log.write_text('ABC: Aborted (core dumped)\n')
    (tmp_path/'idwpt_runtime_synthesis.json').write_text(json.dumps(report))
    monkeypatch.setattr(build, 'SOURCE', source)
    monkeypatch.setattr(build, 'DIRECTORY', tmp_path)

    def unexpected_tool(*args, **kwargs):
        pytest.fail('Stale runtime evidence reached tool discovery')

    monkeypatch.setattr(build, 'Tool', unexpected_tool)
    if damage == 'valid':
        assert build.inventory() == report
    else:
        with pytest.raises(RuntimeError, match='stale or incompatible runtime-decoder'):
            build.route()
