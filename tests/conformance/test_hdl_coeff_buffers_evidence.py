"""Reject stale source, memory placement, K set or netlist before routing."""
import hashlib
import json
import pytest
from tools import build_hdl_coeff_buffers as build


@pytest.mark.parametrize('damage', ['source', 'netlist', 'missing', 'base', 'name', 'K', 'duplicate', 'mapping', 'mapping_log', 'valid'])
def test_ownership_timing_requires_current_complete_inventory(tmp_path, monkeypatch, damage):
    source = tmp_path / 'coeff_buffer_owner.v'
    source.write_text('module original; endmodule\n')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = []
    for k in (2, 4, 8):
        netlist = tmp_path / f'coeff_buffer_owner_{source_hash[:12]}_{build.NETLIST_TAG}_k{k}_ecp5.json'
        netlist.write_text('{}\n')
        log = netlist.with_name(netlist.name.removesuffix('_ecp5.json')+'_ecp5.log')
        log.write_text('ABC: + &verify\nABC: Networks are equivalent.\n')
        rows.append({**build.mapping_evidence(log), 'K': k, 'netlist': netlist.name, 'netlist_sha256': hashlib.sha256(netlist.read_bytes()).hexdigest()})
    report = {'source_sha256': source_hash, 'mapping': build.MAPPING, 'base': 0x8000, 'configurations': rows}
    if damage == 'source': source.write_text('module changed; endmodule\n')
    elif damage == 'netlist': netlist.write_text('{"changed": true}\n')
    elif damage == 'missing': netlist.unlink()
    elif damage == 'base': report['base'] = 0x9000
    elif damage == 'name': rows[0]['netlist'] = '../unrelated.json'
    elif damage == 'K': rows.pop()
    elif damage == 'duplicate': rows[2] = rows[0]
    elif damage == 'mapping': report['mapping'] = 'abc'
    elif damage == 'mapping_log': log.write_text('ABC: Aborted (core dumped)\n')
    (tmp_path / 'coeff_buffer_owner_synthesis.json').write_text(json.dumps(report))
    monkeypatch.setattr(build, 'SOURCE', source)
    monkeypatch.setattr(build, 'DIRECTORY', tmp_path)

    def unexpected_tool(*args, **kwargs):
        pytest.fail('Stale ownership evidence reached tool discovery')

    monkeypatch.setattr(build, 'Tool', unexpected_tool)
    if damage == 'valid':
        assert build.inventory() == report
        return
    with pytest.raises(RuntimeError, match='stale or incompatible ownership'):
        build.route()
