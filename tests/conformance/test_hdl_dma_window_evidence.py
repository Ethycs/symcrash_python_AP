"""Window timing must not silently use a different deadline or dependency."""
import hashlib
import json
import pytest
from tools import build_hdl_dma_window as build


@pytest.mark.parametrize('damage', [*build.SOURCES, 'netlist', 'missing', 'base', 'headroom', 'window', 'name', 'K', 'duplicate', 'mapping', 'mapping_log', 'valid'])
def test_windowed_timing_requires_current_dependencies_and_budget(tmp_path, monkeypatch, damage):
    sources = {name: tmp_path / name for name in build.SOURCES}
    for path in sources.values(): path.write_text('module original; endmodule\n')

    def hashes():
        return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}

    def fingerprint():
        return hashlib.sha256(json.dumps(hashes(), sort_keys=True).encode()).hexdigest()

    key = fingerprint()
    rows = []
    for k in (2, 4, 8):
        netlist = tmp_path / f'ccs_dma_window_{key[:12]}_{build.NETLIST_TAG}_k{k}_ecp5.json'
        netlist.write_text('{}\n')
        log = netlist.with_name(netlist.name.removesuffix('_ecp5.json')+'_ecp5.log')
        log.write_text('ABC: + &verify\nABC: Networks are equivalent.\n')
        rows.append({**build.mapping_evidence(log), 'K': k, 'netlist': netlist.name, 'netlist_sha256': hashlib.sha256(netlist.read_bytes()).hexdigest()})
    report = {'sources': hashes(), 'fingerprint': key, 'mapping': build.MAPPING, 'base': 0x8000, 'headroom': 2,
              'window_cycles': 40000, 'configurations': rows}
    if damage in sources: sources[damage].write_text('module changed; endmodule\n')
    elif damage == 'netlist': netlist.write_text('{"changed": true}\n')
    elif damage == 'missing': netlist.unlink()
    elif damage == 'base': report['base'] = 0x9000
    elif damage == 'headroom': report['headroom'] = 0
    elif damage == 'window': report['window_cycles'] = 80000
    elif damage == 'name': rows[0]['netlist'] = '../unrelated.json'
    elif damage == 'K': rows.pop()
    elif damage == 'duplicate': rows[2] = rows[0]
    elif damage == 'mapping': report['mapping'] = 'abc'
    elif damage == 'mapping_log': log.write_text('ABC: Aborted (core dumped)\n')
    (tmp_path / 'ccs_dma_window_synthesis.json').write_text(json.dumps(report))
    monkeypatch.setattr(build, 'source_hashes', hashes)
    monkeypatch.setattr(build, 'fingerprint', fingerprint)
    monkeypatch.setattr(build, 'DIRECTORY', tmp_path)

    def unexpected_tool(*args, **kwargs):
        pytest.fail('Stale window evidence reached tool discovery')

    monkeypatch.setattr(build, 'Tool', unexpected_tool)
    if damage == 'valid':
        assert build.inventory() == report
        return
    with pytest.raises(RuntimeError, match='stale or incompatible windowed CCS'):
        build.route()
