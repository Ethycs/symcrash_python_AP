"""Both source dependencies and the exact queue netlist must precede timing."""
import hashlib
import json

import pytest

from tools import build_hdl_dma_priority as build


@pytest.mark.parametrize('damage', ['scheduler', 'engine', 'netlist', 'missing', 'window', 'name'])
def test_priority_dma_rejects_stale_dependency_or_netlist(tmp_path, monkeypatch, damage):
    sources = {name: tmp_path / name for name in ('dma_burst.v', 'dma_priority.v')}
    for path in sources.values(): path.write_text('module initial_version; endmodule\n')

    def hashes():
        return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}

    def fingerprint():
        return hashlib.sha256(json.dumps(hashes(), sort_keys=True).encode()).hexdigest()

    key = fingerprint()
    netlist = tmp_path / f'dma_priority_{key[:12]}_ecp5.json'
    netlist.write_text('{}\n')
    inventory = {'sources': hashes(), 'fingerprint': key, 'max_cycles': 40000,
                 'netlist': netlist.name, 'netlist_sha256': hashlib.sha256(netlist.read_bytes()).hexdigest()}
    if damage in ('scheduler', 'engine'):
        sources['dma_priority.v' if damage == 'scheduler' else 'dma_burst.v'].write_text('module changed; endmodule\n')
    elif damage == 'netlist': netlist.write_text('{"changed": true}\n')
    elif damage == 'missing': netlist.unlink()
    elif damage == 'window': inventory['max_cycles'] = 80000
    else: inventory['netlist'] = '../unrelated.json'
    (tmp_path / 'dma_priority_synthesis.json').write_text(json.dumps(inventory))
    monkeypatch.setattr(build, 'DIRECTORY', tmp_path)
    monkeypatch.setattr(build, 'source_hashes', hashes)
    monkeypatch.setattr(build, 'fingerprint', fingerprint)

    def unexpected_tool(*args, **kwargs):
        pytest.fail('Stale queue evidence reached tool discovery')

    monkeypatch.setattr(build, 'Tool', unexpected_tool)
    with pytest.raises(RuntimeError, match='stale or incompatible priority DMA'):
        build.route()
