"""DMA place-and-route must reject incompatible/stale inputs before tool lookup."""
import hashlib
import json

import pytest

from tools import build_hdl_dma


@pytest.mark.parametrize('damage', ['source', 'netlist', 'missing', 'window', 'name'])
def test_dma_route_rejects_stale_synthesis(tmp_path, monkeypatch, damage):
    source = tmp_path / 'dma_burst.v'
    source.write_text('module dma_burst; endmodule\n')
    directory = tmp_path / 'build'
    directory.mkdir()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    netlist = directory / f'dma_burst_{source_hash[:12]}_ecp5.json'
    netlist.write_text('{}\n')
    inventory = {'source': source.name, 'source_sha256': source_hash,
                 'max_cycles': 40000, 'netlist': netlist.name,
                 'netlist_sha256': hashlib.sha256(netlist.read_bytes()).hexdigest()}
    if damage == 'source':
        source.write_text('module changed; endmodule\n')
    elif damage == 'netlist':
        netlist.write_text('{"changed": true}\n')
    elif damage == 'missing':
        netlist.unlink()
    elif damage == 'window':
        inventory['max_cycles'] = 80000
    else:
        inventory['netlist'] = '../unrelated.json'
    (directory / 'dma_burst_synthesis.json').write_text(json.dumps(inventory))
    monkeypatch.setattr(build_hdl_dma, 'ROOT', tmp_path)
    monkeypatch.setattr(build_hdl_dma, 'SOURCE', source)
    monkeypatch.setattr(build_hdl_dma, 'DIRECTORY', directory)

    def unexpected_discovery(*args, **kwargs):
        pytest.fail('Invalid DMA inventory reached tool discovery')

    monkeypatch.setattr(build_hdl_dma.shutil, 'which', unexpected_discovery)
    with pytest.raises(RuntimeError, match='stale or incompatible DMA'):
        build_hdl_dma.route()
