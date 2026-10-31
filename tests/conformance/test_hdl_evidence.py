"""Reject stale hardware evidence before invoking the place-and-route tool."""
import hashlib
import json

import pytest

from tools import route_hdl_idwpt


@pytest.mark.parametrize('damage', ['source', 'netlist', 'missing_netlist', 'missing_configuration', 'headroom'])
def test_route_rejects_stale_or_incompatible_synthesis(tmp_path, monkeypatch, damage):
    source = tmp_path / 'core.v'
    source.write_text('module core; endmodule\n')
    directory = tmp_path / 'build'
    directory.mkdir()
    netlist = directory / 'ccs_idwpt_k4_ecp5.json'
    netlist.write_text('{}\n')
    inventory = {
        'source': source.name,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'headroom_bits': 2,
        'configurations': [{'K': 4, 'netlist_sha256': hashlib.sha256(netlist.read_bytes()).hexdigest()}],
    }
    if damage == 'source':
        source.write_text('module changed; endmodule\n')
    elif damage == 'netlist':
        netlist.write_text('{"modified": true}\n')
    elif damage == 'missing_netlist':
        netlist.unlink()
    elif damage == 'missing_configuration':
        inventory['configurations'] = []
    else:
        inventory['headroom_bits'] = 3
    (directory / 'ccs_idwpt_synthesis.json').write_text(json.dumps(inventory))
    monkeypatch.setattr(route_hdl_idwpt, 'ROOT', tmp_path)
    monkeypatch.setattr(route_hdl_idwpt, 'DIRECTORY', directory)

    def unexpected_discovery(*args, **kwargs):
        pytest.fail('Stale evidence reached tool discovery')

    monkeypatch.setattr(route_hdl_idwpt.shutil, 'which', unexpected_discovery)
    with pytest.raises(RuntimeError, match='[Ss]tale or incompatible|Missing or stale'):
        route_hdl_idwpt.route(slots=(4,))
