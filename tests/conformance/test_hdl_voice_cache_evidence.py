"""Do not route a cache netlist from stale sources, mapping or geometry."""
import json
from copy import deepcopy
import pytest
from tools import build_hdl_voice_cache as build
from tools import check_hdl_voice_cache_mapping as mapping


@pytest.mark.parametrize('damage', ['source', 'netlist', 'missing', 'page', 'pages', 'payload', 'name', 'mapping', 'log', 'valid'])
def test_voice_cache_evidence(tmp_path, monkeypatch, damage):
    source = tmp_path / 'voice_state_cache.v'
    source.write_text('module voice_state_cache; endmodule\n')
    monkeypatch.setattr(build, 'SOURCE', source)
    monkeypatch.setattr(build, 'DIRECTORY', tmp_path)
    name = build.stem()
    netlist, log = (tmp_path / (name + suffix) for suffix in ('_ecp5.json', '_ecp5.log'))
    netlist.write_text('{}\n')
    log.write_text('ABC: + &verify\nABC: Networks are equivalent.\n')
    report = {'source_sha256': build.digest(source), 'mapping': build.MAPPING,
        'logical_page_bytes': 2048, 'pages': 16, 'logical_payload_bytes': 32768,
        'netlist': netlist.name, 'netlist_sha256': build.digest(netlist), **build.evidence(log)}
    if damage == 'source': source.write_text('module altered; endmodule\n')
    elif damage == 'netlist': netlist.write_text('{"altered":true}\n')
    elif damage == 'missing': netlist.unlink()
    elif damage == 'page': report['logical_page_bytes'] = 1024
    elif damage == 'pages': report['pages'] = 8
    elif damage == 'payload': report['logical_payload_bytes'] = 16384
    elif damage == 'name': report['netlist'] = '../unrelated.json'
    elif damage == 'mapping': report['mapping'] = 'abc'
    elif damage == 'log': log.write_text('ABC: Aborted (core dumped)\n')
    (tmp_path / 'voice_cache_synthesis.json').write_text(json.dumps(report))
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid voice-cache evidence reached routing')
    monkeypatch.setattr(build, 'Tool', unexpected)
    if damage == 'valid':
        assert build.inventory() == report
    else:
        with pytest.raises(RuntimeError, match='stale or incompatible voice-cache'):
            build.route()


@pytest.mark.parametrize('damage', ['width', 'register', 'write_mode', 'clock', 'reset', 'select', 'address', 'initial', 'count', 'valid'])
def test_mapped_ram_model_rejects_unsupported_profile(damage):
    params, pins = {}, {}
    for port in ('A', 'B'):
        params.update({f'DATA_WIDTH_{port}': '1001', f'REGMODE_{port}': 'NOREG',
            f'WRITEMODE_{port}': 'NORMAL', f'CLK{port}MUX': f'CLK{port}', f'CSDECODE_{port}': '0b000'})
        pins.update({f'CLK{port}': [2], f'RST{port}': ['0'], f'OCE{port}': ['1']})
        for bit in range(3):
            pins[f'CS{port}{bit}'] = ['0']
            pins[f'AD{port}{bit}'] = ['0']
    params['INITVAL_00'] = '0' * 320
    cell = {'type': 'DP16KD', 'parameters': params, 'connections': pins}
    top = {'ports': {'clk': {'bits': [2]}}, 'cells': {str(i): deepcopy(cell) for i in range(15)}}
    p, c = top['cells']['0']['parameters'], top['cells']['0']['connections']
    if damage == 'width': p['DATA_WIDTH_A'] = '10010'
    elif damage == 'register': p['REGMODE_A'] = 'OUTREG'
    elif damage == 'write_mode': p['WRITEMODE_B'] = 'WRITETHROUGH'
    elif damage == 'clock': c['CLKB'] = [3]
    elif damage == 'reset': c['RSTA'] = [4]
    elif damage == 'select': c['CSA0'] = ['1']
    elif damage == 'address': c['ADB2'] = [5]
    elif damage == 'initial': p['INITVAL_00'] = '1'
    elif damage == 'count': del top['cells']['14']
    if damage == 'valid':
        mapping.validate_ram_mapping(top)
    else:
        with pytest.raises(AssertionError):
            mapping.validate_ram_mapping(top)
