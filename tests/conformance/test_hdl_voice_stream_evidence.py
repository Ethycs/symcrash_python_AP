"""The shared renderer must route only matching, fully verified synthesis."""
import json
import pytest
from tools import build_hdl_voice_stream as build


@pytest.mark.parametrize('declaration,accepted', [
    ('module voice_cache_stream #(parameter HEADROOM_BITS=2) (); endmodule', True),
    ('module voice_cache_stream #(parameter integer HEADROOM_BITS = 2) (); endmodule', True),
    ('module voice_cache_stream #(parameter HEADROOM_BITS=0) (); endmodule', False),
    ('module voice_cache_stream #(parameter HEADROOM_BITS=15) (); endmodule', False),
    ('module voice_cache_stream #(parameter HEADROOM_BITS=OTHER) (); endmodule', False),
    ('module voice_cache_stream (); endmodule', False),
    ('// module voice_cache_stream #(parameter HEADROOM_BITS=2)\n'
     'module voice_cache_stream #(parameter HEADROOM_BITS=0) (); endmodule', False),
    ('/* module voice_cache_stream #(parameter HEADROOM_BITS=2) */\n'
     'module voice_cache_stream #(parameter HEADROOM_BITS=15) (); endmodule', False),
])
def test_voice_stream_declared_timing_profile(tmp_path, monkeypatch, declaration, accepted):
    monkeypatch.setattr(build, 'HDL', tmp_path)
    (tmp_path / 'voice_cache_stream.v').write_text(declaration)
    if accepted:
        build.require_default_headroom()
    else:
        with pytest.raises(RuntimeError, match='declared default HEADROOM_BITS=2'):
            build.require_default_headroom()


@pytest.mark.parametrize('damage', ['cache', 'decoder', 'renderer', 'top', 'netlist', 'missing', 'name', 'mapping', 'log', 'headroom', 'page', 'pages', 'valid'])
def test_voice_stream_synthesis_evidence(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(build, 'HDL', tmp_path)
    monkeypatch.setattr(build, 'DIRECTORY', tmp_path)
    for name in build.SOURCE_NAMES:
        (tmp_path / name).write_text('module original; endmodule\n')
    name = build.stem()
    netlist, log = (tmp_path / (name + suffix) for suffix in ('_ecp5.json', '_ecp5.log'))
    netlist.write_text('{}\n')
    log.write_text('ABC: + &verify\nABC: Networks are equivalent.\n')
    report = {'sources': build.hashes(), 'fingerprint': build.fingerprint(), 'mapping': build.MAPPING,
        'headroom': 2, 'logical_page_bytes': 2048, 'pages': 16, 'netlist': netlist.name,
        'netlist_sha256': build.digest(netlist), **build.evidence(log)}
    changed = {'cache': 'voice_state_cache.v', 'decoder': 'ccs_idwpt_runtime.v',
        'renderer': 'voice_cache_renderer.v', 'top': 'voice_cache_stream.v'}
    if damage in changed: (tmp_path / changed[damage]).write_text('module changed; endmodule\n')
    elif damage == 'netlist': netlist.write_text('{"changed":true}\n')
    elif damage == 'missing': netlist.unlink()
    elif damage == 'name': report['netlist'] = '../unrelated.json'
    elif damage == 'mapping': report['mapping'] = 'abc'
    elif damage == 'log': log.write_text('ABC: Aborted (core dumped)\n')
    elif damage == 'headroom': report['headroom'] = 0
    elif damage == 'page': report['logical_page_bytes'] = 1024
    elif damage == 'pages': report['pages'] = 8
    (tmp_path / 'voice_stream_synthesis.json').write_text(json.dumps(report))
    def unexpected(*args, **kwargs):
        pytest.fail('Stale voice-stream synthesis reached routing')
    monkeypatch.setattr(build, 'Tool', unexpected)
    if damage == 'valid':
        assert build.inventory() == report
    else:
        with pytest.raises(RuntimeError, match='stale or incompatible voice-stream'):
            build.route()
