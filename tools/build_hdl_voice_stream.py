"""Map and route one voice-state cache plus one shared runtime decoder."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest
from tools.hdl_mapping import MAPPING, PRELUDE, evidence, matches

HDL = ROOT / 'src/symcrash_verilog'
DIRECTORY = ROOT / 'build/hdl-tests'
SOURCE_NAMES = ('voice_cache_stream.v', 'voice_cache_renderer.v', 'voice_state_cache.v', 'ccs_idwpt_runtime.v')


def hashes():
    return {name: digest(HDL / name) for name in SOURCE_NAMES}


def fingerprint():
    return hashlib.sha256(json.dumps(hashes(), sort_keys=True).encode()).hexdigest()


def stem():
    return 'voice_stream_' + fingerprint()[:12] + '_abc9'


def require_default_headroom():
    text = (HDL / 'voice_cache_stream.v').read_text(encoding='utf-8')
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    if not re.search(r'\bmodule\s+voice_cache_stream\s*#\(\s*parameter(?:\s+integer)?\s+HEADROOM_BITS\s*=\s*2\s*\)', text):
        raise RuntimeError('Voice-stream timing requires declared default HEADROOM_BITS=2')


def synthesize():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Use the qualified default-parameter synthesis recipe. An unnecessary
    # chparam changes mapping, so prove the declared profile before mapping.
    require_default_headroom()
    tool, sources, key, name = Tool('yosys'), hashes(), fingerprint(), stem()
    script, log, netlist = (DIRECTORY / (name + suffix) for suffix in ('_ecp5.ys', '_ecp5.log', '_ecp5.json'))
    script.write_text(PRELUDE + 'read_verilog -sv ' + ' '.join('"' + tool.path(HDL / n) + '"' for n in SOURCE_NAMES)
        + '\n'
        + f'synth_ecp5 -abc9 -top voice_cache_stream -json "{tool.path(netlist)}"\ncheck -assert\nstat\n', encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    result.check_returncode()
    assert hashes() == sources
    cells = Counter(c['type'] for c in json.loads(netlist.read_text())['modules']['voice_cache_stream']['cells'].values())
    report = {'sources': sources, 'fingerprint': key, 'runner_sha256': digest(Path(__file__)),
        'mapping': MAPPING, 'headroom': 2, 'tool': tool.version('-V'), 'netlist': netlist.name,
        'netlist_sha256': digest(netlist), **evidence(log), 'cells': dict(sorted(cells.items())),
        'logical_page_bytes': 2048, 'pages': 16, 'physical_bram_bytes': cells['DP16KD'] * 2304,
        'physical_dpr_bytes': cells['TRELLIS_DPR16X4'] * 8,
        'scope': 'One opaque voice cache and one decoder; DMA/window, operators, scheduler, mix and board are separate gates.'}
    (DIRECTORY / 'voice_stream_synthesis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def inventory():
    report = json.loads((DIRECTORY / 'voice_stream_synthesis.json').read_text())
    name = stem()
    netlist = DIRECTORY / (name + '_ecp5.json')
    valid = (report.get('sources') == hashes() and report.get('fingerprint') == fingerprint()
        and report.get('mapping') == MAPPING and report.get('headroom') == 2
        and report.get('logical_page_bytes') == 2048 and report.get('pages') == 16
        and report.get('netlist') == netlist.name and netlist.is_file()
        and digest(netlist) == report.get('netlist_sha256') and matches(report, DIRECTORY, name))
    if not valid:
        raise RuntimeError('Missing, stale or incompatible voice-stream synthesis')
    return report


def route():
    synthesis, tool = inventory(), Tool('nextpnr-ecp5')
    rows = []
    report = {'sources': synthesis['sources'], 'fingerprint': synthesis['fingerprint'],
        'netlist_sha256': synthesis['netlist_sha256'], 'mapping': MAPPING, 'headroom': 2,
        'device': 'ECP5-25K', 'package': 'CSFBGA285', 'speed_grade': 6, 'clock_mhz': 100,
        'tool': tool.version('--version'), 'runs': rows,
        'scope': 'Out-of-context cache plus decoder; excludes DMA/window and board placement.'}
    for seed in (1, 2, 3):
        name = stem() + f'_seed{seed}'
        log, timing = (DIRECTORY / (name + suffix) for suffix in ('_route.log', '_timing.json'))
        if timing.exists():
            timing.unlink()
        result = tool.run_to_log(['--25k', '--package', 'CSFBGA285', '--speed', '6',
            '--out-of-context', '--freq', '100', '--seed', str(seed), '--json',
            tool.path(DIRECTORY / synthesis['netlist']), '--report', tool.path(timing)], log)
        raw = json.loads(timing.read_text()) if timing.exists() else {}
        clock = raw.get('fmax', {}).get('clk', {})
        row = {'seed': seed, 'exit_code': result.returncode,
            'passed': result.returncode == 0 and clock.get('constraint') == 100 and clock.get('achieved', 0) >= 100,
            'fmax': raw.get('fmax', {}), 'utilization': raw.get('utilization', {}), 'route_log_sha256': digest(log)}
        if timing.exists():
            row['timing_sha256'] = digest(timing)
        assert inventory()['netlist_sha256'] == synthesis['netlist_sha256']
        rows.append(row)
        (DIRECTORY / 'voice_stream_timing.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(row), flush=True)
    return all(row['passed'] for row in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('synth', 'route'))
    args = parser.parse_args()
    if args.action == 'synth':
        print(json.dumps(synthesize()), flush=True)
    else:
        sys.exit(0 if route() else 1)
