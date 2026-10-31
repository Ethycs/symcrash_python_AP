"""Map and route the documented sixteen-page voice cache at 100 MHz."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest
from tools.hdl_mapping import MAPPING, PRELUDE, evidence, matches

SOURCE = ROOT / 'src/symcrash_verilog/voice_state_cache.v'
DIRECTORY = ROOT / 'build/hdl-tests'


def stem():
    return 'voice_cache_' + digest(SOURCE)[:12] + '_abc9'


def synthesize():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    tool = Tool('yosys')
    key, name = digest(SOURCE), stem()
    script, log, netlist = (DIRECTORY / (name + suffix)
                            for suffix in ('_ecp5.ys', '_ecp5.log', '_ecp5.json'))
    script.write_text(PRELUDE + f'read_verilog -sv "{tool.path(SOURCE)}"\n'
        f'synth_ecp5 -abc9 -top voice_state_cache -json "{tool.path(netlist)}"\n'
        'check -assert\nstat\n', encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    result.check_returncode()
    assert digest(SOURCE) == key
    cells = Counter(c['type'] for c in json.loads(netlist.read_text())
                    ['modules']['voice_state_cache']['cells'].values())
    report = {'source_sha256': key, 'runner_sha256': digest(Path(__file__)),
        'mapping': MAPPING, 'tool': tool.version('-V'), 'netlist': netlist.name,
        'netlist_sha256': digest(netlist), **evidence(log), 'cells': dict(sorted(cells.items())),
        'logical_page_bytes': 2048, 'pages': 16, 'logical_payload_bytes': 32768,
        'physical_bram_bytes': cells['DP16KD'] * 2304,
        'physical_dpr_bytes': cells['TRELLIS_DPR16X4'] * 8,
        'scope': 'Opaque voice cache only; full state layout, shared decoder, scheduler and board remain separate gates.'}
    (DIRECTORY / 'voice_cache_synthesis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def inventory():
    report = json.loads((DIRECTORY / 'voice_cache_synthesis.json').read_text())
    name = stem()
    netlist = DIRECTORY / (name + '_ecp5.json')
    valid = (report.get('source_sha256') == digest(SOURCE) and report.get('mapping') == MAPPING
        and report.get('logical_page_bytes') == 2048 and report.get('pages') == 16
        and report.get('logical_payload_bytes') == 32768 and report.get('netlist') == netlist.name
        and netlist.is_file() and digest(netlist) == report.get('netlist_sha256')
        and matches(report, DIRECTORY, name))
    if not valid:
        raise RuntimeError('Missing, stale or incompatible voice-cache synthesis')
    return report


def route():
    synthesis, tool = inventory(), Tool('nextpnr-ecp5')
    rows = []
    report = {'source_sha256': synthesis['source_sha256'], 'netlist_sha256': synthesis['netlist_sha256'],
        'mapping': MAPPING, 'device': 'ECP5-25K', 'package': 'CSFBGA285', 'speed_grade': 6,
        'clock_mhz': 100, 'tool': tool.version('--version'), 'runs': rows,
        'scope': 'Out-of-context voice cache; no combined DMA/decoder or board placement claim.'}
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
            'fmax': raw.get('fmax', {}), 'utilization': raw.get('utilization', {}),
            'route_log_sha256': digest(log)}
        if timing.exists():
            row['timing_sha256'] = digest(timing)
        assert inventory()['netlist_sha256'] == synthesis['netlist_sha256']
        rows.append(row)
        (DIRECTORY / 'voice_cache_timing.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
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
