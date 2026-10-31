"""Synthesize/route coefficient ownership control; payload RAM is external."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest
from tools.hdl_mapping import MAPPING, NETLIST_TAG, PRELUDE, evidence as mapping_evidence, matches as mapping_matches

SOURCE = ROOT / 'src/symcrash_verilog/coeff_buffer_owner.v'
DIRECTORY = ROOT / 'build/hdl-tests'
BASE = 0x8000
SLOTS = (2, 4, 8)


def synthesize():
    tool = Tool('yosys')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    source_hash = digest(SOURCE)
    report = {'source_sha256': source_hash, 'tool': tool.version('-V'), 'mapping': MAPPING, 'base': BASE,
              'scope': 'Ownership control only; shared DMA, payload memory, decoder and PHY excluded', 'configurations': []}
    for k in SLOTS:
        stem = f'coeff_buffer_owner_{source_hash[:12]}_{NETLIST_TAG}_k{k}'
        netlist, script, log = (DIRECTORY / (stem + suffix) for suffix in ('_ecp5.json', '_ecp5.ys', '_ecp5.log'))
        script.write_text(PRELUDE + f'read_verilog -sv "{tool.path(SOURCE)}"\n'
                          f'chparam -set K {k} -set BASE {BASE} coeff_buffer_owner\n'
                          f'synth_ecp5 -abc9 -top coeff_buffer_owner -json "{tool.path(netlist)}"\n'
                          'check -assert\nstat\n', encoding='utf-8')
        result = tool.run(['-s', tool.path(script)])
        log.write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode: raise RuntimeError(f'Ownership synthesis failed; see {log}')
        cells = Counter(cell['type'] for cell in json.loads(netlist.read_text())['modules']['coeff_buffer_owner']['cells'].values())
        record = {'K': k, 'netlist': netlist.name, 'netlist_sha256': digest(netlist),
                  **mapping_evidence(log), 'cells': dict(sorted(cells.items())), 'external_payload_bytes': 2*64*k*2}
        report['configurations'].append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    (DIRECTORY / 'coeff_buffer_owner_synthesis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def inventory():
    report = json.loads((DIRECTORY / 'coeff_buffer_owner_synthesis.json').read_text())
    source_hash = digest(SOURCE)
    valid = report.get('source_sha256') == source_hash and report.get('base') == BASE
    valid = valid and report.get('mapping') == MAPPING
    rows = report.get('configurations', [])
    valid = valid and len(rows) == 3 and {row.get('K') for row in rows} == set(SLOTS)
    for row in rows:
        expected = f"coeff_buffer_owner_{source_hash[:12]}_{NETLIST_TAG}_k{row.get('K')}_ecp5.json"
        netlist = DIRECTORY / expected
        valid = valid and row.get('netlist') == expected and netlist.is_file() and row.get('netlist_sha256') == digest(netlist)
        valid = valid and mapping_matches(row, DIRECTORY, expected.removesuffix('_ecp5.json'))
    if not valid:
        raise RuntimeError('Missing, stale or incompatible ownership synthesis; rerun pixi run synth-hdl-coeff-buffers')
    return report


def route(seeds=(1, 2, 3)):
    synthesis = inventory()
    tool = Tool('nextpnr-ecp5')
    report = {'source_sha256': synthesis['source_sha256'], 'base': BASE,
              'mapping': MAPPING, 'tool': tool.version('--version'), 'device': 'ECP5-25K', 'package': 'CSFBGA285',
              'speed_grade': 6, 'clock_mhz': 100,
              'scope': 'Ownership control only; out-of-context excludes I/O and global clocks', 'runs': []}
    for row in synthesis['configurations']:
        for seed in seeds:
            stem = f"coeff_buffer_owner_{synthesis['source_sha256'][:12]}_{NETLIST_TAG}_k{row['K']}_seed{seed}"
            timing, routed, log = (DIRECTORY / (stem + suffix) for suffix in ('_timing.json', '_routed.json', '_route.log'))
            if timing.exists(): timing.unlink()
            result = tool.run_to_log(['--25k', '--package', 'CSFBGA285', '--speed', '6', '--out-of-context',
                              '--freq', '100', '--seed', str(seed), '--json', tool.path(DIRECTORY / row['netlist']),
                              '--report', tool.path(timing), '--write', tool.path(routed)], log)
            log.write_text(result.stdout + result.stderr, encoding='utf-8')
            measured = json.loads(timing.read_text()) if timing.is_file() else {}
            clock = measured.get('fmax', {}).get('clk', {})
            passed = result.returncode == 0 and clock.get('constraint') == 100 and clock.get('achieved', 0) >= 100
            record = {'K': row['K'], 'seed': seed, 'exit_code': result.returncode, 'passed': passed,
                      'netlist_sha256': row['netlist_sha256'],
                      'fmax': measured.get('fmax', {}), 'utilization': measured.get('utilization', {})}
            report['runs'].append(record)
            print(json.dumps({k: record[k] for k in ('K', 'seed', 'exit_code', 'passed', 'fmax')}), flush=True)
            (DIRECTORY / 'coeff_buffer_owner_timing_inventory.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(report['scope'], flush=True)
    return all(r['passed'] for r in report['runs'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('synth', 'route'))
    parser.add_argument('--seeds', type=int, nargs='+', default=(1, 2, 3))
    args = parser.parse_args()
    if args.action == 'synth':
        synthesize()
        return 0
    return 0 if route(args.seeds) else 1


if __name__ == '__main__':
    sys.exit(main())
