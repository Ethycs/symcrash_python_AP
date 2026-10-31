"""Synthesize/route the combined ownership, payload RAM, feeder, CCS and fallback."""
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
from tools.check_hdl_ccs_stream import HDL, SOURCES, source_hashes, fingerprint

DIRECTORY = ROOT / 'build/hdl-tests'
BASE = 0x8000
HEADROOM = 2
SLOTS = (2, 4, 8)


def synthesize():
    tool = Tool('yosys')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    key = fingerprint()
    report = {'sources': source_hashes(), 'fingerprint': key, 'tool': tool.version('-V'), 'mapping': MAPPING,
              'base': BASE, 'headroom': HEADROOM,
              'scope': 'Ownership, payload RAM, coefficient feeder, decoder and replay; shared DMA and PHY excluded', 'configurations': []}
    for k in SLOTS:
        stem = f'ccs_stream_{key[:12]}_{NETLIST_TAG}_k{k}'
        netlist, script, log = (DIRECTORY / (stem + suffix) for suffix in ('_ecp5.json', '_ecp5.ys', '_ecp5.log'))
        script.write_text(PRELUDE + '\n'.join(f'read_verilog -sv "{tool.path(HDL / name)}"' for name in SOURCES) + '\n'
                          f'chparam -set K {k} -set HEADROOM_BITS {HEADROOM} -set BASE {BASE} ccs_stream\n'
                          f'synth_ecp5 -abc9 -top ccs_stream -json "{tool.path(netlist)}"\n'
                          'check -assert\nstat\n', encoding='utf-8')
        result = tool.run(['-s', tool.path(script)])
        log.write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode: raise RuntimeError(f'CCS stream synthesis failed; see {log}')
        cells = Counter(cell['type'] for cell in json.loads(netlist.read_text())['modules']['ccs_stream']['cells'].values())
        record = {'K': k, 'netlist': netlist.name, 'netlist_sha256': digest(netlist),
                  **mapping_evidence(log), 'cells': dict(sorted(cells.items())), 'payload_bytes': 2*64*k*2,
                  'feeder_fifo_bytes': 32, 'replay_bytes': 128,
                  'physical_block_ram_bytes': cells['DP16KD']*18432//8,
                  'physical_distributed_ram_bytes': cells['TRELLIS_DPR16X4']*64//8}
        report['configurations'].append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    (DIRECTORY / 'ccs_stream_synthesis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def inventory():
    report = json.loads((DIRECTORY / 'ccs_stream_synthesis.json').read_text())
    key = fingerprint()
    valid = report.get('sources') == source_hashes() and report.get('fingerprint') == key
    valid = valid and report.get('base') == BASE and report.get('headroom') == HEADROOM
    valid = valid and report.get('mapping') == MAPPING
    rows = report.get('configurations', [])
    valid = valid and len(rows) == 3 and {row.get('K') for row in rows} == set(SLOTS)
    for row in rows:
        expected = f"ccs_stream_{key[:12]}_{NETLIST_TAG}_k{row.get('K')}_ecp5.json"
        netlist = DIRECTORY / expected
        valid = valid and row.get('netlist') == expected and netlist.is_file() and row.get('netlist_sha256') == digest(netlist)
        valid = valid and mapping_matches(row, DIRECTORY, expected.removesuffix('_ecp5.json'))
    if not valid:
        raise RuntimeError('Missing, stale or incompatible CCS stream synthesis; rerun pixi run synth-hdl-ccs-stream')
    return report


def route(seeds=(1, 2, 3)):
    synthesis = inventory()
    tool = Tool('nextpnr-ecp5')
    report = {'sources': synthesis['sources'], 'fingerprint': synthesis['fingerprint'], 'base': BASE, 'headroom': HEADROOM,
              'mapping': MAPPING, 'tool': tool.version('--version'), 'device': 'ECP5-25K', 'package': 'CSFBGA285',
              'speed_grade': 6, 'clock_mhz': 100,
              'scope': 'Combined stream, no shared DMA/PHY; out-of-context excludes I/O and global clocks', 'runs': []}
    for row in synthesis['configurations']:
        for seed in seeds:
            stem = f"ccs_stream_{synthesis['fingerprint'][:12]}_{NETLIST_TAG}_k{row['K']}_seed{seed}"
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
            (DIRECTORY / 'ccs_stream_timing_inventory.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
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
