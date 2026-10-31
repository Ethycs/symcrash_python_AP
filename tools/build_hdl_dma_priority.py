"""Synthesize and route the 256-descriptor scheduler with its actual DMA engine."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest
from tools.check_hdl_dma_priority import source_hashes, fingerprint, HDL

DIRECTORY = ROOT / 'build/hdl-tests'
MAX_CYCLES = 40000


def synthesize():
    tool = Tool('yosys')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    key = fingerprint()
    stem = f'dma_priority_{key[:12]}'
    netlist, script, log = (DIRECTORY / (stem + suffix) for suffix in ('_ecp5.json', '_ecp5.ys', '_ecp5.log'))
    script.write_text('\n'.join(f'read_verilog -sv "{tool.path(HDL / name)}"' for name in source_hashes()) + '\n'
                      f'chparam -set MAX_CYCLES {MAX_CYCLES} dma_priority\n'
                      f'synth_ecp5 -top dma_priority -json "{tool.path(netlist)}"\n'
                      'check -assert\nstat\n', encoding='utf-8')
    result = tool.run(['-s', tool.path(script)])
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode: raise RuntimeError(f'Priority DMA synthesis failed; see {log}')
    cells = Counter(cell['type'] for cell in json.loads(netlist.read_text())['modules']['dma_priority']['cells'].values())
    report = {'sources': source_hashes(), 'fingerprint': key, 'tool': tool.version('-V'),
              'max_cycles': MAX_CYCLES, 'netlist': netlist.name, 'netlist_sha256': digest(netlist),
              'cells': dict(sorted(cells.items())), 'descriptor_bytes': 2048, 'link_bytes': 256,
              'burst_fifo_bytes': 32, 'physical_block_ram_bytes': cells['DP16KD']*18432//8,
              'scope': 'DMA priority queue plus burst engine; no PHY, staging ownership or aggregate admission'}
    (DIRECTORY / 'dma_priority_synthesis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


def inventory():
    report = json.loads((DIRECTORY / 'dma_priority_synthesis.json').read_text())
    key = fingerprint()
    expected = f'dma_priority_{key[:12]}_ecp5.json'
    netlist = DIRECTORY / expected
    if (report.get('sources') != source_hashes() or report.get('fingerprint') != key
            or report.get('max_cycles') != MAX_CYCLES or report.get('netlist') != expected
            or not netlist.is_file() or report.get('netlist_sha256') != digest(netlist)):
        raise RuntimeError('Missing, stale or incompatible priority DMA synthesis; rerun pixi run synth-hdl-dma-priority')
    return report


def route(seeds=(1, 2, 3)):
    synthesis = inventory()
    tool = Tool('nextpnr-ecp5')
    report = {'sources': synthesis['sources'], 'fingerprint': synthesis['fingerprint'],
              'netlist_sha256': synthesis['netlist_sha256'], 'max_cycles': MAX_CYCLES,
              'tool': tool.version('--version'), 'device': 'ECP5-25K', 'package': 'CSFBGA285',
              'speed_grade': 6, 'clock_mhz': 100,
              'scope': 'Experimental out-of-context; excludes board I/O and global clock routing', 'runs': []}
    for seed in seeds:
        stem = f"dma_priority_{synthesis['fingerprint'][:12]}_seed{seed}"
        timing, routed, log = (DIRECTORY / (stem + suffix) for suffix in ('_timing.json', '_routed.json', '_route.log'))
        if timing.exists(): timing.unlink()
        result = tool.run(['--25k', '--package', 'CSFBGA285', '--speed', '6', '--out-of-context',
                          '--freq', '100', '--seed', str(seed), '--json', tool.path(DIRECTORY / synthesis['netlist']),
                          '--report', tool.path(timing), '--write', tool.path(routed)])
        log.write_text(result.stdout + result.stderr, encoding='utf-8')
        measured = json.loads(timing.read_text()) if timing.is_file() else {}
        clock = measured.get('fmax', {}).get('clk', {})
        passed = result.returncode == 0 and clock.get('constraint') == 100 and clock.get('achieved', 0) >= 100
        record = {'seed': seed, 'exit_code': result.returncode, 'passed': passed,
                  'fmax': measured.get('fmax', {}), 'utilization': measured.get('utilization', {})}
        report['runs'].append(record)
        print(json.dumps({k: record[k] for k in ('seed', 'exit_code', 'passed', 'fmax')}), flush=True)
        (DIRECTORY / 'dma_priority_timing_inventory.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
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
