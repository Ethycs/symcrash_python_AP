"""Synthesize or route the DMA burst engine, preserving source-hashed evidence.

Uses native tools or the isolated WSL setup in the HDL README. Routing is
experimental out-of-context, at 100 MHz; no board/PHY timing is implied.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/symcrash_verilog/dma_burst.v'
DIRECTORY = ROOT / 'build/hdl-tests'
MAX_CYCLES = 40000


class Tool:
    def __init__(self, name):
        self.wsl = None
        if shutil.which(name):
            self.command = [name]
            return
        package = 'yosys' if name == 'yosys' else 'nextpnr'
        prefix = ROOT / 'build/hdl-tools' / package
        if os.name != 'nt' or not shutil.which('wsl') or not (prefix / 'usr/bin' / name).is_file():
            raise RuntimeError(f'Install {name}; see src/symcrash_verilog/README.md')
        self.wsl = ['wsl', '-d', os.environ.get('SYMCRASH_HDL_WSL', 'Ubuntu-24.04'), '--']
        self.command = self.wsl + ['env', 'LD_LIBRARY_PATH=' + self.path(prefix / 'usr/lib/x86_64-linux-gnu')]
        if name == 'yosys':
            self.command += ['YOSYS_ABC=' + self.path(prefix / 'usr/bin/yosys-abc')]
        self.command += [self.path(prefix / 'usr/bin' / name)]

    def path(self, path):
        value = Path(path).resolve().as_posix()
        if self.wsl and value[1:2] == ':':
            value = '/mnt/' + value[0].lower() + value[2:]
        return value

    def run(self, args):
        return subprocess.run(self.command + args, cwd=ROOT, capture_output=True, text=True, timeout=180)

    def run_to_log(self, args, log, timeout=600):
        """Keep long placement output observable, including a terminal timeout."""
        command = self.command + args
        with log.open('w', encoding='utf-8') as output:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                        text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                output.write(f'\nTool execution timed out after {timeout} seconds.\n')
                output.flush()
                raise
        return subprocess.CompletedProcess(command, result.returncode,
                                           stdout=log.read_text(encoding='utf-8'), stderr='')

    def version(self, option):
        result = self.run([option])
        result.check_returncode()
        return (result.stdout + result.stderr).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthesize():
    tool = Tool('yosys')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    source_hash = digest(SOURCE)
    stem = f'dma_burst_{source_hash[:12]}'
    netlist = DIRECTORY / (stem + '_ecp5.json')
    script = DIRECTORY / (stem + '_ecp5.ys')
    log = DIRECTORY / (stem + '_ecp5.log')
    script.write_text(f'read_verilog -sv "{tool.path(SOURCE)}"\n'
                      f'chparam -set MAX_CYCLES {MAX_CYCLES} dma_burst\n'
                      f'synth_ecp5 -top dma_burst -json "{tool.path(netlist)}"\n'
                      'check -assert\nstat\n', encoding='utf-8')
    result = tool.run(['-s', tool.path(script)])
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'DMA synthesis failed; see {log}')
    cells = Counter(cell['type'] for cell in json.loads(netlist.read_text())['modules']['dma_burst']['cells'].values())
    report = {'source': SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': source_hash,
              'tool': tool.version('-V'), 'max_cycles': MAX_CYCLES,
              'netlist': netlist.name, 'netlist_sha256': digest(netlist),
              'cells': dict(sorted(cells.items())), 'payload_staging_bytes': 32,
              'physical_block_ram_bytes': cells['DP16KD']*18432//8,
              'scope': 'Standalone DMA burst engine; no PHY, priority queue or staging-buffer allocation'}
    (DIRECTORY / 'dma_burst_synthesis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


def inventory():
    report = json.loads((DIRECTORY / 'dma_burst_synthesis.json').read_text())
    source_hash = digest(SOURCE)
    expected = f'dma_burst_{source_hash[:12]}_ecp5.json'
    netlist = DIRECTORY / expected
    if (report.get('source') != SOURCE.relative_to(ROOT).as_posix()
            or report.get('source_sha256') != source_hash or report.get('max_cycles') != MAX_CYCLES
            or report.get('netlist') != expected or not netlist.is_file()
            or report.get('netlist_sha256') != digest(netlist)):
        raise RuntimeError('Missing, stale or incompatible DMA synthesis; rerun pixi run synth-hdl-dma')
    return report


def route(seeds=(1, 2, 3)):
    synthesis = inventory()  # Reject stale artifacts before tool discovery.
    tool = Tool('nextpnr-ecp5')
    report = {'source_sha256': synthesis['source_sha256'], 'netlist_sha256': synthesis['netlist_sha256'],
              'max_cycles': MAX_CYCLES, 'tool': tool.version('--version'),
              'device': 'ECP5-25K', 'package': 'CSFBGA285', 'speed_grade': 6, 'clock_mhz': 100,
              'scope': 'Experimental out-of-context; excludes board I/O and global clock routing', 'runs': []}
    for seed in seeds:
        stem = f"dma_burst_{synthesis['source_sha256'][:12]}_seed{seed}"
        timing, routed, log = (DIRECTORY / (stem + suffix) for suffix in ('_timing.json', '_routed.json', '_route.log'))
        if timing.exists():
            timing.unlink()
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
        (DIRECTORY / 'dma_burst_timing_inventory.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
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
