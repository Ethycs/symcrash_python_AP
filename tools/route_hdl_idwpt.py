"""Check standalone ECP5-25K decoder timing at 100 MHz, speed grade 6.

Requires a current `pixi run synth-hdl` inventory and native nextpnr-ecp5,
or the isolated WSL tools described in the HDL README. This experimental
out-of-context probe excludes board I/O and global clock routing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'build/hdl-tests'


def route(slots=(2, 4, 8), seeds=(1,)):
    inventory = json.loads((DIRECTORY / 'ccs_idwpt_synthesis.json').read_text())
    source_hash = hashlib.sha256((ROOT / inventory['source']).read_bytes()).hexdigest()
    if source_hash != inventory['source_sha256'] or inventory['headroom_bits'] != 2:
        raise RuntimeError('Stale or incompatible synthesis inventory; rerun pixi run synth-hdl')
    configurations = {record['K']: record for record in inventory['configurations']}
    for k in slots:
        netlist = DIRECTORY / f'ccs_idwpt_k{k}_ecp5.json'
        if (k not in configurations or not netlist.is_file()
                or hashlib.sha256(netlist.read_bytes()).hexdigest() != configurations[k].get('netlist_sha256')):
            raise RuntimeError(f'Missing or stale K={k} netlist; rerun pixi run synth-hdl')

    wsl = None
    if shutil.which('nextpnr-ecp5'):
        command = ['nextpnr-ecp5']
    elif os.name == 'nt' and shutil.which('wsl'):
        prefix = ROOT / 'build/hdl-tools/nextpnr'
        if not (prefix / 'usr/bin/nextpnr-ecp5').is_file():
            raise RuntimeError('Install nextpnr-ecp5; see src/symcrash_verilog/README.md')
        wsl = ['wsl', '-d', os.environ.get('SYMCRASH_HDL_WSL', 'Ubuntu-24.04'), '--']
        local = prefix.as_posix()
        local = '/mnt/' + local[0].lower() + local[2:]
        command = wsl + ['env', 'LD_LIBRARY_PATH=' + local + '/usr/lib/x86_64-linux-gnu',
                        local + '/usr/bin/nextpnr-ecp5']
    else:
        raise RuntimeError('Install nextpnr-ecp5; see src/symcrash_verilog/README.md')

    def tool_path(path):
        value = path.resolve().as_posix()
        if wsl and value[1:2] == ':':
            value = '/mnt/' + value[0].lower() + value[2:]
        return value

    version = subprocess.run(command + ['--version'], check=True, capture_output=True,
                             text=True, timeout=30)
    report = {'source_sha256': source_hash, 'tool': (version.stdout + version.stderr).strip(),
              'device': 'ECP5-25K', 'package': 'CSFBGA285', 'speed_grade': 6,
              'clock_mhz': 100, 'headroom_bits': 2,
              'scope': 'Experimental out-of-context; excludes board I/O and global clock routing',
              'runs': []}
    for k in slots:
        for seed in seeds:
            name = f'ccs_idwpt_k{k}_seed{seed}'
            timing = DIRECTORY / (name + '_timing.json')
            routed = DIRECTORY / (name + '_routed.json')
            log = DIRECTORY / (name + '_route.log')
            # A failed invocation must not adopt an older report at this name.
            if timing.exists():
                timing.unlink()
            result = subprocess.run(command + ['--25k', '--package', 'CSFBGA285', '--speed', '6',
                '--out-of-context', '--freq', '100', '--seed', str(seed),
                '--json', tool_path(DIRECTORY / f'ccs_idwpt_k{k}_ecp5.json'),
                '--report', tool_path(timing), '--write', tool_path(routed)],
                cwd=ROOT, capture_output=True, text=True, timeout=180)
            log.write_text(result.stdout + result.stderr, encoding='utf-8')
            measured = json.loads(timing.read_text()) if timing.is_file() else {}
            clocks = measured.get('fmax', {})
            clock = clocks.get('clk', {})
            passed = (result.returncode == 0 and clock.get('constraint') == 100
                      and clock.get('achieved', 0) >= 100)
            record = {'K': k, 'seed': seed, 'exit_code': result.returncode, 'passed': passed,
                      'fmax': clocks, 'utilization': measured.get('utilization', {}),
                      'netlist_sha256': configurations[k]['netlist_sha256']}
            report['runs'].append(record)
            print(json.dumps({key: record[key] for key in ('K', 'seed', 'exit_code', 'passed', 'fmax')}), flush=True)
            (DIRECTORY / 'ccs_idwpt_timing_inventory.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(report['scope'], flush=True)
    return all(record['passed'] for record in report['runs'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slots', nargs='+', type=int, choices=(2, 4, 8), default=(2, 4, 8))
    parser.add_argument('--seeds', nargs='+', type=int, default=(1,))
    args = parser.parse_args()
    return 0 if route(args.slots, args.seeds) else 1


if __name__ == '__main__':
    sys.exit(main())
