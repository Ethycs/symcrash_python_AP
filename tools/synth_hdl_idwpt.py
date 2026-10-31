"""Synthesize the current CCS decoder for ECP5; report resources, not timing.

Requires native Yosys, or the isolated WSL tools in the HDL README. Logs,
netlists and a source-hashed inventory are written to build/hdl-tests.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/symcrash_verilog/ccs_idwpt.v'


def synthesize(slots=(2, 4, 8)):
    wsl = None
    if shutil.which('yosys'):
        command = ['yosys']
    elif os.name == 'nt' and shutil.which('wsl'):
        prefix = ROOT / 'build/hdl-tools/yosys'
        if not (prefix / 'usr/bin/yosys').is_file():
            raise RuntimeError('Install Yosys; see src/symcrash_verilog/README.md')
        wsl = ['wsl', '-d', os.environ.get('SYMCRASH_HDL_WSL', 'Ubuntu-24.04'), '--']
        local = prefix.as_posix()
        local = '/mnt/' + local[0].lower() + local[2:]
        command = wsl + ['env', 'LD_LIBRARY_PATH=' + local + '/usr/lib/x86_64-linux-gnu',
                        'YOSYS_ABC=' + local + '/usr/bin/yosys-abc', local + '/usr/bin/yosys']
    else:
        raise RuntimeError('Install Yosys; see src/symcrash_verilog/README.md')

    def tool_path(path):
        value = path.resolve().as_posix()
        if wsl and value[1:2] == ':':
            value = '/mnt/' + value[0].lower() + value[2:]
        return value

    directory = ROOT / 'build/hdl-tests'
    directory.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(command + ['-V'], check=True, capture_output=True,
                             text=True, timeout=30).stdout.strip()
    report = {'source': SOURCE.relative_to(ROOT).as_posix(),
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'tool': version, 'headroom_bits': 2, 'configurations': [],
              'scope': 'ECP5 synthesis only; no packing, placement, routing or timing proof'}
    for k in slots:
        netlist = directory / f'ccs_idwpt_k{k}_ecp5.json'
        script = directory / f'ccs_idwpt_k{k}_ecp5.ys'
        log = directory / f'ccs_idwpt_k{k}_ecp5.log'
        script.write_text(f'read_verilog -sv "{tool_path(SOURCE)}"\n'
                          f'chparam -set K {k} ccs_idwpt\n'
                          f'synth_ecp5 -top ccs_idwpt -json "{tool_path(netlist)}"\n'
                          'check -assert\nstat\n', encoding='utf-8')
        result = subprocess.run(command + ['-s', tool_path(script)], cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        log.write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Yosys failed for K={k}; see {log}')
        cells = Counter(cell['type'] for cell in json.loads(netlist.read_text())
                        ['modules']['ccs_idwpt']['cells'].values())
        lanes = min(k, 4)
        record = {'K': k, 'cells': dict(sorted(cells.items())),
                  'netlist_sha256': hashlib.sha256(netlist.read_bytes()).hexdigest(),
                  'logical_array_bytes': 2*64*lanes*2 + 16*8,
                  'physical_block_ram_bytes': cells['DP16KD']*18432//8,
                  'cycles_without_stalls': (64+247)*(k//lanes)+16}
        report['configurations'].append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    destination = directory / 'ccs_idwpt_synthesis.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'{version}\nInventory: {destination}\n{report["scope"]}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slots', nargs='+', type=int, choices=(2, 4, 8), default=(2, 4, 8))
    args = parser.parse_args()
    synthesize(args.slots)


if __name__ == '__main__':
    main()
