"""Simulate current CCS HDL against independent rational and Python vectors.

Requires native iverilog/vvp, or the isolated Ubuntu tools documented in the
HDL README. An unavailable simulator is an error for this acceptance command.
"""
import argparse
from fractions import Fraction
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.audit_lifting_arithmetic import inverse

HDL = ROOT / 'src' / 'symcrash_verilog'


class SimulatorUnavailable(RuntimeError):
    pass


class Simulator:
    def __init__(self):
        self.wsl = None
        if shutil.which('iverilog') and shutil.which('vvp'):
            self.compiler, self.runtime = ['iverilog'], ['vvp']
            return
        prefix = ROOT / 'build' / 'hdl-tools' / 'iverilog'
        if os.name == 'nt' and shutil.which('wsl') and (prefix / 'usr/bin/iverilog').is_file():
            self.wsl = ['wsl', '-d', os.environ.get('SYMCRASH_HDL_WSL', 'Ubuntu-24.04'), '--']
            self.compiler = self.wsl + [self.path(prefix / 'usr/bin/iverilog'), '-B',
                self.path(prefix / 'usr/lib/x86_64-linux-gnu/ivl')]
            self.runtime = self.wsl + [self.path(prefix / 'usr/bin/vvp')]
            return
        raise SimulatorUnavailable('Install Icarus Verilog (iverilog and vvp); see src/symcrash_verilog/README.md')

    def path(self, path):
        absolute = Path(path).resolve().as_posix()
        if self.wsl and len(absolute) > 2 and absolute[1] == ':':
            return '/mnt/' + absolute[0].lower() + absolute[2:]
        return absolute

    def run(self, command):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise AssertionError(f'HDL command failed ({result.returncode}):\n{result.stdout}\n{result.stderr}')
        return result.stdout


def coefficient_cases(slots, exhaustive=True):
    result = [np.zeros((64, slots), dtype=np.int16),
              np.ones((64, slots), dtype=np.int16),
              np.full((64, slots), -1, dtype=np.int16),
              np.resize(np.array([-32768, 32767, 1, -3], dtype=np.int16), (64, slots))]
    positions = range(64*slots) if exhaustive else range(min(8, 64*slots))
    for position in positions:
        values = np.zeros(64*slots, dtype=np.int16)
        values[position] = 32767 if position%2 else -32768
        result.append(values.reshape(64, slots))
    rng = np.random.default_rng(0xCDF5300 + slots)
    for magnitude in (3, 8191, 32767):
        for _ in range(8):
            result.append(rng.integers(-magnitude, magnitude+1, (64, slots), dtype=np.int16))
    return result


def reference_audio(coefficients, headroom):
    slots = coefficients.shape[1]
    result = np.empty(64, dtype=np.int16)
    for slot in range(slots):
        result[slot::slots] = inverse(coefficients[:, slot], headroom)[0::slots]
    return result


def check(slots=4, headroom=2, stalls=False, exhaustive=True, simulator=None):
    from symcrash_python_ap.vm.hotpath_jit import idwpt_polyphase
    simulator = simulator or Simulator()
    cases = coefficient_cases(slots, exhaustive)
    expected = [reference_audio(c, headroom) for c in cases]
    for coefficients, oracle in zip(cases, expected):
        actual = np.empty(64, dtype=np.int16)
        idwpt_polyphase(coefficients.ravel(), actual, headroom)
        np.testing.assert_array_equal(actual, oracle)
    # Use a workspace-local folder, accessible to both native tools and WSL.
    directory = ROOT / 'build' / 'hdl-tests'
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='idwpt-', dir=directory) as temporary:
        temporary = Path(temporary)
        coeffs, outputs, executable = (temporary / name for name in ('coeffs.hex', 'expected.hex', 'simulation.vvp'))
        for path, values in [(coeffs, cases), (outputs, expected)]:
            path.write_text(''.join(f'{int(v)&0xffff:04x}\n' for v in np.asarray(values).flat), encoding='ascii')
        simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_ccs_idwpt',
            f'-Ptb_ccs_idwpt.K={slots}', f'-Ptb_ccs_idwpt.HEADROOM_BITS={headroom}',
            f'-Ptb_ccs_idwpt.CASES={len(cases)}', f'-Ptb_ccs_idwpt.STALLS={int(stalls)}',
            '-o', simulator.path(executable), simulator.path(HDL / 'ccs_idwpt.v'),
            simulator.path(HDL / 'tb_ccs_idwpt.v')])
        output = simulator.run(simulator.runtime + [simulator.path(executable),
            '+coeffs=' + simulator.path(coeffs), '+expected=' + simulator.path(outputs)])
        assert f'PASS cases={len(cases)} K={slots} headroom={headroom} stalls={int(stalls)}' in output, output
        return next(line for line in output.splitlines() if line.startswith('PASS '))


def check_rounding(simulator=None):
    simulator = simulator or Simulator()
    directory = ROOT / 'build' / 'hdl-tests'
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='rounding-', dir=directory) as temporary:
        temporary = Path(temporary)
        outputs, executable = temporary / 'expected.hex', temporary / 'simulation.vvp'
        values = (round(Fraction(value, divisor)) for divisor, low, high in
                  [(4, -65536, 65535), (2, -98304, 98302)] for value in range(low, high+1))
        outputs.write_text(''.join(f'{value&0x1ffff:05x}\n' for value in values), encoding='ascii')
        simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_ccs_rounding', '-o', simulator.path(executable),
            simulator.path(HDL / 'ccs_idwpt.v'), simulator.path(HDL / 'tb_ccs_rounding.v')])
        output = simulator.run(simulator.runtime + [simulator.path(executable), '+expected=' + simulator.path(outputs)])
        assert 'PASS rounding_sums=327679 signed_ties=RNE saturation_inputs=262144' in output, output
        return next(line for line in output.splitlines() if line.startswith('PASS '))


def check_headroom(simulator=None):
    simulator = simulator or Simulator()
    directory = ROOT / 'build' / 'hdl-tests'
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='headroom-', dir=directory) as temporary:
        executable = Path(temporary) / 'simulation.vvp'
        simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_ccs_headroom', '-o',
            simulator.path(executable), simulator.path(HDL / 'ccs_idwpt.v'),
            simulator.path(HDL / 'tb_ccs_headroom.v')])
        output = simulator.run(simulator.runtime + [simulator.path(executable)])
        assert 'PASS headroom_values=2097152' in output, output
        return next(line for line in output.splitlines() if line.startswith('PASS '))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Use a small impulse subset; not the complete input-position gate')
    args = parser.parse_args()
    simulator = Simulator()
    print(check_rounding(simulator), flush=True)
    print(check_headroom(simulator), flush=True)
    for slots in (2, 4, 8):
        for stalls in (False, True):
            print(check(slots, 2, stalls, not args.quick, simulator), flush=True)
    for headroom in (0, 1, 3, 15):
        print(check(4, headroom, True, False, simulator), flush=True)
    print('PASS: arithmetic, every tested input position, grouping, reset, stalls and exact cycles.', flush=True)
    print('FPGA placement/timing and whole-system budgets require separate evidence.', flush=True)


if __name__ == '__main__':
    main()
