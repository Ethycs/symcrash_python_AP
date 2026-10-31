"""Check cumulative DMA deadlines and prepared-frame audio through the whole path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.check_hdl_idwpt import Simulator, SimulatorUnavailable, coefficient_cases, reference_audio

HDL = ROOT / 'src/symcrash_verilog'
SOURCES = ('dma_window.v', 'ccs_dma_window.v', 'ccs_stream.v', 'coeff_buffer_owner.v',
           'ccs_idwpt.v', 'dma_priority.v', 'dma_burst.v')
WINDOWS = (1, 2, 16, 64, 40000)
STREAMS = [(k, 2, stalls) for k in (2, 4, 8) for stalls in (False, True)] + [(4, 15, True)]


def source_hashes():
    return {name: hashlib.sha256((HDL / name).read_bytes()).hexdigest() for name in SOURCES}


def fingerprint():
    return hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()


def check_window(limit=40000, simulator=None):
    if limit not in WINDOWS: raise ValueError('Unsupported validation window')
    simulator = simulator or Simulator()
    directory = ROOT / 'build/hdl-tests'
    directory.mkdir(parents=True, exist_ok=True)
    evidence = directory / f'dma_window_simulation_{fingerprint()[:12]}_{limit}.json'
    if evidence.exists(): evidence.unlink()
    with TemporaryDirectory(prefix='dma-window-', dir=directory) as temporary:
        executable = Path(temporary) / 'simulation.vvp'
        simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_dma_window',
            f'-Ptb_dma_window.WINDOW_CYCLES={limit}', '-o', simulator.path(executable),
            *(simulator.path(HDL / name) for name in SOURCES), simulator.path(HDL / 'tb_dma_window.v')])
        output = simulator.run(simulator.runtime + [simulator.path(executable)])
        assert f'PASS window limit={limit} ' in output, output
        summary = next(line for line in output.splitlines() if line.startswith('PASS '))
        report = {'sources': source_hashes(), 'fingerprint': fingerprint(), 'window_cycles': limit,
                  'testbench_sha256': hashlib.sha256((HDL / 'tb_dma_window.v').read_bytes()).hexdigest(),
                  'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'summary': summary}
        evidence.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        return summary


def check_stream(k=4, headroom=2, stalls=True, simulator=None):
    from symcrash_python_ap.vm.hotpath_jit import idwpt_polyphase
    simulator = simulator or Simulator()
    cases = coefficient_cases(k, exhaustive=False)
    expected = [reference_audio(c, headroom) for c in cases]
    for coefficients, oracle in zip(cases, expected):
        actual = np.empty(64, dtype=np.int16)
        idwpt_polyphase(coefficients.ravel(), actual, headroom)
        np.testing.assert_array_equal(actual, oracle)
    directory = ROOT / 'build/hdl-tests'
    directory.mkdir(parents=True, exist_ok=True)
    evidence = directory / f'ccs_dma_window_simulation_{fingerprint()[:12]}_k{k}_h{headroom}_{int(stalls)}.json'
    if evidence.exists(): evidence.unlink()
    with TemporaryDirectory(prefix='ccs-dma-window-', dir=directory) as temporary:
        temporary = Path(temporary)
        coeffs, outputs, executable = (temporary / name for name in ('coeffs.hex', 'expected.hex', 'simulation.vvp'))
        for path, values in ((coeffs, cases), (outputs, expected)):
            path.write_text(''.join(f'{int(v)&0xffff:04x}\n' for v in np.asarray(values).flat), encoding='ascii')
        simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_ccs_dma_window', f'-Ptb_ccs_dma_window.K={k}',
            f'-Ptb_ccs_dma_window.HEADROOM_BITS={headroom}', f'-Ptb_ccs_dma_window.STALLS={int(stalls)}',
            f'-Ptb_ccs_dma_window.CASES={len(cases)}', '-o', simulator.path(executable),
            *(simulator.path(HDL / name) for name in SOURCES), simulator.path(HDL / 'tb_ccs_dma_window.v')])
        output = simulator.run(simulator.runtime + [simulator.path(executable),
            '+coeffs=' + simulator.path(coeffs), '+expected=' + simulator.path(outputs)])
        assert f'PASS windowed_audio K={k} ' in output, output
        summary = next(line for line in output.splitlines() if line.startswith('PASS '))
        report = {'sources': source_hashes(), 'fingerprint': fingerprint(), 'K': k, 'headroom': headroom,
                  'stalls': bool(stalls), 'cases': len(cases), 'window_cycles': 40000,
                  'testbench_sha256': hashlib.sha256((HDL / 'tb_ccs_dma_window.v').read_bytes()).hexdigest(),
                  'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'reference_sha256': hashlib.sha256((ROOT / 'tools/audit_lifting_arithmetic.py').read_bytes()).hexdigest(),
                  'coefficients_sha256': hashlib.sha256(coeffs.read_bytes()).hexdigest(),
                  'expected_sha256': hashlib.sha256(outputs.read_bytes()).hexdigest(), 'summary': summary}
        evidence.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    simulator = Simulator()
    for limit in ((64,) if args.quick else WINDOWS):
        print(check_window(limit, simulator), flush=True)
    for configuration in ([(4, 2, False)] if args.quick else STREAMS):
        print(check_stream(*configuration, simulator=simulator), flush=True)


if __name__ == '__main__':
    main()
