"""Exercise complete packed VM pages through the qualified cache and renderer.

The current VM bridge is K4. Optional external records are preserved/decoded
by Python; this bench covers the core page's DMA, ownership, update and render
path. It does not claim HDL execution of the external engine records.
"""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_hdl_dma import digest
from tools.check_hdl_idwpt import Simulator, reference_audio
from tools.check_hdl_voice_stream import SOURCE_NAMES, validate_summary, write_words
from symcrash_python_ap.voice.hardware_vm_page import pack_vm_page, unpack_vm_page, page_layout
from tests.conformance.test_vm_snapshot_replay import prepared, assert_state_equal


HDL = ROOT / 'src/symcrash_verilog'
DIRECTORY = ROOT / 'build/hdl-tests/vm_page'
KINDS = ('a', 'b', 'pwm', 'fm6', 'morph', 'sample', 'sample_coeff', 'wavetable', 'granular')


def prepare_vectors(folder):
    folder.mkdir(parents=True, exist_ok=True)
    pages, updates, expected, rows = [], [], [], []
    for i, kind in enumerate(KINDS * 2):
        voice_id = 0x100 + i % 16 + ((i // 16) % 2) * 256
        context = 0xa0000000 + i
        state, _ = prepared(kind, voice_id=voice_id)
        # Retain high-bit metadata while the cache reads only coefficients.
        state.phase = 0x80000000 + i
        state.prng_state = 0xfedcba98 ^ i
        options = dict(context_tag=context, page_address=i*2048,
                       external_start=0x04000000+i*65536, external_capacity=65536)
        before = pack_vm_page(state, **options)
        raw = {record.address: record.data for record in before.external}
        restored = unpack_vm_page(before.page, lambda address, count: raw[address],
            voice_id=voice_id, context_tag=context, page_address=options['page_address'])
        assert_state_equal(state, restored)
        pages.append(np.frombuffer(before.page, dtype='<u2'))
        expected.append(reference_audio(state.coeff_state.reshape(64, 4), 2))
        changed = np.bitwise_xor(state.coeff_state.view(np.uint16), np.uint16(0x5a5a)).view(np.int16)
        state.coeff_state[:] = changed
        state.tail_previous_coeffs[:] = changed
        after = pack_vm_page(state, **options)
        assert before.page[:1024] == after.page[:1024]
        assert before.page[1536:] == after.page[1536:]
        assert before.external == after.external
        updates.append(np.frombuffer(after.page, dtype='<u2'))
        expected.append(reference_audio(changed.reshape(64, 4), 2))
        for record in before.external:
            (folder / f'case{i:02d}_external{record.field_id}.bin').write_bytes(record.data)
        rows.append(dict(case=i, kind=kind, voice_id=voice_id, context_tag=context,
                         page_address=options['page_address'], external_bytes=before.external_bytes))
    write_words(folder / 'pages.hex', pages)
    write_words(folder / 'updates.hex', updates)
    (folder / 'expected.hex').write_text(''.join(f'{int(v)&65535:04x}\n' for v in np.asarray(expected).flat), encoding='ascii')
    (folder / 'modes.hex').write_text('1\n' * len(rows), encoding='ascii')
    (folder / 'offsets.hex').write_text('40\n' * len(rows), encoding='ascii')
    (folder / 'states.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    return rows


def check(stalls=False, directory=DIRECTORY):
    folder = Path(directory) / f'latency100_stalls{int(stalls)}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'inventory.json').unlink(missing_ok=True)
    sources = {name: digest(HDL / name) for name in SOURCE_NAMES}
    rows = prepare_vectors(folder)
    cases = len(rows)
    simulator = Simulator()
    with TemporaryDirectory(prefix='simulate-', dir=folder) as temporary:
        executable = Path(temporary) / 'vm_page.vvp'
        simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_voice_cache_stream',
            f'-Ptb_voice_cache_stream.CASES={cases}', '-Ptb_voice_cache_stream.HEADROOM_BITS=2',
            '-Ptb_voice_cache_stream.LATENCY=100', f'-Ptb_voice_cache_stream.STALLS={int(stalls)}',
            '-o', simulator.path(executable), *(simulator.path(HDL / name) for name in SOURCE_NAMES)])
        output = simulator.run(simulator.runtime + [simulator.path(executable),
            *(f'+{name}=' + simulator.path(folder / f'{name}.hex') for name in ('pages', 'updates', 'expected', 'modes', 'offsets'))])
    (folder / 'simulation.log').write_text(output, encoding='utf-8')
    summary = next(line for line in output.splitlines() if line.startswith('PASS voice_stream '))
    fields, service = validate_summary(summary, cases, 2, 100, stalls, ((cases+3)//4)*32)
    assert service['K2'] == service['K8'] == (999999, 0), 'This bridge must not silently change reference K'
    assert 348 <= service['K4'][0] <= service['K4'][1] <= 400
    if not stalls:
        assert service['K4'] == (348, 348)
    assert sources == {name: digest(HDL / name) for name in SOURCE_NAMES}
    report = dict(scope='K4 complete VM core pages; external records are checked by Python, not executed by HDL',
        sources=sources, runner_sha256=digest(Path(__file__)),
        page_codec_sha256=digest(ROOT / 'src/symcrash_python_ap/voice/hardware_vm_page.py'),
        fixture_sha256=digest(ROOT / 'tests/conformance/test_vm_snapshot_replay.py'),
        oracle_sha256=digest(ROOT / 'tools/audit_lifting_arithmetic.py'),
        states=rows, layout=page_layout(), summary=summary, counts=fields,
        service_cycles=service['K4'], artifacts={p.name: digest(p) for p in folder.iterdir()
            if p.is_file() and p.name != 'inventory.json'})
    (folder / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    rows = []
    for stalls in (False, True):
        rows.append(check(stalls, args.directory))
        print(rows[-1]['summary'], flush=True)
    (args.directory / 'matrix.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
