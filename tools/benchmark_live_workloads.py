"""Qualify current live workloads through control, voices, VM, Material and Safety.

This desktop harness includes queued command consumption in measured rendering.
Command production, source compilation, construction and verification are outside
the sustained deadline. It does not qualify device I/O or cold-start deadlines.
Material cases use the existing host parameter defaults, not full parameter sweeps.
Timing conservatively includes one Material dispatch-observer call per block.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import sys
import time

import numpy as np

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / 'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT))

from symcrash_python_ap.audio.live_renderer import LiveSynthRenderer
from symcrash_python_ap.instrument_ir import (
    ADSR, InstrumentIR, L0Graph, L0Node, Layers, NodeParams, load_instrument,
)
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir
from symcrash_python_ap.primitives import BLOCK_SIZE, Q15_ONE, SAMPLE_RATE_HZ
from symcrash_python_ap.vm import hotpath_jit as hot
from tools.benchmark_live_renderer import BUDGET_US, positive, timing_summary

MATERIAL_NAMES = (
    'Clean Glue', 'Tape Bus', 'Rank Grit', 'Tropical Cut', 'Catastrophe Fold',
    'Catastrophe Cusp', 'Delay', 'Chorus', 'Reverb', 'SVF', 'Karplus-Strong',
)
WORKLOADS = ('controls', 'note-turnover') + tuple(f'material-{i}' for i in range(11))
ACID_PATH = ROOT / 'presets/instruments/acid_bass_v0.yaml'


def make_preset(workload):
    if workload not in WORKLOADS:
        raise ValueError(f'Unknown live workload: {workload}')
    if workload in ('controls', 'note-turnover'):
        return compiled_preset_from_ir(load_instrument(str(ACID_PATH)))
    material = int(workload.split('-')[1])
    # No Filter node: the host's filter override must not turn every Material
    # case into program 9. All changes belong to this benchmark fixture only.
    authored = InstrumentIR(layers=Layers(L0_graph=L0Graph(nodes=[
        L0Node('osc', 'Osc', NodeParams(waveform='saw')),
        L0Node('amp', 'Amp', NodeParams(adsr=ADSR(2.0, 10.0, 0.7, 4.0))),
    ])))
    authored.meta.instrument_id = f'live-budget-{workload}'
    authored.meta.name = f'Live budget {MATERIAL_NAMES[material]}'
    authored.compile.material.material_id = material
    return compiled_preset_from_ir(authored)


def expected_material(workload):
    return int(workload.split('-')[1]) if workload.startswith('material-') else 9


class LiveWorkload:
    def __init__(self, workload, voices, *, observe_dispatch=False):
        if workload not in WORKLOADS or not 1 <= voices <= 16:
            raise ValueError('Choose a declared workload and 1..16 voices')
        self.kind = workload
        self.voices = voices
        self.notes = [48 + 3 * voice for voice in range(voices)]
        self.renderer = LiveSynthRenderer(max_voices=voices)
        self.renderer.select_preset(make_preset(workload))
        for note in self.notes:
            self.renderer.note_on(note, 100)
        self.material_calls = []
        if observe_dispatch:
            original = self.renderer.stage.material.step_block

            def record(left, right, params, material_id, *args, **kwargs):
                self.material_calls.append(material_id)
                return original(left, right, params, material_id, *args, **kwargs)

            self.renderer.stage.material.step_block = record
        self.previous = np.zeros(2, dtype=np.int64)
        self.pitches = set()
        self.started_notes = 0
        self.turnovers = 0
        self.minimum_voices = voices
        self.maximum_peak = 0
        self.maximum_slew = 0
        self.new_note = None

    def enqueue(self, block):
        """Producer work is outside the audio callback's measured interval."""
        self.new_note = None
        if self.kind == 'controls':
            self.renderer.set_pot(8, (block * 977) % (Q15_ONE + 1))
            self.renderer.set_pot(9, 16000 + (block * 127) % 12000)
        elif self.kind == 'note-turnover' and block and block % 8 == 0:
            epoch = block // 8
            slot = (epoch - 1) % self.voices
            other = set(self.notes)
            note = 24 + (epoch * 7) % 104
            while note in other:
                note = 24 + (note - 23) % 104
            self.renderer.note_off(self.notes[slot])
            self.renderer.note_on(note, 77 + epoch % 51)
            self.notes[slot] = self.new_note = note
            self.turnovers += 1

    def inspect(self, output):
        """Untimed witnesses reject silent/bypassed or incorrectly routed work."""
        wide = np.asarray(output, dtype=np.int64)
        if wide.shape != (2, BLOCK_SIZE):
            raise AssertionError(f'Unexpected live block shape {wide.shape}')
        peak = int(np.abs(wide).max())
        differences = np.diff(np.column_stack((self.previous, wide)), axis=1)
        slew = int(np.abs(differences).max())
        assert peak <= 32000 and slew <= 2000, (peak, slew)
        self.maximum_peak = max(self.maximum_peak, peak)
        self.maximum_slew = max(self.maximum_slew, slew)
        self.previous[:] = wide[:, -1]
        frames = [item.frame for item in self.renderer.last_engine_output.render_batch]
        self.minimum_voices = min(self.minimum_voices, len(frames))
        assert len(frames) == self.voices, 'Workload lost a rendered voice'
        self.pitches.update((frame.note, frame.cents) for frame in frames)
        self.started_notes += sum(bool(frame.is_starting) for frame in frames)
        if self.new_note is not None:
            assert any(frame.note == self.new_note and frame.is_starting for frame in frames), 'Queued note did not start'
        assert self.material_calls, 'Actual Material dispatch must be observed'
        actual = self.material_calls[-1]
        assert actual == expected_material(self.kind), (actual, self.kind)
        return wide.astype(np.int16)

    def witnesses(self, blocks):
        if self.kind == 'controls' and blocks >= 8:
            assert len(self.pitches) > self.voices, 'Pitch control did not reach rendering'
        if self.kind == 'note-turnover' and blocks >= 9:
            assert self.turnovers and self.started_notes >= self.voices + self.turnovers
        if self.material_calls:
            assert self.material_calls == [expected_material(self.kind)] * blocks
        return dict(minimum_rendered_voices=self.minimum_voices,
                    distinct_rendered_pitches=len(self.pitches), note_turnovers=self.turnovers,
                    started_note_frames=self.started_notes, material_id=expected_material(self.kind),
                    material_name=MATERIAL_NAMES[expected_material(self.kind)],
                    peak=self.maximum_peak, slew=self.maximum_slew)


def verify_workload(workload, voices, blocks=96):
    """Observe the actual Material dispatch and compare every reference/JIT sample."""
    original_flag = hot.jit_enabled()
    reference = None
    reference_witness = None
    try:
        for enabled in (False, True):
            hot.set_jit_enabled(enabled)
            run = LiveWorkload(workload, voices, observe_dispatch=True)
            audio = np.empty((blocks, 2, BLOCK_SIZE), dtype=np.int16)
            for block in range(blocks):
                run.enqueue(block)
                audio[block] = run.inspect(run.renderer.render_block())
            witness = run.witnesses(blocks)
            assert len(run.material_calls) == blocks
            assert np.any(audio), 'Reference replay produced only silence'
            if not enabled:
                reference, reference_witness = audio, witness
            else:
                np.testing.assert_array_equal(reference, audio)
                assert witness == reference_witness
        return dict(blocks=blocks, audio_sha256=hashlib.sha256(audio.astype('<i2').tobytes()).hexdigest(),
                    **witness)
    finally:
        hot.set_jit_enabled(original_flag)


def measure_workload(workload, voices, blocks=750, warmup=100, profile=None):
    original_flag = hot.jit_enabled()
    try:
        hot.set_jit_enabled(True)
        run = LiveWorkload(workload, voices, observe_dispatch=True)
        audio = np.empty((blocks, 2, BLOCK_SIZE), dtype=np.int16)
        times = np.empty(blocks, dtype=np.float64)
        first_us = None
        for block in range(warmup + blocks):
            run.enqueue(block)
            start = time.perf_counter_ns()
            output = run.renderer.render_block()
            elapsed = (time.perf_counter_ns() - start) / 1000
            observed = run.inspect(output)
            if block == 0:
                first_us = elapsed
            if block >= warmup:
                times[block - warmup] = elapsed
                audio[block - warmup] = observed
        assert np.any(audio), 'Measured workload produced only silence'
        result = dict(workload=workload, voices=voices, warmup_blocks=warmup,
                      first_block_us=first_us, **timing_summary(times),
                      witnesses=run.witnesses(warmup + blocks),
                      audio_sha256=hashlib.sha256(audio.astype('<i2').tobytes()).hexdigest())
        if profile is not None:
            profiler = cProfile.Profile()
            for block in range(warmup + blocks, warmup + blocks + 100):
                run.enqueue(block)
                profiler.enable()
                output = run.renderer.render_block()
                profiler.disable()
                run.inspect(output)
            with Path(profile).open('w', encoding='utf-8') as stream:
                pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(50)
        return result, times, audio
    finally:
        hot.set_jit_enabled(original_flag)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workloads', choices=WORKLOADS, nargs='+', default=list(WORKLOADS))
    parser.add_argument('--voices', type=positive, nargs='+', default=[1, 4])
    parser.add_argument('--blocks', type=positive, default=750)
    parser.add_argument('--warmup', type=positive, default=100)
    parser.add_argument('--verify-blocks', type=positive, default=96)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not hot.NUMBA_AVAILABLE:
        parser.error('Numba is required for reference/JIT live qualification')
    if any(voices > 16 for voices in args.voices) or len(set(args.voices)) != len(args.voices):
        parser.error('--voices must contain distinct values in 1..16')
    if len(set(args.workloads)) != len(args.workloads):
        parser.error('--workloads must not repeat a case')
    if args.verify_blocks < 16:
        parser.error('--verify-blocks must be at least 16 to exercise note turnover')
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((ROOT / 'src/symcrash_python_ap').rglob('*.py'))}
    report = dict(scope='desktop live queue consumption through final stereo Safety; no device I/O',
        sample_rate=SAMPLE_RATE_HZ, block_size=BLOCK_SIZE, budget_us=BUDGET_US,
        python=platform.python_version(), platform=platform.platform(),
        controls='Pot 8 detune and pot 9 master gain change each block through the public queue',
        note_turnover='One note-off/new-note pair each eight blocks, rotating across the voice slots',
        materials='Current runtime IDs 0..10 at existing default ParamFrame values; explicit dispatch witnesses',
        measurement_instrumentation='One Material wrapper call and integer-list append per rendered block are included in timing; no inferred dispatch',
        first_block_context='after construction and reference replay; resources/JIT may already be warm',
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        acid_preset_sha256=hashlib.sha256(ACID_PATH.read_bytes()).hexdigest(),
        source_sha256=source_hashes, results=[], completed=False, passed=False)
    for workload in args.workloads:
        for voices in args.voices:
            name = f'{workload}-voices-{voices}'
            try:
                replay = verify_workload(workload, voices, args.verify_blocks)
                row, times, audio = measure_workload(workload, voices, args.blocks,
                    args.warmup, args.output / f'{name}-profile.txt')
                row['reference_jit'] = replay
                row['status'] = 'measured'
                np.savez(args.output / f'{name}.npz', times_us=times, audio=audio)
            except Exception as error:
                # Failed qualification is explicit and never supplies passing timing.
                row = dict(workload=workload, voices=voices, status='qualification_failed',
                           error=f'{type(error).__name__}: {error}',
                           no_observed_deadline_misses=False)
            report['results'].append(row)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(row), flush=True)
    for filename, expected in source_hashes.items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    report['completed'] = True
    report['passed'] = all(row['status'] == 'measured' and row['no_observed_deadline_misses']
                           for row in report['results'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS: all workloads qualified without observed misses' if report['passed']
          else 'FAIL: workload qualification or observed live deadlines failed')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
