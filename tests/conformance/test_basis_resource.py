"""Prepared Mode B assets: exact arithmetic, integrity and real render use."""
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from bisect import bisect_left, bisect_right
import json
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.fixed import midi_to_phase_inc_fixed
from symcrash_python_ap.vm import VMState, vm_step_block, harmonic_bank as bank
from symcrash_python_ap.vm import basis_resource as resource, basis_resource_builder as builder
from symcrash_python_ap.vm.carrier_spectrum import harmonic_limit
from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes, vmstate_from_bytes
from tests.conformance.test_carrier_spectrum import frame
from tests.conformance.test_vm_snapshot_replay import execution_path


@lru_cache(maxsize=256)
def reference_basis(note, pitch, count):
    """Scalar FWPT frequency anchors plus independent Fraction arithmetic."""
    from symcrash_python_ap.pitch_rom import MIDI_CENT_PHASE_Q32
    cent = MIDI_CENT_PHASE_Q32.index(pitch)
    if cent % 100 == 0:
        return bank.generate_basis_frames(note, count, phase_inc=pitch)
    frequencies, grid = reference_frequencies()
    cosine, sine = np.empty((count,64,4),dtype=np.int16), np.empty((count,64,4),dtype=np.int16)
    for h in range(count):
        frequency=pitch*(h+1)
        if frequency in frequencies:
            source=reference_frequency(frequency)
            cosine[h],sine[h]=source.Ucos_h[0],source.Usin_h[0]
            continue
        left=max(0,min(len(grid)-2,bisect_right(grid,frequency)-1))
        if left==0:
            indices=(0,1,2,3)
        elif left==len(grid)-2:
            indices=tuple(range(len(grid)-4,len(grid)))
        else:
            span=grid[left+1]-grid[left]
            indices=(max(0,bisect_right(grid,grid[left]-span)-1),left,left+1,
                     min(len(grid)-1,bisect_left(grid,grid[left+1]+span)))
        nodes=[grid[i] for i in indices]
        weights=[]
        for i in range(4):
            weight=2**30
            for j in range(4):
                if i != j:
                    ratio=round(Fraction((frequency-nodes[j])*2**30,nodes[i]-nodes[j]))
                    weight=round(Fraction(weight*ratio,2**30))
            weights.append(weight)
        largest=max(range(4),key=lambda i:abs(weights[i]))
        weights[largest] += 2**30-sum(weights)
        for name,output in (('Ucos_h',cosine),('Usin_h',sine)):
            values=[getattr(reference_frequency(node),name)[0].reshape(-1).tolist() for node in nodes]
            mixed=[max(-32768,min(32767,round(Fraction(sum(w*x for w,x in zip(weights,items)),2**30))))
                   for items in zip(*values)]
            output[h]=np.array(mixed,dtype=np.int16).reshape(64,4)
    return bank.HarmonicBasisFrames(reference_frequency(pitch).DC_frame,cosine,sine,
                                   count,note,pitch*48000/2**32)


@lru_cache(maxsize=1)
def reference_frequencies():
    from tests.conformance.test_pitch_conversion import oracle
    values=set()
    for note in range(128):
        capacity=min(32,9*2**31//(10*oracle(max(0,note-2),0)))
        for h in range(1,capacity+1):
            values.add(oracle(note,0)*h)
    grid=[]
    for frequency in sorted(values):
        if not grid or frequency-grid[-1]>=64:
            grid.append(frequency)
    return values,grid


@lru_cache(maxsize=2048)
def reference_frequency(frequency):
    return bank.generate_basis_frames(0,1,phase_inc=frequency)


def forbidden(*args, **kwargs):
    raise AssertionError('Resource generation or loading occurred during playback')


def prohibit_preparation(monkeypatch):
    import importlib
    from symcrash_python_ap.vm import hotpath_jit
    from symcrash_python_ap.wavelet import fast_lifting
    # Load and guard module-level aliases too. Lazy snapshot imports must not
    # retain a patched alias after monkeypatch restores the defining module.
    for name in ('sample_source', 'granular_engine', 'wavetable_engine',
                 'matrix_coeffs', 'coeff_lut_streaming'):
        module = importlib.import_module('symcrash_python_ap.vm.' + name)
        monkeypatch.setattr(module, 'fwpt_jit', forbidden)
    monkeypatch.setattr(bank, 'generate_basis_frames', forbidden)
    monkeypatch.setattr(builder, '_generate_pitch', forbidden)
    monkeypatch.setattr(builder, 'build_basis_resource', forbidden)
    monkeypatch.setattr(resource.PreparedBasisAtlas, '__init__', forbidden)
    monkeypatch.setattr(hotpath_jit, 'fwpt_64', forbidden)
    monkeypatch.setattr(fast_lifting, 'fwpt_jit', forbidden)


@pytest.fixture(scope='module')
def atlas():
    return resource.prepare_basis_resources()


@pytest.mark.parametrize('note,cents', [(0, -199), (0, 0), (8, 37), (20, -99),
    (21, 147), (32, 99), (48, -150), (60, -199), (69, 0), (69, 25),
    (80, -101), (88, 77), (100, 100), (108, -39), (109, 80), (120, 0), (127, 199)])
def test_all_harmonics_match_scalar_sinecore_transform(atlas, note, cents):
    pitch = midi_to_phase_inc_fixed(note, cents)
    count = harmonic_limit(pitch)
    expected = reference_basis(note, pitch, count)
    actual = atlas.basis(note, pitch, count)
    for key in ('DC_frame', 'Ucos_h', 'Usin_h'):
        a, b = getattr(actual, key), getattr(expected, key)
        np.testing.assert_array_equal(a, b)
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a.flags.writeable = True
    for prefix in (0, 1, count):
        np.testing.assert_array_equal(atlas.basis(note, pitch, prefix).Usin_h, expected.Usin_h[:prefix])


def test_complete_integer_pitch_domain(atlas):
    # Every cent residue and semitone target for all MIDI notes; the combined
    # absolute pitch now clamps consistently to the 0..12700 cent domain.
    from symcrash_python_ap.pitch_rom import MIDI_CENT_PHASE_Q32
    keys = set(MIDI_CENT_PHASE_Q32)
    for note in range(128):
        for cents in range(-199, 200):
            assert midi_to_phase_inc_fixed(note, cents) in keys
        for semitone in range(-130, 131):
            for residue in (0, 1, 37, 99):
                assert midi_to_phase_inc_fixed(note, 100 * semitone + residue) in keys
        for cents in (-10**12, 10**12):
            assert midi_to_phase_inc_fixed(note, cents) in keys
    assert len(keys) == 12701
    assert atlas.keys == MIDI_CENT_PHASE_Q32[::100]
    assert atlas.path.stat().st_size == resource.resource_size() == 3124940
    assert atlas.path.stat().st_size < 4*1024*1024
    for pitch in MIDI_CENT_PHASE_Q32:
        assert atlas.basis(0,pitch,0).H_max == 0


@pytest.fixture
def small_resource(tmp_path):
    keys = tuple(sorted(midi_to_phase_inc_fixed(n, 37) for n in (0, 69, 127)))
    path = tmp_path / 'basis.bin'
    builder.build_basis_resource(path, pitch_increments=keys)
    return path, keys


def test_builder_is_deterministic_and_partial_resources_are_explicit(small_resource, tmp_path):
    path, keys = small_resource
    other = builder.build_basis_resource(tmp_path / 'other.bin', pitch_increments=keys)
    assert path.read_bytes() == other.read_bytes()
    with pytest.raises(ValueError, match='coverage'):
        resource.PreparedBasisAtlas(path)
    partial = resource.PreparedBasisAtlas(path, expected_keys=keys)
    with pytest.raises(ValueError, match='absent'):
        partial.basis(60, midi_to_phase_inc_fixed(60), 1)
    with pytest.raises(ValueError, match='bandlimit'):
        partial.basis(127, keys[-1], 32)


@pytest.mark.parametrize('damage', ['magic', 'version', 'count', 'recipe', 'digest', 'pitch', 'dc', 'coeff', 'truncate', 'append'])
def test_resource_rejects_corruption(small_resource, damage):
    path, keys = small_resource
    data = bytearray(path.read_bytes())
    offsets = dict(magic=0, version=8, count=12, recipe=16, digest=48,
                   pitch=resource.HEADER.size,
                   dc=resource.HEADER.size+4*len(keys)+12*len(resource.frequency_directory(keys)), coeff=len(data)-1)
    if damage == 'truncate':
        data = data[:-1]
    elif damage == 'append':
        data += b'x'
    else:
        data[offsets[damage]] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError):
        resource.PreparedBasisAtlas(path, expected_keys=keys)


@pytest.mark.parametrize('keys', [(), (1, 1), (2, 1), (-1,), (2**32,), (True,), (1.5,)])
def test_builder_rejects_invalid_pitch_directory(tmp_path, keys):
    with pytest.raises(ValueError):
        builder.build_basis_resource(tmp_path / 'invalid.bin', pitch_increments=keys)
    assert not list(tmp_path.iterdir())


def test_interrupted_build_and_failed_adoption_preserve_previous_resource(atlas, tmp_path):
    target = tmp_path / 'basis.bin'
    target.write_bytes(b'previous resource')
    with pytest.raises(AssertionError):
        builder.build_basis_resource(target, pitch_increments=(1000000,), progress=forbidden)
    assert target.read_bytes() == b'previous resource'
    assert list(tmp_path.iterdir()) == [target]
    before = resource.prepared_basis_atlas()
    with pytest.raises(ValueError):
        resource.prepare_basis_resources(target, build_if_missing=False)
    with pytest.raises(FileNotFoundError):
        resource.prepare_basis_resources(tmp_path / 'absent.bin', build_if_missing=False)
    assert resource.prepared_basis_atlas() is before


@pytest.mark.parametrize('wire', [False, True])
def test_unprepared_vm_rejects_before_musical_state_changes(atlas, monkeypatch, wire):
    state = VMState.create(0)
    before = state.snapshot()
    monkeypatch.setattr(resource, '_active', None)
    bank.clear_basis_atlas()
    f = frame()
    if wire:
        from tests.conformance.test_scheduled_vm import packet
        f = packet(carrier_model_u8=1).to_voice_frame()
    with pytest.raises(RuntimeError, match='Prepare'):
        vm_step_block(state, f)
    assert state.snapshot() == before


def commands():
    result = []
    for i, (note, cents) in enumerate([(0, 0), (69, -199), (69, 25), (127, -37),
                                    (100, 147), (60, -150), (48, 99), (108, 0)]):
        result.append(frame(note=note, cents=cents, H_request=32, legato_mode=True,
                            is_starting=i == 0, partial_amps_q15=[10000] * 32))
        result.append(replace(result[-1], partial_amps_q15=[], oeis_sequence_id=1))
    return result


def test_cache_misses_and_json_binary_restore_need_no_forward_transform(atlas, monkeypatch, execution_path):
    frames = commands()
    reference_state = VMState.create(0)
    with monkeypatch.context() as patch:
        patch.setattr(bank, 'get_basis_for_pitch', reference_basis)
        expected = [vm_step_block(reference_state, f)[1].copy() for f in frames]
    state = VMState.create(0)
    prohibit_preparation(monkeypatch)
    for i, f in enumerate(frames):
        bank.clear_basis_atlas()
        np.testing.assert_array_equal(vm_step_block(state, f)[1], expected[i])
        if i == 7:
            snapshots = [VMState.restore(json.loads(json.dumps(state.snapshot()))),
                         vmstate_from_bytes(vmstate_to_bytes(state))]
            for restored in snapshots:
                for subsequent, output in zip(frames[i+1:], expected[i+1:]):
                    bank.clear_basis_atlas()
                    np.testing.assert_array_equal(vm_step_block(restored, subsequent)[1], output)
    assert state.snapshot() == reference_state.snapshot()


def test_views_are_bounded_and_cold_lookup_uses_only_prepared_data(atlas, monkeypatch):
    prohibit_preparation(monkeypatch)
    bank.clear_basis_atlas()
    from symcrash_python_ap.pitch_rom import MIDI_CENT_PHASE_Q32
    for pitch in MIDI_CENT_PHASE_Q32[:160]:
        bank.get_basis_for_pitch(0, pitch, 1)
    assert bank.get_basis_for_pitch.cache_info().currsize == 128
    bank.clear_basis_atlas()
    assert resource.prepared_basis_atlas() is atlas
    assert bank.get_basis_for_pitch(0, atlas.keys[0], 0).Usin_h.shape == (0, 64, 4)


@pytest.mark.parametrize('binary', [False, True], ids=['json', 'core'])
def test_fresh_process_restores_and_renders_without_resource_generation(atlas, tmp_path, binary):
    state = VMState.create(0)
    frames = commands()
    for f in frames[:8]:
        vm_step_block(state, f)
    path = tmp_path / ('checkpoint.bin' if binary else 'checkpoint.json')
    if binary:
        path.write_bytes(vmstate_to_bytes(state))
    else:
        path.write_text(json.dumps(state.snapshot()))
    expected = [vm_step_block(state, f)[1].tolist() for f in frames[8:]]
    code = '''
import json, sys
from pathlib import Path
from symcrash_python_ap.vm import VMState, vm_step_block, harmonic_bank as bank
from symcrash_python_ap.vm import basis_resource_builder as builder
from symcrash_python_ap.voice.voice_state_codec import vmstate_from_bytes
from tests.conformance.test_basis_resource import commands, forbidden
bank.generate_basis_frames = builder._generate_pitch = builder.build_basis_resource = forbidden
path = Path(sys.argv[1])
state = vmstate_from_bytes(path.read_bytes()) if path.suffix == '.bin' else VMState.restore(json.loads(path.read_text()))
audio = []
for frame in commands()[8:]:
    bank.clear_basis_atlas()
    audio.append(vm_step_block(state, frame)[1].tolist())
print(json.dumps(audio))
'''
    child = subprocess.run([sys.executable, '-c', code, str(path)], check=True, capture_output=True, text=True)
    assert json.loads(child.stdout) == expected


@pytest.mark.parametrize('host_kind', ['master', 'workbench', 'executor', 'orchestrator', 'unified'])
def test_hosts_prepare_before_late_voice_allocation(atlas, monkeypatch, tmp_path, host_kind):
    # Remove the process singleton to prove each host's constructor prepares it.
    # Reuse a verified immutable image to avoid five redundant whole-file reads.
    monkeypatch.setattr(resource, '_active', None)
    monkeypatch.setattr(resource, 'PreparedBasisAtlas', lambda path: atlas)
    session = None
    if host_kind == 'master':
        from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
        host = SynthMaster(SynthConfig(n_voices=4))
        host.load_preset_from_yaml('presets/instruments/super_saw_v0.yaml')
    elif host_kind == 'workbench':
        import shutil
        from symcrash_workbench.session import WorkbenchSession
        from symcrash_workbench.streaming import StreamingVoice
        path = tmp_path / 'preset.yaml'
        shutil.copy('presets/instruments/string_pad_v0.yaml', path)
        session = WorkbenchSession(path, autoplay=False, log_dir=tmp_path)
        host = StreamingVoice(session)
    elif host_kind == 'executor':
        from symcrash_python_ap.vm.executor import VMExecutor
        host = VMExecutor()
    elif host_kind == 'orchestrator':
        from symcrash_python_ap.voice.orchestrator import BlockOrchestrator
        host = BlockOrchestrator(n_voices=4)
    else:
        from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
        host = UnifiedOutputStage()
    assert resource.prepared_basis_atlas() is atlas
    monkeypatch.setattr(resource, 'PreparedBasisAtlas', type(atlas))
    prohibit_preparation(monkeypatch)
    try:
        if host_kind in ('master', 'workbench'):
            host.note_on(69)
            for _ in range(12):
                bank.clear_basis_atlas()
                host.render_block()
        elif host_kind == 'executor':
            for f in commands():
                bank.clear_basis_atlas()
                host.step_voice(0, f)
        elif host_kind == 'orchestrator':
            from tests.conformance.test_scheduled_vm import packet
            for i, f in enumerate(commands()):
                bank.clear_basis_atlas()
                host.update_frame(0, packet(seq=i+1, note_u8=f.note, cents_i16=f.cents,
                                            carrier_model_u8=1))
                host.step_block()
            assert host.vm_states[0]._mode_b_state is not None
            assert host.vm_states[0].mode_b_state.phase_inc == midi_to_phase_inc_fixed(108)
        else:
            for f in commands():
                bank.clear_basis_atlas()
                host.render_block(f)
    finally:
        if session is not None:
            host.stop()
            session.close()


def test_fresh_preparation_compiles_detuned_kernels_before_first_audio_block(atlas):
    code = '''
import numpy as np
from dataclasses import replace
from symcrash_python_ap.vm import VMState, vm_step_block, hotpath_jit as hot, harmonic_bank as bank
from symcrash_python_ap.vm.resources import prepare_oscillator_resources
from symcrash_python_ap.vm.basis_anchors import compiled_descriptor_arrays
from tests.conformance.test_basis_resource import commands
hot.set_jit_enabled(True)
prepare_oscillator_resources()
for kernel in (compiled_descriptor_arrays, hot.mode_b_interpolated_coefficients):
    assert kernel.signatures, 'Startup left a detuned kernel uncompiled: '+kernel.__name__
    kernel.disable_compile()
state = VMState.create(0)
peak = 0
for frame in commands():
    for count in (0,1,32):
        bank.clear_basis_atlas()
        output = vm_step_block(state,replace(frame,H_request=count))[1]
        peak = max(peak,int(np.abs(output.astype(np.int32)).max()))
assert peak > 0
print('prepared kernels render cold detuned blocks without compilation')
'''
    child = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    assert 'without compilation' in child.stdout
