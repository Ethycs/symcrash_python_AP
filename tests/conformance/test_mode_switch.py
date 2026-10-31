"""WSE-STD-210 section 9: serviced A/B handoff, fades and replay."""
import json
from copy import deepcopy
from fractions import Fraction
import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from symcrash_python_ap.fixed import midi_to_phase_inc_fixed
from symcrash_python_ap.vm import VMState, vm_step_block, coeff_lut_streaming as lut
from symcrash_python_ap.vm import harmonic_bank as bank
from symcrash_python_ap.vm.scheduling import ExecutionPolicy
from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes, vmstate_from_bytes
from tests.conformance.test_carrier_spectrum import frame
from tests.conformance.test_vm_snapshot_replay import execution_path, assert_state_equal


def command(mode, **kw):
    result = frame(**dict(dict(coeff_osc_mode=mode, coeff_table_id=0, waveform_type=0,
                   H_request=1, gain_q15=32767, legato_mode=True), **kw))
    result.backbone_ops = []
    return result


@pytest.mark.parametrize('first', [0, 1])
def test_switches_share_phase_and_reach_target_in_four_serviced_blocks(first, execution_path):
    state = VMState.create()
    calls = []
    a, b = lut.ccs_generate_coeffs_polyphase, bank.mode_b_generate_coeffs_vectorized

    def observe(mode, fn):
        def generate(osc, *args, **kw):
            calls.append((mode, osc.phase, osc.phase_inc))
            return fn(osc, *args, **kw)
        return generate

    phase = 0
    with patch.object(lut, 'ccs_generate_coeffs_polyphase', observe(0, a)), \
            patch.object(bank, 'mode_b_generate_coeffs_vectorized', observe(1, b)):
        for i, mode in enumerate([first]*5 + [1-first]*6 + [first]*6):
            cents = 37 if i < 7 else -19
            increment = midi_to_phase_inc_fixed(69, cents)
            calls.clear()
            vm_step_block(state, command(mode, cents=cents))
            assert calls and all(p == phase and inc == increment for _, p, inc in calls)
            if 5 <= i < 9 or 11 <= i < 15:
                elapsed = i-4 if i < 9 else i-10
                expected = (65536*elapsed//4) if mode else (65536*(4-elapsed)//4)
                assert state.mode_switch.mix_b_q16 == expected
                assert len(calls) == (2 if elapsed < 4 else 1)
            else:
                assert len(calls) == 1
            assert state.osc_mode == mode
            phase = (phase + 64*increment) % 2**32


@pytest.mark.parametrize('first', [0, 1])
@pytest.mark.parametrize('codec', ['json', 'binary'])
def test_checkpoint_midfade_and_scheduler_hold(first, codec, execution_path):
    state = VMState.create()
    for _ in range(5):
        vm_step_block(state, command(first))
    vm_step_block(state, command(1-first))
    assert state.mode_switch.remaining_blocks == 3
    if codec == 'json':
        restored = VMState.restore(json.loads(json.dumps(state.snapshot())))
    else:
        restored = vmstate_from_bytes(vmstate_to_bytes(state))
    assert_state_equal(state, restored)
    for mode, serviced in [(first, False), (1-first, True), (first, True),
                           (1-first, True)] + [(1-first, True)]*5:
        before = state.snapshot()
        policy = ExecutionPolicy(serviced=serviced)
        expected = vm_step_block(state, command(mode), policy=policy)[1].copy()
        np.testing.assert_array_equal(vm_step_block(restored, command(mode), policy=policy)[1], expected)
        assert_state_equal(state, restored)
        if not serviced:
            assert state.snapshot() == before


@pytest.mark.parametrize('first', [0, 1])
def test_rapid_reversal_starts_from_current_mix(first):
    state = VMState.create()
    for mode in [first]*5 + [1-first, first, 1-first, first, first, first, first]:
        before = getattr(state, 'mode_switch', None)
        was_initialized = before is not None and before.initialized
        if was_initialized:
            old_mix, old_mode = before.mix_b_q16, state.osc_mode
        vm_step_block(state, command(mode))
        if was_initialized and old_mode != mode:
            assert state.mode_switch.mix_b_q16 == round(Fraction(3*old_mix + mode*65536, 4))
    assert state.mode_switch.mix_b_q16 == first*65536
    assert state.mode_switch.remaining_blocks == 0


@pytest.mark.parametrize('mix', [0, 1, 16384, 32767, 32768, 32769, 49152, 65535, 65536])
def test_coefficient_blend_matches_independent_rational_rounding(mix):
    from symcrash_python_ap.vm.mode_switch import blend_coefficients
    values = [-32768, -32767, -3, -2, -1, 0, 1, 2, 3, 32766, 32767]
    pairs = [(a, b) for a in values for b in values]
    a, b = (np.array(items, dtype=np.int16) for items in zip(*pairs))
    expected = [round(Fraction(x * (65536 - mix) + y * mix, 65536)) for x, y in pairs]
    actual = blend_coefficients(a, b, mix)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int16


@pytest.mark.parametrize('first', [0, 1])
def test_actual_carriers_are_blended_before_existing_coefficient_pipeline(first, execution_path, monkeypatch):
    from symcrash_python_ap.vm import executor, mode_switch
    original_tail = executor._run_coeff_tail
    original_blend = mode_switch.blend_coefficients
    blends = []

    def observe_blend(a, b, mix):
        expected = np.array([round(Fraction(int(x)*(65536-mix)+int(y)*mix, 65536))
                             for x, y in zip(a, b)], dtype=np.int16)
        result = original_blend(a, b, mix)
        np.testing.assert_array_equal(result, expected)
        blends.append(expected)
        return result

    def observe_tail(state, cfg, *args, **kw):
        if blends:
            # command() uses full carrier refresh: the existing Q15 injection
            # precedes tail processing and is independent of the new fade.
            expected = [round(Fraction(int(x)*32767, 32768)) for x in blends[-1]]
            np.testing.assert_array_equal(state.coeff_state, expected)
        return original_tail(state, cfg, *args, **kw)

    monkeypatch.setattr(mode_switch, 'blend_coefficients', observe_blend)
    monkeypatch.setattr(executor, '_run_coeff_tail', observe_tail)
    state = VMState.create()
    for mode in [first]*5 + [1-first]*4 + [first]*4:
        blends.clear()
        vm_step_block(state, command(mode, coeff_table_id=1, waveform_type=5, H_request=16))


@pytest.mark.parametrize('source', [1, 2])
def test_outgoing_b_keeps_spectrum_modulation_and_tracks_pitch(source, execution_path, monkeypatch):
    original = bank.mode_b_generate_coeffs_vectorized
    state = VMState.create()
    last_b = command(1, waveform_type=3, H_request=32, fm_source=source,
        fm_depth_q15=8192, fm_modulator_q15=[1000, -2000, 3000, -4000],
        harmonic_gains_q15=[40000]*32, harmonic_gain_bound_q15=65536)
    for _ in range(5):
        vm_step_block(state, last_b)
    weights = deepcopy(state.mode_b_state.weights)
    last_modulator = state.mode_b_state.fm_mod_buffer.copy()
    calls = []

    def observe(osc, basis, output, gains=None, bound=32768):
        assert_state_equal(weights, osc.weights)
        assert tuple(gains) == (40000,)*32 and bound == 65536
        assert osc.fm_depth_q15 == 8192
        np.testing.assert_array_equal(osc.fm_mod_buffer,
            state.prev_fm_mod_buffer if source == 1 else last_modulator)
        calls.append((osc.phase, osc.phase_inc, basis.H_max))
        return original(osc, basis, output, gains, bound)

    monkeypatch.setattr(bank, 'mode_b_generate_coeffs_vectorized', observe)
    phase = state.mode_b_state.phase
    for i, note in enumerate([69, 120, 48, 48]):
        increment = midi_to_phase_inc_fixed(note, 37)
        calls.clear()
        vm_step_block(state, command(0, note=note, cents=37))
        if i < 3:
            assert len(calls) == 1 and calls[0][:2] == (phase, increment)
            assert calls[0][2] <= 9*2**31 // (10*increment)
        else:
            assert not calls
        phase = (phase + 64*increment) % 2**32


@pytest.mark.parametrize('first', [0, 1])
def test_rejected_frame_and_missing_outgoing_resource_leave_fade_unchanged(first, monkeypatch):
    from symcrash_python_ap.vm import basis_resource, mode_a_resource
    state = VMState.create()
    for _ in range(5):
        vm_step_block(state, command(first))
    vm_step_block(state, command(1-first))
    before = state.snapshot()
    rejected = command(first, wire_data=SimpleNamespace(reject_code=1))
    assert not np.any(vm_step_block(state, rejected)[1])
    assert state.snapshot() == before
    module = mode_a_resource if first == 0 else basis_resource
    with monkeypatch.context() as context:
        context.setattr(module, '_active', None)
        with pytest.raises(RuntimeError, match='Prepare'):
            vm_step_block(state, command(1-first))
    assert state.snapshot() == before


@pytest.mark.parametrize('first', [0, 1])
def test_explicit_note_start_cancels_fade_and_applies_authored_phase(first):
    state = VMState.create()
    for mode in [first]*5 + [1-first]:
        vm_step_block(state, command(mode))
    phase = 0xe1234567
    vm_step_block(state, command(first, is_starting=True, osc_start_phase_q32=phase))
    osc = state.ccs_state if first == 0 else state.mode_b_state
    assert osc.phase == (phase + 64*midi_to_phase_inc_fixed(69, 0)) % 2**32
    assert state.mode_switch.remaining_blocks == 0
    assert state.mode_switch.mix_b_q16 == first*65536


def test_literal_v2_extension_and_v1_migration():
    from tests.conformance.test_vm_binary_replay import literal_core_v1
    from symcrash_python_ap.vm.mode_switch import ModeSwitchState
    legacy = literal_core_v1()
    # v2 appends record 19, seven fields, to the frozen v1 record layout.
    literal = bytearray(legacy)
    struct.pack_into('<H', literal, 0, 2)
    struct.pack_into('<H', literal, 10, 41)
    literal += struct.pack('<HHBqqqBqq', 19, 7, 0, 0, 0, 0, 0, 32768, 0)
    state = vmstate_from_bytes(literal)
    assert state.mode_switch == ModeSwitchState()
    assert vmstate_to_bytes(state) == literal
    assert vmstate_to_bytes(state, version=1) == legacy
    snapshot = state.snapshot()
    snapshot['snapshot_version'] = 1
    del snapshot['mode_switch']
    assert VMState.restore(snapshot).mode_switch == ModeSwitchState()


@pytest.mark.parametrize('name,value', [('initialized', 1), ('mix_b_q16', -1),
    ('mix_b_q16', 65537), ('start_mix_b_q16', True), ('remaining_blocks', 4),
    ('b_fm_source', 3), ('b_gain_bound', 1), ('b_harmonic_gains', [32768]*31)])
def test_invalid_fade_state_is_rejected_by_both_checkpoint_encoders(name, value):
    state = VMState.create()
    setattr(state.mode_switch, name, value)
    with pytest.raises(ValueError):
        state.snapshot()
    with pytest.raises(ValueError):
        vmstate_to_bytes(state)
    snapshot = VMState.create().snapshot()
    from symcrash_python_ap.vm.snapshot_codec import encode
    snapshot['mode_switch']['fields'][name] = encode(value)
    with pytest.raises(ValueError):
        VMState.restore(snapshot)


@pytest.mark.parametrize('incoming', [0, 1])
def test_note_start_does_not_require_outgoing_resource(incoming, monkeypatch):
    from symcrash_python_ap.vm import basis_resource, mode_a_resource
    state = VMState.create()
    for _ in range(5):
        vm_step_block(state, command(1-incoming))
    module = basis_resource if incoming == 0 else mode_a_resource
    with monkeypatch.context() as context:
        context.setattr(module, '_active', None)
        vm_step_block(state, command(incoming, is_starting=True, osc_start_phase_q32=12345))
    assert state.mode_switch.remaining_blocks == 0
    assert state.mode_switch.mix_b_q16 == incoming*65536


def test_v1_cannot_silently_drop_live_fade_state():
    state = VMState.create()
    vm_step_block(state, command(1))
    with pytest.raises(ValueError, match='cannot preserve'):
        vmstate_to_bytes(state, version=1)


@pytest.mark.parametrize('changes', [dict(initialized=False, remaining_blocks=1),
    dict(mix_b_q16=1), dict(remaining_blocks=0, mix_b_q16=16384)])
def test_checkpoint_rejects_inconsistent_fade_progress(changes):
    state = VMState.create()
    for mode in [0]*5 + [1]:
        vm_step_block(state, command(mode))
    snapshot = state.snapshot()
    snapshot['mode_switch']['fields'].update(changes)
    with pytest.raises(ValueError):
        VMState.restore(snapshot)
    for name, value in changes.items():
        setattr(state.mode_switch, name, value)
    with pytest.raises(ValueError):
        vmstate_to_bytes(state)


@pytest.mark.parametrize('first', [0, 1])
@pytest.mark.parametrize('note', [48, 60, 69, 84])
def test_midband_sine_switch_boundaries_follow_existing_ratio_guidance(first, note, execution_path):
    state = VMState.create()
    audio = np.asarray([vm_step_block(state, command(mode, note=note, cents=37))[1].copy()
                        for mode in [first]*7 + [1-first]*7 + [first]*7 + [1-first]*7],
                       dtype=np.int64)[4:]
    intra_rms = np.sqrt(np.mean(np.diff(audio, axis=1).astype(np.float64)**2))
    boundary_peak = np.max(np.abs(audio[1:, 0] - audio[:-1, -1]))
    assert intra_rms > 0 and boundary_peak / intra_rms < 3


@pytest.mark.parametrize('other', [2, 3, 4, 5])
def test_other_oscillator_modes_cancel_the_ab_fade(other):
    state = VMState.create()
    for mode in [0]*5 + [1, other, 0]:
        vm_step_block(state, command(mode))
        if mode == other:
            assert not state.mode_switch.initialized
            assert state.mode_switch.remaining_blocks == 0
    assert state.mode_switch.mix_b_q16 == 0


@pytest.mark.parametrize('codec', ['json', 'binary'])
def test_midfade_checkpoint_continues_in_fresh_process(codec, tmp_path):
    state = VMState.create()
    for mode in [1]*5 + [0]:
        vm_step_block(state, command(mode, cents=37))
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.write_bytes(json.dumps(state.snapshot()).encode() if codec == 'json'
                           else vmstate_to_bytes(state))
    expected = [vm_step_block(state, command(mode, cents=37))[1].copy()
                for mode in [0, 1, 0, 0, 0, 0, 1]]
    script = '''import json, sys, numpy as np
from pathlib import Path
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.voice.voice_state_codec import vmstate_from_bytes
from tests.conformance.test_mode_switch import command
raw = Path(sys.argv[1]).read_bytes()
state = VMState.restore(json.loads(raw)) if sys.argv[2] == 'json' else vmstate_from_bytes(raw)
blocks = [vm_step_block(state, command(mode, cents=37))[1].copy()
          for mode in [0, 1, 0, 0, 0, 0, 1]]
np.save(sys.argv[3], blocks)
'''
    output = tmp_path / 'audio.npy'
    subprocess.run([sys.executable, '-c', script, str(checkpoint), codec, str(output)], check=True,
                   capture_output=True, text=True)
    np.testing.assert_array_equal(np.load(output), expected)
