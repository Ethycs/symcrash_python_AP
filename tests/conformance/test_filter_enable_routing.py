"""The program enable bit, not rounded cutoff, owns the live SVF override."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import struct

import numpy as np
import pytest

from symcrash_python_ap.audio.live_renderer import LiveSynthRenderer
from symcrash_python_ap.frames import ParamFrame, VoiceFrame
from symcrash_python_ap.instrument_ir import L0Node, NodeParams
from symcrash_python_ap.library.capsule_codec import pack_voice_params, unpack_voice_params
from symcrash_python_ap.library.central_registry import get_central_registry
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir, compiled_preset_from_capsule
from symcrash_python_ap.macro_runtime import MacroRuntime, SystemMacro
from symcrash_python_ap.primitives import Q15_ONE
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
from symcrash_python_ap.voice.voice_engine_registry import VoiceParamID, ValueType
from symcrash_python_ap.vm.harmonic_bank import LFOState, LFOTarget, LFORouting, LFOWaveform
from tests.conformance.test_native_preset_transport import source, owner, capsule_for
from tests.conformance.test_host_preset_ownership import host
from tests.conformance.test_scheduled_vm import schedule
from tests.conformance.test_vm_snapshot_replay import execution_path


def authored(enabled=None, cutoff=Q15_ONE, mode='lp', **kwargs):
    ir = source()
    ir.compile.material.material_id = 2
    if enabled is not None:
        ir.layers.L0_graph.nodes.append(L0Node('filter', 'Filter', NodeParams(
            filter_enable=enabled, filter_cutoff_q15=cutoff, mode=mode, **kwargs)))
    return ir


def observe(stage, monkeypatch):
    calls = []
    original = stage.material.step_block

    def record(left, right, params, material_id, *args, **kwargs):
        calls.append((material_id, kwargs['filter_cutoff_q15'],
                      kwargs['filter_reso_q15'], kwargs['filter_mode']))
        return original(left, right, params, material_id, *args, **kwargs)

    monkeypatch.setattr(stage.material, 'step_block', record)
    return calls


@pytest.mark.parametrize('route', ['native', 'capsule'])
@pytest.mark.parametrize('enabled,mode', [(None, 'lp'), (False, 'lp'),
    (True, 'lp'), (True, 'hp'), (True, 'bp'), (True, 'notch')])
def test_authored_enable_at_maximum_cutoff_reaches_actual_live_dispatch(
        monkeypatch, route, enabled, mode, execution_path):
    ir = authored(enabled, mode=mode)
    capsule = capsule_for(ir)
    assert capsule.defaults.voice_param_defaults[VoiceParamID.FILTER_ENABLE] == int(bool(enabled))
    preset = (compiled_preset_from_ir(ir) if route == 'native'
              else compiled_preset_from_capsule(capsule))
    assert preset.filter_enabled is bool(enabled)
    renderer = LiveSynthRenderer(max_voices=1)
    renderer.select_preset(preset)
    renderer.note_on(60, 100)
    calls = observe(renderer.stage, monkeypatch)
    for _ in range(4):
        renderer.render_block()
        frame = renderer.last_engine_output.render_batch[0].frame
        assert frame.filter_enabled is bool(enabled)
        assert frame.filter_cutoff_q15 == 32766  # Existing Q15 RNE is unchanged.
    expected = (9, 32766, 0, ('lp', 'hp', 'bp', 'notch').index(mode)) if enabled else (2, Q15_ONE, 0, 0)
    assert calls == [expected] * 4


def test_legacy_hz_authoring_still_enables_the_first_filter():
    ir = authored(False, cutoff_hz=2000.0)
    for preset in (compiled_preset_from_ir(ir), compiled_preset_from_capsule(capsule_for(ir))):
        assert preset.filter_enabled is True
        assert preset.native_carrier.filter_cutoff_q15 < Q15_ONE


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('lfo_enabled', [False, True])
def test_enable_is_independent_of_unchanged_macro_and_lfo_values(
        monkeypatch, enabled, lfo_enabled, execution_path):
    ir = authored(True, cutoff=25000, mode='hp', filter_resonance_q15=5000)
    lfo = LFOState(id='cutoff')
    lfo.configure_lfo(rate_hz=17.0, depth_q15=19000, waveform=LFOWaveform.TRIANGLE,
        routings=[LFORouting(LFOTarget.FILTER_CUTOFF, 9000), LFORouting(LFOTarget.FILTER_RESO, 2000)])
    lfo.enabled = lfo_enabled
    ir.modulators = [lfo]
    preset = compiled_preset_from_ir(ir)
    actual = owner(replace(preset, filter_enabled=enabled))
    legacy = owner(replace(preset, filter_enabled=None))
    for engine in (actual, legacy):
        engine.macro_runtime = MacroRuntime()
        engine.macro_runtime.macro_curr_q15[SystemMacro.BRIGHTNESS] = 18000
        engine.macro_runtime.macro_curr_q15[SystemMacro.VIBRATO] = Q15_ONE
    stage = UnifiedOutputStage()
    stage.configure_preset(preset)
    calls = observe(stage, monkeypatch)
    cutoffs = []
    for _ in range(32):
        a, b = actual.step_voice_simple(0), legacy.step_voice_simple(0)
        values_a, values_b = asdict(a), asdict(b)
        assert values_a.pop('filter_enabled') is enabled
        assert values_b.pop('filter_enabled') is None
        assert values_a == values_b
        cutoffs.append(a.filter_cutoff_q15)
        stage.render_frames([a])
    assert (len(set(cutoffs)) > 1) is lfo_enabled
    assert all(row[0] == (9 if enabled else 2) for row in calls)


@pytest.mark.parametrize('global_filter', [False, True])
def test_disabled_voice_cannot_darken_enabled_voice_or_steal_full_cutoff_mode(monkeypatch, global_filter):
    stage = UnifiedOutputStage()
    calls = observe(stage, monkeypatch)
    params = ParamFrame(filter_enabled=global_filter, filter_cutoff_q15=Q15_ONE,
                        filter_mode=3, filter_reso_q15=1000)
    frames = [VoiceFrame(voice_id=0, filter_enabled=False, filter_cutoff_q15=10,
                         filter_reso_q15=30000, filter_mode=0),
              VoiceFrame(voice_id=1, filter_enabled=True, filter_cutoff_q15=Q15_ONE,
                         filter_reso_q15=2000, filter_mode=1),
              VoiceFrame(voice_id=2, filter_enabled=True, filter_cutoff_q15=Q15_ONE,
                         filter_reso_q15=3000, filter_mode=2)]
    stage.render_frames(frames, params)
    assert calls == [(9, Q15_ONE, 3000, 3 if global_filter else 1)]


@pytest.mark.parametrize('native_present', [False, True])
def test_capsules_without_enable_keep_legacy_inference(monkeypatch, native_present):
    capsule = capsule_for(authored())
    del capsule.defaults.voice_param_defaults[VoiceParamID.FILTER_ENABLE]
    if not native_present:
        del capsule.sound_extras['native_carrier']
    preset = compiled_preset_from_capsule(capsule)
    assert preset.filter_enabled is None
    engine = owner(preset)
    stage = UnifiedOutputStage()
    stage.configure_preset(preset)
    calls = observe(stage, monkeypatch)
    stage.render_frames([engine.step_voice_simple(0)])
    assert calls == [(9, 32766, 0, 0)]


@pytest.mark.parametrize('old_enabled', [False, True])
@pytest.mark.parametrize('release', [False, True])
def test_filter_ownership_survives_program_change_release_and_scheduler_hold(
        tmp_path, monkeypatch, old_enabled, release):
    old = compiled_preset_from_ir(authored(old_enabled))
    new = compiled_preset_from_ir(authored(not old_enabled))
    synth = host(tmp_path, 'filter-owner', old)
    first = synth.note_on(60, 100)
    synth.render_block()
    if release:
        synth.note_off(60)
    synth.load_compiled_preset(new)
    second = synth.note_on(67, 100)
    calls = observe(synth, monkeypatch)
    synth.render_block()
    for _ in range(3):
        assert synth._last_render_frames[first].filter_enabled is old_enabled
        assert synth._last_render_frames[second].filter_enabled is (not old_enabled)
        synth.render_block(schedule=schedule())
    assert all(row[0] == 9 for row in calls)


@pytest.mark.parametrize('version', range(1, 7))
def test_legacy_voice_snapshots_migrate_detached_and_keep_future_frames(version):
    preset = compiled_preset_from_ir(authored())
    before = owner(replace(preset, filter_enabled=None))
    before.step_voice_simple(0)
    saved = before.voices[0].snapshot()
    saved['snapshot_version'] = version
    del saved['compiled_preset']['fields']['filter_enabled']
    original = deepcopy(saved)
    restored = VoiceEngine(max_voices=2)
    restored.voices[0] = VoiceStateWithProgram.restore(json.loads(json.dumps(saved)))
    assert saved == original
    assert restored.voices[0].compiled_preset.filter_enabled is None
    assert restored.voices[0].snapshot()['snapshot_version'] == 7
    for _ in range(12):
        assert before.step_voice_simple(0) == restored.step_voice_simple(0)


@pytest.mark.parametrize('enabled', [False, True, None])
def test_current_snapshot_keeps_enable_and_requires_the_field(enabled):
    before = owner(replace(compiled_preset_from_ir(authored()), filter_enabled=enabled))
    saved = before.voices[0].snapshot()
    restored = VoiceStateWithProgram.restore(json.loads(json.dumps(saved)))
    assert restored.compiled_preset.filter_enabled is enabled
    del saved['compiled_preset']['fields']['filter_enabled']
    with pytest.raises(ValueError, match='Incomplete'):
        VoiceStateWithProgram.restore(saved)


@pytest.mark.parametrize('value', [1, 'false', 0.5])
def test_invalid_authored_enable_does_not_become_truthy(value):
    ir = authored(value)
    with pytest.raises(ValueError, match='boolean'):
        compiled_preset_from_ir(ir)


def test_existing_boolean_wire_entry_preserves_program_scope_and_canonical_order():
    values = {VoiceParamID.FILTER_ENABLE: 1, VoiceParamID.FM6_TOPO_BLEND: 1234,
              VoiceParamID.FM6_OP0_RATIO: 65536, VoiceParamID.TREE_ID: 2}
    blob = pack_voice_params(values)
    assert blob == pack_voice_params(dict(reversed(list(values.items()))))
    assert struct.pack('<BBHh', 1, 3, 97, 1) in blob  # Existing 6-byte BOOL entry.
    registry = get_central_registry().voice_engine
    order = []
    offset = 0
    while offset < len(blob):
        persistence, kind, param_id = struct.unpack_from('<BBH', blob, offset)
        entry = registry.get(param_id)
        assert persistence == entry.persistence and kind == entry.value_type
        order.append((persistence, int(entry.domain), param_id))
        offset += 8 if kind in (int(ValueType.U32), int(ValueType.I32)) else 6
    assert offset == len(blob) and order == sorted(order)
    assert unpack_voice_params(blob) == values
    assert unpack_voice_params(struct.pack('<BBHh', 0, 3, 97, 1)) == {97: 1}


@pytest.mark.parametrize('value,expected', [(-9, False), (0, False), (1, True), (9, True)])
def test_typed_enable_load_clamps_registry_range(value, expected):
    capsule = capsule_for(authored())
    capsule.defaults.voice_param_defaults[VoiceParamID.FILTER_ENABLE] = value
    assert compiled_preset_from_capsule(capsule).filter_enabled is expected


def test_legacy_emitter_adapter_without_enable_keeps_its_prior_frames():
    from types import SimpleNamespace
    from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter

    preset = replace(compiled_preset_from_ir(authored()), filter_enabled=None)
    fields = dict(vars(preset))
    del fields['filter_enabled']
    adapter = SimpleNamespace(**fields, backbone_ops=preset.backbone_ops,
                              backbone_families=preset.backbone_families)
    old, explicit_legacy = VoiceFrameEmitter(adapter), VoiceFrameEmitter(preset)
    for emitter in (old, explicit_legacy):
        emitter.note_on(60, 24000)
    for block in range(12):
        if block == 7:
            old.note_off()
            explicit_legacy.note_off()
        a, b = old.emit_frame(), explicit_legacy.emit_frame()
        assert a == b and a.filter_enabled is None
