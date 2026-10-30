"""Packet controls, native send transport, and the shared stereo output path."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import pytest

from symcrash_python_ap.audio_routing import AuxiliaryReturns, PostMixControls
from symcrash_python_ap.frames import MaterialParams, ParamFrame, VoiceFlags, VoiceFramePacketV1
from symcrash_python_ap.material.material_engine import MaterialEngine
from symcrash_python_ap.safety.safety_engine import SafetyEngine
from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.scheduling import ExecutionPolicy, ExecutionReport
from symcrash_python_ap.voice.scheduler import ScheduleResult, VoiceScheduleInfo, HealthTelemetry
from symcrash_python_ap.wire_frame import WireEvent
from symcrash_python_ap.wavelet.sine_core import cosine_q15, sine_q15


def clamp(x):
    return max(-32768, min(32767, x))


def multiply(a, b):
    # Independent exact arithmetic oracle, including negative half-way values.
    return clamp(round(Fraction(int(a) * int(b), 32768)))


def event(lane=5, variant=0, strength=32767, p0=128, p1=1):
    return WireEvent(lane, variant, strength, 0, 0, 0, p0, p1, 0)


def frame(*, voice=0, seq=1, events=(), **kwargs):
    p = VoiceFramePacketV1(voice_id=voice, frame_seq=seq, flags=VoiceFlags.GATE,
                          gain_q15=12000, carrier_role_u8=1, **kwargs)
    for e in events:
        p.event_mask |= 1 << e.lane
        p.event_strength_q15[e.lane] = e.strength_q15
        # Literal frozen layout; no production payload encoder.
        p.event_payload_u64[e.lane] = (e.lane << 60 | e.variant << 56 |
                                       e.param0 << 16 | e.param1 << 8)
    return p.to_voice_frame()


def schedule(*ids):
    return ScheduleResult([VoiceScheduleInfo(i, True, True) for i in ids], 0, False,
                          HealthTelemetry(voices_serviced=len(ids)))


@pytest.fixture
def synth(tmp_path):
    host = SynthMaster(SynthConfig(output_dir=str(tmp_path / 'audio')))
    host.disable('cockpit')
    return host


@pytest.mark.parametrize('variant', [0, 1, 2])
@pytest.mark.parametrize('duration', [0, 1, 3])
def test_burst_lifetime_includes_arrival_and_uses_independent_q8_oracle(variant, duration):
    controls = PostMixControls()
    base = ParamFrame(drive_q15=1000, bias_q15=-100)
    burst = event(variant=variant, strength=513, p0=128, p1=duration)
    amount = round(Fraction(513 * 128, 256))  # 256.5 -> 256, ties to even.
    for tick in range(max(1, duration) + 2):
        current = controls.step(base, [(7, burst)] if tick == 0 else [])
        alive = tick < max(1, duration)
        assert current.drive_q15 == 1000 + (amount if alive and variant != 1 else 0)
        assert current.bias_q15 == -100 + (amount if alive and variant != 0 else 0)
    assert (base.drive_q15, base.bias_q15) == (1000, -100)
    assert controls.bursts == []


def test_overlapping_bursts_expire_independently_and_saturate_once():
    controls = PostMixControls()
    base = ParamFrame(drive_q15=30000, bias_q15=-20000)
    a = event(variant=2, p0=255, p1=3)
    b = event(variant=2, p0=255, p1=1)
    assert controls.step(base, [(0, a)]).bias_q15 == -20000 + 32639
    full = controls.step(base, [(1, b)])
    assert (full.drive_q15, full.bias_q15) == (32767, 32767)
    assert controls.step(base).bias_q15 == 12639
    assert controls.step(base) == base


def test_global_k_order_latches_without_rewriting_the_param_frame():
    controls = PostMixControls()
    base = ParamFrame(K_mode=1)
    params = controls.step(base, [(7, event(3, p0=3, p1=0)), (2, event(3, p0=0, p1=0))])
    assert params.K_mode == 3
    assert controls.step(base).K_mode == 3
    assert base.K_mode == 1
    unsupported = event(3, p0=9, p1=2)
    cloud = event(6)
    assert controls.step(base, [(0, unsupported), (1, cloud)]).K_mode == 3
    assert controls.oversample_request == 2
    assert controls.unhandled_events == ((0, unsupported), (1, cloud))


def test_vm_delivers_consumers_once_and_never_on_held_rejected_or_stale_blocks():
    events = [event(3, p0=2, p1=0), event(5), event(6)]
    current = frame(events=events)
    state, report = VMState(), ExecutionReport()
    vm_step_block(state, current, report=report)
    assert report.consumer_events == tuple(events)
    vm_step_block(state, current, report=report)
    assert report.serviced and report.consumer_events == ()  # Continuous buses still run.
    for candidate, policy in [(frame(seq=0, events=events), None),
                              (frame(seq=2, events=events), ExecutionPolicy(serviced=False)),
                              (frame(seq=2, events=events, reject_code=1), None)]:
        before = json.dumps(state.snapshot(), sort_keys=True)
        vm_step_block(state, candidate, policy=policy, report=report)
        assert not report.serviced and report.consumer_events == ()
        assert json.dumps(state.snapshot(), sort_keys=True) == before
    vm_step_block(state, frame(seq=2, events=events), report=report)
    assert report.consumer_events == tuple(events)


def test_aux_inputs_use_wide_sums_and_independent_q15_rounding():
    buses = AuxiliaryReturns(block_size=4)
    inputs = buses.empty_inputs()
    voices = [([30000, -30000, 1, -3], [-30000, 30000, -1, 3], 32767, 16384),
              ([30000, -30000, 1, -3], [-30000, 30000, -1, 3], 32767, 16384),
              ([-30000, 30000, 1, -3], [30000, -30000, -1, 3], 32767, 16384)]
    for left, right, a, b in voices:
        buses.add_voice(inputs, left, right, a, b)
    assert buses.process(inputs) == ([0] * 4, [0] * 4)  # No invented default FX.
    for name, index in [('A', 2), ('B', 3)]:
        expected = tuple([clamp(sum(multiply(v[ch][i], v[index]) for v in voices))
                          for i in range(4)] for ch in (0, 1))
        assert buses.last_inputs[name] == expected
    assert buses.last_inputs['A'][0][0] == 29999  # Clipping each addition would fail.


def test_aux_engines_are_independent_configuration_is_detached_and_reconfigure_keeps_state():
    buses = AuxiliaryReturns()
    params = ParamFrame(drive_q15=1000)
    buses.configure('A', 6, params)
    buses.configure('B', 6, params)
    a, b = buses.buses['A'], buses.buses['B']
    params.drive_q15 = 9999
    assert a.params.drive_q15 == b.params.drive_q15 == 1000
    assert a.engine is not b.engine
    buses.process(buses.empty_inputs())
    assert a.engine.delay_L.write_idx == b.engine.delay_L.write_idx == 64
    buses.configure('A', 7, ParamFrame())
    assert buses.buses['A'].engine is a.engine
    buses.configure('A', None)
    assert buses.buses['A'] is None and buses.buses['B'] is b


@pytest.mark.parametrize('name,material,params,gain,error', [
    ('C', 0, ParamFrame(), 32767, ValueError), ('A', 11, ParamFrame(), 32767, ValueError),
    ('A', 0, None, 32767, TypeError), ('A', 0, ParamFrame(), -1, ValueError),
])
def test_invalid_aux_configuration_is_atomic(name, material, params, gain, error):
    buses = AuxiliaryReturns()
    buses.configure('A', 1, ParamFrame())
    before = buses.buses.copy()
    with pytest.raises(error):
        buses.configure(name, material, params, gain)
    assert buses.buses == before


def test_shared_pipeline_matches_independent_stereo_routing_and_actual_dsp(synth):
    params = ParamFrame(gate_q15=14000, drive_q15=10000, tone_q15=22000)
    aux_params = ParamFrame(drive_q15=7000, tone_q15=12000)
    synth.set_material(1)
    synth.configure_aux('A', 2, aux_params, 19000)
    synth.configure_aux('B', 0, ParamFrame(), 9000)
    states = [VMState.create(i) for i in range(2)]
    main, aux_a, aux_b, safety = MaterialEngine(), MaterialEngine(), MaterialEngine(), SafetyEngine()
    # Documented host CCS profile; standalone Safety defaults enable DC/OLA.
    safety.LIMIT_CEILING = safety.LIMIT_THRESHOLD = 32000
    safety.max_slew = 2000
    for seq in range(1, 7):
        frames = [frame(voice=i, seq=seq, pan_q15=(-8000 if i == 0 else 12000),
                        sendA_q15=18000, sendB_q15=9000) for i in range(2)]
        dry = [[0] * 64, [0] * 64]
        sends = {name: [[0] * 64, [0] * 64] for name in ('A', 'B')}
        for state, current in zip(states, frames):
            _, mono = vm_step_block(state, current)
            phase = (current.pan_q15 + 32768) << 14
            for ch, gain in enumerate((cosine_q15(phase), sine_q15(phase))):
                panned = [multiply(x, gain) for x in mono]
                for j, sample in enumerate(panned):
                    dry[ch][j] += sample
                    sends['A'][ch][j] += multiply(sample, 18000)
                    sends['B'][ch][j] += multiply(sample, 9000)
        dry = [[clamp(x) for x in ch] for ch in dry]
        sends = {name: tuple([clamp(x) for x in ch] for ch in bus) for name, bus in sends.items()}
        processed = main.step_block(*dry, params, 1)
        a = aux_a.step_block(*sends['A'], aux_params, 2)
        b = aux_b.step_block(*sends['B'], ParamFrame(), 0)
        summed = [[multiply(clamp(x + multiply(y, 19000) + multiply(z, 9000)), 14000)
                   for x, y, z in zip(processed[ch], a[ch], b[ch])] for ch in (0, 1)]
        expected = safety.step_block(*summed, params, apply_master_ola=False, apply_dc_removal=False)
        left, right, _ = synth.render_frames(reversed(frames), params=params)
        assert (left, right) == expected
        assert synth.auxiliary_returns.last_inputs == sends
    assert any(left) and any(right)
    assert synth.voice_engine.block_count == 0


def test_packet_controls_reach_material_and_safety_and_bursts_advance_without_voices(synth, monkeypatch):
    observed = []
    original = synth.safety.step_block
    def safety(left, right, params, **kwargs):
        observed.append(params.K_mode)
        return original(left, right, params, **kwargs)
    monkeypatch.setattr(synth.safety, 'step_block', safety)
    synth.set_material(1)
    baseline = deepcopy(synth.material)
    base = ParamFrame(drive_q15=8192)
    first = frame(events=[event(3, p0=3, p1=0), event(5, variant=2, p1=3)])
    synth.render_frames([first], params=base)
    assert synth.last_post_mix_params.K_mode == 3
    assert synth.material.params.drive_q15 == 8320  # RNE(16384 * 256 / 32768).
    assert synth.material.params.bias_q15 == 128
    assert synth.material.tape_L != baseline.tape_L
    synth.render_frames([first], params=base)  # Duplicate cannot add another burst.
    assert len(synth.post_mix_controls.bursts) == 1
    synth.render_frames([], params=base)
    assert synth.last_post_mix_params.drive_q15 == 24576
    synth.render_frames([], params=base)
    assert synth.last_post_mix_params.drive_q15 == 8192
    assert synth.post_mix_controls.bursts == []
    assert observed == [3] * 4


def test_material_burst_changes_audio_and_explicit_global_filter_is_consumed(synth):
    synth.disable('safety')
    synth.set_material(1)
    control = deepcopy(synth)
    hit = frame(events=[event(5, variant=2, p1=4)])
    actual = synth.render_frames([hit])[:2]
    without = control.render_frames([frame()])[:2]
    assert actual != without
    # The global filter can also be selected without a native voice cutoff.
    filtered = deepcopy(control)
    a = control.render_frames([frame(seq=2)])[:2]
    b = filtered.render_frames([frame(seq=2)], params=ParamFrame(
        filter_enabled=True, filter_cutoff_q15=14000, filter_reso_q15=4000))[:2]
    assert a != b


def test_full_packet_output_matches_between_python_and_jit(synth):
    from symcrash_python_ap.vm import hotpath_jit
    previous = hotpath_jit.jit_enabled()
    traces = []
    try:
        for jit in (False, True):
            hotpath_jit.set_jit_enabled(jit)
            host = deepcopy(synth)
            host.set_material(1)
            host.configure_aux('B', 7, ParamFrame(pressure_q15=20000))
            trace = []
            for seq in range(1, 10):
                events = [event(3, p0=2, p1=0), event(5, variant=2, p1=3)] if seq == 1 else []
                current = frame(seq=seq, events=events, sendB_q15=12000)
                selection = schedule(0) if seq % 3 else schedule()
                trace.append(host.render_frames([current], schedule=selection)[:2])
            traces.append(trace)
    finally:
        hotpath_jit.set_jit_enabled(previous)
    assert traces[0] == traces[1]
    assert any(x for left, right in traces[0] for x in left + right)


def test_holding_packet_keeps_accepted_sends_pan_and_vm_state(synth):
    synth.disable('safety')
    synth.render_frames([frame(pan_q15=-32768, sendA_q15=16384)])
    state = synth._voice_slots[0].vm_state
    before = json.dumps(state.snapshot(), sort_keys=True)
    pending = frame(seq=2, pan_q15=32767, sendB_q15=32767, events=[event(5)])
    synth.render_frames([pending], schedule=schedule())
    assert any(synth.auxiliary_returns.last_inputs['A'][0])
    assert not any(synth.auxiliary_returns.last_inputs['A'][1])
    assert not any(synth.auxiliary_returns.last_inputs['B'][0])
    assert not any(synth.auxiliary_returns.last_inputs['B'][1])
    assert synth.post_mix_controls.bursts == []
    assert json.dumps(state.snapshot(), sort_keys=True) == before
    synth.render_frames([pending], schedule=schedule(0))
    assert any(synth.auxiliary_returns.last_inputs['B'][1])
    assert synth.last_post_mix_params.drive_q15 > 8192


def test_rejected_packet_never_initializes_voice_or_consumer(synth):
    rejected = frame(reject_code=1, events=[event(3, p0=3), event(5)], sendA_q15=32767)
    left, right, _ = synth.render_frames([rejected])
    assert not any(left + right)
    assert synth._voice_slots[0].vm_state is None
    assert synth.post_mix_controls.k_mode is None and not synth.post_mix_controls.bursts
    assert not synth.last_execution_reports[0].serviced


def test_aux_delay_tail_continues_after_voice_disappears_and_master_mutes_it(synth):
    synth.disable('safety')
    params = ParamFrame(drive_q15=88, tone_q15=0, pressure_q15=32767)
    synth.configure_aux('A', 6, params)
    bus = synth.auxiliary_returns.buses['A']
    # Start from an exact known DSP state to isolate routing/tail lifetime.
    # Proportional slew has a quantization deadband, not exact target arrival.
    bus.engine.params = MaterialParams(drive_q15=88, tone_q15=0, pressure_q15=32767)
    synth.render_frames([frame(pan_q15=-32768, sendA_q15=32767)])
    pos = bus.engine.delay_L.write_idx
    first, _, _ = synth.render_frames([])  # Delay = 128 samples, not yet returned.
    left, right, _ = synth.render_frames([])
    assert bus.engine.delay_L.write_idx == pos + 128
    assert any(left) and not any(right)
    synth.render_frames([frame(seq=2, pan_q15=-32768, sendA_q15=32767)])
    synth.render_frames([], params=ParamFrame(gate_q15=0))
    left, right, _ = synth.render_frames([], params=ParamFrame(gate_q15=0))
    assert not any(left + right)


@pytest.mark.parametrize('bad', ['duplicate', 'slot', 'fractional_slot', 'boolean_slot', 'schedule'])
def test_external_validation_precedes_render_state_changes(synth, bad):
    frames, selection = [frame()], None
    if bad == 'duplicate':
        frames *= 2
    elif bad == 'slot':
        frames[0].voice_id = 8
    elif bad in ('fractional_slot', 'boolean_slot'):
        frames.append(frame(voice=1))
        frames[-1].voice_id = 1.5 if bad == 'fractional_slot' else True
    else:
        selection = schedule(1)
    with pytest.raises(ValueError):
        synth.render_frames(frames, schedule=selection)
    assert all(slot.vm_state is None for slot in synth._voice_slots)
    assert not synth._audio_L


@pytest.mark.parametrize('frames,params', [([object()], None), ([frame()], {})])
def test_external_type_errors_precede_render_state_changes(synth, frames, params):
    with pytest.raises(TypeError):
        synth.render_frames(frames, params=params)
    assert all(slot.vm_state is None for slot in synth._voice_slots)
    assert not synth._audio_L


def authored_ir():
    import yaml
    from symcrash_python_ap.instrument_ir import load_instrument_from_dict
    root = Path(__file__).resolve().parents[2]
    data = yaml.safe_load((root / 'presets/instruments/acoustic_marimba_v0.yaml').read_text(encoding='utf-8'))
    graph = data['instrument_ir_v0']['graph']
    graph['amp1']['output'] = 'aux'
    graph['aux'] = {'type': 'FXSend', 'send_a': 0.25, 'send_b': 0.5}
    return load_instrument_from_dict(data)


@pytest.mark.parametrize('codec', ['direct', 'bytes', 'bundle'])
def test_authored_sends_survive_capsule_and_native_voice_render(synth, tmp_path, codec):
    from symcrash_python_ap.library.compiler import CapsuleCompiler
    from symcrash_python_ap.library.capsule_codec import (
        capsule_from_bytes, capsule_to_bytes, save_capsule_bundle, load_capsule_bundle)
    from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir, compiled_preset_from_capsule
    ir = authored_ir()
    if codec == 'direct':
        preset = compiled_preset_from_ir(ir)
    else:
        compiled = CapsuleCompiler().compile_from_ir(ir)
        assert compiled.success, compiled.errors
        capsule = compiled.capsule
        if codec == 'bytes':
            capsule = capsule_from_bytes(capsule_to_bytes(capsule))
        else:
            save_capsule_bundle(capsule, str(tmp_path / 'capsule'))
            capsule = load_capsule_bundle(str(tmp_path / 'capsule'))
        preset = compiled_preset_from_capsule(capsule)
    assert (preset.sendA_q15, preset.sendB_q15) == (8192, 16384)
    synth.load_compiled_preset(preset)
    vid = synth.note_on(60)
    for _ in range(8):
        synth.render_block()
    current = synth._last_render_frames[vid]
    assert (current.sendA_q15, current.sendB_q15) == (8192, 16384)
    assert any(synth.auxiliary_returns.last_inputs['A'][0])
    assert synth.auxiliary_returns.last_inputs['A'] != synth.auxiliary_returns.last_inputs['B']
    packet = VoiceFramePacketV1.from_voice_frame(current)
    assert (packet.sendA_q15, packet.sendB_q15) == (8192, 16384)


@pytest.mark.parametrize('value', [-0.1, 1.1, float('nan'), float('inf'), '0.5'])
def test_invalid_authored_sends_are_rejected(value):
    from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir
    ir = authored_ir()
    ir.layers.L0_graph.nodes[-1].params.send_a = value
    with pytest.raises(ValueError, match='FXSend'):
        compiled_preset_from_ir(ir)


def test_cmmm_send_state_survives_snapshot_and_abi_conversion():
    from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
    engine = VoiceEngine()
    vid = engine.allocate_voice()
    engine.note_on(vid)
    voice = engine.voices[vid]
    voice.sendA_q15, voice.sendB_q15 = 12345, 6789
    restored = VoiceStateWithProgram.restore(voice.snapshot(), {voice.program_id: voice.program})
    assert (restored.sendA_q15, restored.sendB_q15) == (12345, 6789)
    abi = engine.step_voice(vid)
    current = engine._abi_to_voice_frame(abi, vid)
    assert (current.sendA_q15, current.sendB_q15) == (12345, 6789)
