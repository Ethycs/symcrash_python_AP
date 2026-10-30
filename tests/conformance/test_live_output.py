"""Shared final output and live-device adapters, without opening an audio device."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sys
import time
import uuid

import numpy as np
import pytest
import yaml

from symcrash_python_ap.audio.block_stream import AudioBlockReader
from symcrash_python_ap.audio.live_renderer import LiveSynthRenderer
from symcrash_python_ap.control_frame import ControlFrame, GlobalControls
from symcrash_python_ap.control_registry import ControlID
from symcrash_python_ap.frames import ParamFrame, VoiceFrame, VoiceFramePacketV1, VoiceFlags
from symcrash_python_ap.instrument_ir import SafetyConfig
from symcrash_python_ap.library.preset_compiler import PresetCompiler
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter


@pytest.fixture(params=[False, True], ids=['python','jit'])
def execution_path(request):
    old = hot.jit_enabled()
    if request.param and not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    hot.set_jit_enabled(request.param)
    try:
        yield
    finally:
        hot.set_jit_enabled(old)


def frame(voice, seq, *, note=69, pan=0, events=False):
    p = VoiceFramePacketV1(voice_id=voice, frame_seq=seq, flags=VoiceFlags.GATE,
                           note_u8=note, gain_q15=24000, pan_q15=pan,
                           sendA_q15=12000, carrier_role_u8=1)
    if events:
        p.event_mask = 1 << 5
        p.event_strength_q15[5] = 15000
        p.event_payload_u64[5] = 5 << 60 | 2 << 56 | 128 << 16 | 3 << 8
    return p.to_voice_frame()


def host(tmp_path):
    result = SynthMaster(SynthConfig(n_voices=2, output_dir=str(tmp_path/'audio'),
                                     library_dir=str(tmp_path/'library')))
    result.disable('cockpit')
    return result


@pytest.mark.parametrize('all_on', [False,True])
def test_unified_and_master_share_real_polyphonic_output_and_state(tmp_path, execution_path, all_on):
    master, stage = host(tmp_path), UnifiedOutputStage()
    config = SafetyConfig(dc_removal=all_on, master_ola=all_on, slew_limit_q15=1500)
    for renderer in (master,stage):
        renderer.configure_safety(config)
        renderer.set_material(1)
        renderer.configure_aux('A', 6, ParamFrame(drive_q15=12000), 16000)
    for block in range(32):
        seq = max(1,block)  # Duplicate first packet may not replay the material hit.
        frames = ([frame(0,seq,note=60,pan=-10000,events=block<2),
                   frame(1,seq,note=79,pan=14000)] if block<24 else [])
        params = ParamFrame(K_mode=2, drive_q15=7000, gate_q15=9000 if block%5==0 else 32767)
        expected = master.render_frames(deepcopy(frames), params=params)[:2]
        actual = stage.render_frames(deepcopy(frames), params)
        assert actual == expected, block
        assert asdict(stage.safety.rg) == asdict(master.safety.rg)
        assert stage.safety.master_ola_L == master.safety.master_ola_L
        assert stage.last_post_mix_params == master.last_post_mix_params
        assert stage.post_mix_controls.bursts == master.post_mix_controls.bursts
        assert stage.auxiliary_returns.buses['A'].engine.delay_L.write_idx == master.auxiliary_returns.buses['A'].engine.delay_L.write_idx


def test_output_clock_advances_once_for_a_batch_and_for_silence(monkeypatch):
    stage = UnifiedOutputStage()
    clocks = []
    monkeypatch.setattr(stage.tuning, 'advance_block', lambda: clocks.append(1))
    stage.render_voice(frame(0,1))
    assert clocks == []  # Raw VM inspection does not own output time.
    stage.render_frames([frame(0,2),frame(1,1)])
    stage.render_frames([])
    assert clocks == [1,1]
    assert stage.safety.master_ola_L


def test_stale_and_rejected_packets_do_not_replay_consumer_events(tmp_path):
    master, stage = host(tmp_path), UnifiedOutputStage()
    for frames in ([frame(0,4,events=True)], [frame(0,3,events=True)], [], [frame(0,5)]):
        assert stage.render_frames(deepcopy(frames)) == master.render_frames(deepcopy(frames))[:2]
        assert stage.post_mix_controls.bursts == master.post_mix_controls.bursts
    bad = frame(1,1,events=True)
    from dataclasses import replace
    bad.wire_data = replace(bad.wire_data, reject_code=1)
    assert stage.render_frames([deepcopy(bad)]) == master.render_frames([bad])[:2]
    assert 1 not in stage.vm_states


@pytest.mark.parametrize('bad', [[object()], [VoiceFrame(voice_id=-1)],
                               [VoiceFrame(voice_id=0),VoiceFrame(voice_id=0)]])
def test_invalid_batches_fail_before_allocating_or_advancing(bad):
    stage = UnifiedOutputStage()
    with pytest.raises((TypeError,ValueError)):
        stage.render_frames(bad)
    assert stage.vm_states == {}
    assert stage.safety.master_ola_L == []


def test_mono_downmix_precedes_safety_and_preserves_joint_constraints(execution_path):
    stage = UnifiedOutputStage()
    stage.configure_safety(SafetyConfig(dc_removal=True, master_ola=True,
                                        slew_limit_q15=301, limiter_threshold_q15=10001))
    previous = 0
    for seq in range(1,50):
        values = stage.render_mono_frames([frame(0,seq,note=96,pan=-20000),frame(1,seq,note=48,pan=32000)])
        assert max(abs(x) for x in values) <= 10001
        assert max(abs(b-a) for a,b in zip([previous]+values,values)) <= 301
        assert abs(sum(values)) <= 64*2048
        if seq>1:
            assert values[0] == previous
        previous = values[-1]
        assert stage.safety.prev_sample_L == stage.safety.prev_sample_R


def test_live_renderer_preserves_two_notes_and_uses_control_owned_pitch(execution_path):
    renderer = LiveSynthRenderer()
    renderer.note_on(60)
    renderer.note_on(79)
    heard = False
    previous = 0
    for _ in range(24):
        values = renderer.render_mono_block()
        heard |= any(values)
        assert max(abs(x) for x in values) <= 32000
        assert max(abs(b-a) for a,b in zip([previous]+values,values)) <= 2000
        previous = values[-1]
    assert heard
    frames = [item.frame for item in renderer.last_engine_output.render_batch]
    assert {f.note for f in frames} == {60,79}
    assert len({f.f0_hz for f in frames}) == 2
    assert set(renderer.stage.vm_states) == {0,1}
    renderer.note_off(60)
    renderer.render_mono_block()
    from symcrash_python_ap.voice.voice_engine import VoicePhase
    phases = {voice.note_u8: voice.phase for voice in renderer.engine.voices.values()}
    assert phases[60] == VoicePhase.RELEASE
    assert phases[79] != VoicePhase.RELEASE


def test_reserved_pot_and_master_gain_follow_canonical_controls():
    a,b = LiveSynthRenderer(),LiveSynthRenderer()
    a.note_on(60)
    b.note_on(60)
    b.set_pot(6,32767)  # Reserved; must not silently become a harmonic multiplier.
    for _ in range(16):
        assert a.render_mono_block() == b.render_mono_block()
    b.set_pot(9,0)
    for _ in range(150):
        output = b.render_mono_block()
    assert b.last_engine_output.param_frame.gate_q15 == 0
    assert output == [0]*64


def test_live_callback_segmentation_neither_drops_nor_pads_samples(execution_path):
    a,b = LiveSynthRenderer(),LiveSynthRenderer()
    a.note_on(69)
    b.note_on(69)
    expected = np.concatenate([a.render_mono_float() for _ in range(8)])
    chunks = []
    for count in (1,63,17,8,100,5,127,191):
        chunk = np.empty(count, dtype=np.float32)
        b.fill_mono(chunk)
        chunks.append(chunk)
    assert sum(map(len,chunks)) == len(expected)
    np.testing.assert_array_equal(np.concatenate(chunks),expected)
    assert a.engine.block_count == b.engine.block_count == 8
    assert np.any(expected)
    np.testing.assert_array_equal(expected*32768,np.rint(expected*32768))


@pytest.mark.parametrize('channels', [1,2])
def test_queued_callback_reader_preserves_partial_blocks_and_reports_underruns(channels):
    source = np.arange(192*channels,dtype=np.float32).reshape(192,channels)/1024
    blocks = iter(np.split(source,3))
    reader = AudioBlockReader(lambda: next(blocks,None), channels)
    chunks = []
    for count in (17,80,3,92):
        result = np.full((count,channels),-1,dtype=np.float32)
        reader.fill(result)
        chunks.append(result)
    np.testing.assert_array_equal(np.concatenate(chunks),source)
    assert reader.underruns == 0
    empty = np.ones((9,channels),dtype=np.float32)
    reader.fill(empty)
    assert not np.any(empty) and reader.underruns == 1


def test_emitter_output_api_leaves_master_gain_for_effect_returns():
    preset = PresetCompiler().compile({'preset': {'preset_id': 'gain'}})
    emitter = VoiceFrameEmitter(preset)
    emitter.note_on(60)
    for _ in range(8):
        emitter.emit_output_block()
    emitter.consume_control_frame(ControlFrame(globals=GlobalControls(master_gain_q15=0)))
    frame_value, params = emitter.emit_output_block()
    assert frame_value.gain_q15 > 0
    assert params.gate_q15 == 0
    assert emitter.block_index == 9


def import_tool(name):
    tools_path = str(Path(__file__).resolve().parents[2]/'tools/synth')
    if tools_path not in sys.path:
        sys.path.insert(0,tools_path)
    return importlib.import_module(name)


def test_pico2_applies_preset_settings_and_post_effect_master_gain(tmp_path,monkeypatch):
    module = import_tool('pico2_live_synth')
    template = Path(__file__).resolve().parents[2]/'presets/instruments/dx_marimba_v0.yaml'
    data = yaml.safe_load(template.read_text(encoding='utf-8'))
    data['instrument_ir_v0'].setdefault('compile',{})['safety'] = {
        'limiter_threshold_q15': 5000, 'slew_limit_q15': 301, 'dc_removal':True}
    path = tmp_path/'preset.yaml'
    path.write_text(yaml.safe_dump(data),encoding='utf-8')
    synth = module.Pico2LiveSynth(virtual_mode=True,preset_path=str(path))
    assert synth.stage.safety.LIMIT_CEILING == 5000
    assert synth.stage.safety.max_slew == 301
    synth.stage.disable_output_stage('safety')
    monkeypatch.setattr(synth.stage.material,'step_block',lambda *args,**kwargs: ([16000]*64,[16000]*64))
    monkeypatch.setattr(synth.cockpit,'tick',lambda: ControlFrame(globals=GlobalControls(master_gain_q15=0)))
    synth._render_block()
    left,right = synth.audio_queue.get_nowait()
    assert left == right == [0]*64


def test_pico2_notes_are_consumed_once_on_the_render_thread(monkeypatch):
    synth = import_tool('pico2_live_synth').Pico2LiveSynth(virtual_mode=True)
    original = synth.emitter._engine.note_on_simple
    calls = []
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(synth.emitter._engine,'note_on_simple',counted)
    synth.note_on(72)
    assert calls == [] and synth.stage.vm_states == {}
    synth._render_block()
    assert calls == [1]
    assert synth.emitter.voice_state.note_u8 == 72
    synth._render_block()
    assert calls == [1]


def test_standalone_callback_uses_48khz_and_complete_output():
    module = import_tool('standalone_synth')
    synth = module.StandaloneSynth()
    assert module.SAMPLE_RATE == 48000
    synth.note_on(69)
    output = np.empty((137,1),dtype=np.float32)
    synth._audio_callback(output,137,None,None)
    assert np.any(output)
    assert synth.engine.block_count == 3
    assert synth.stage.safety.master_ola_L


@pytest.mark.parametrize('tool_name', ['pico2_live_synth','virtual_synth_mp'])
def test_actual_device_callbacks_preserve_partial_blocks(monkeypatch,tool_name):
    module = import_tool(tool_name)
    callbacks = []
    class FakeStream:
        def __init__(self, **kwargs):
            assert kwargs['samplerate'] == 48000
            callbacks.append(kwargs['callback'])
        def start(self):
            pass
    monkeypatch.setattr(module.sd,'OutputStream',FakeStream)
    if tool_name == 'pico2_live_synth':
        app = module.Pico2LiveSynth(virtual_mode=True)
        monkeypatch.setattr(app,'_find_output_device',lambda: None)
        left = list(range(128))
        for start in (0,64):
            app.audio_queue.put((left[start:start+64],[-x for x in left[start:start+64]]))
        expected = np.column_stack((left,[-x for x in left])).astype(np.float32)/32768
    else:
        app = module.VirtualSynthApp()
        expected = np.arange(128,dtype=np.float32).reshape(-1,1)/32768
        blocks = iter(np.split(expected[:,0],2))
        monkeypatch.setattr(app.synth,'read_block',lambda: next(blocks,None))
    app._start_audio()
    outputs = []
    for count in (17,1,110):
        out = np.zeros((count,expected.shape[1]),dtype=np.float32)
        callbacks[0](out,count,None,None)
        outputs.append(out)
    np.testing.assert_array_equal(np.concatenate(outputs),expected)
    assert app.block_reader.underruns == 0


def test_simple_audition_tool_is_import_safe_and_renders_a_release():
    module = import_tool('standalone_synth_simple')
    samples = module.render_note(60,0.01)
    assert module.SAMPLE_RATE_HZ == 48000
    assert np.any(samples)
    assert len(samples) > 480
    assert not np.any(samples[-64:])
    assert np.max(np.abs(np.diff(np.concatenate(([0],samples))))) <= 2000/32768


def test_shared_memory_worker_renders_complete_safe_blocks():
    from symcrash_python_ap.audio.multiprocess_synth import (
        SharedRingBuffer, render_process_main, MSG_NOTE_ON, MSG_STOP)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    name = 'symcrash_test_'+uuid.uuid4().hex
    ring = SharedRingBuffer(name,64,16,create=True)
    process = ctx.Process(target=render_process_main,args=(name,queue,64,16))
    try:
        queue.put((MSG_NOTE_ON,60,100))
        queue.put((MSG_NOTE_ON,79,100))
        process.start()
        blocks = []
        deadline = time.monotonic()+45
        while len(blocks)<32 and time.monotonic()<deadline:
            block = ring.read()
            if block is not None:
                blocks.append(block)
            else:
                if process.exitcode is not None:
                    pytest.fail(f'Render process exited early: {process.exitcode}')
                time.sleep(0.005)
        assert len(blocks)==32
        samples = np.concatenate(blocks)
        assert np.any(samples)
        assert np.max(np.abs(samples)) <= 32000/32768
        assert np.max(np.abs(np.diff(np.concatenate(([0],samples))))) <= 2000/32768
        queue.put((MSG_STOP,))
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        ring.close()
        ring.unlink()
        queue.close()
        queue.join_thread()
