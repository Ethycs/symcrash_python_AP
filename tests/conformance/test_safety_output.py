"""Final emitted audio constraints, independently checked in integer units."""
from copy import deepcopy
from fractions import Fraction
from itertools import product
import random

import numpy as np
import pytest

from symcrash_python_ap.frames import ParamFrame, VoiceFlags, VoiceFramePacketV1
from symcrash_python_ap.safety import safety_engine as safety
from symcrash_python_ap.safety.output_projection import output_feasible, project_output
from symcrash_python_ap.synth_master import SynthConfig, SynthMaster
from symcrash_python_ap.instrument_ir import SafetyConfig
from symcrash_python_ap.vm import hotpath_jit as hot


@pytest.fixture(params=[False, True], ids=['python', 'jit'])
def execution_path(request):
    if request.param and not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    previous = hot.jit_enabled()
    hot.set_jit_enabled(request.param)
    try:
        yield
    finally:
        hot.set_jit_enabled(previous)


def legal(values, previous, peak=32512, slew=8192, dc=2048, anchor=False):
    values = [int(x) for x in values]
    return (all(abs(x) <= peak for x in values)
            and (slew < 0 or all(abs(b-a) <= slew for a,b in zip([previous]+values, values)))
            and (dc < 0 or abs(sum(values)) <= len(values)*dc)
            and (not anchor or values[0] == previous))


@pytest.mark.parametrize('count', [1, 2, 3])
def test_projection_matches_exhaustive_feasibility_and_preserves_legal_input(count):
    # Enumerate every possible output in a small integer domain. This oracle
    # knows only inequalities; it does not reuse the production projection.
    candidates = list(product(range(-2, 3), repeat=count))
    sources = list(product((-3, 0, 3), repeat=count))
    for previous, slew, dc, anchor in product(range(-3, 4), [-1, 0, 1, 2], [-1, 0, 1], [False, True]):
        feasible = [v for v in candidates if legal(v, previous, 2, slew, dc, anchor)]
        assert output_feasible(previous, count, 2, slew, dc, anchor) == bool(feasible)
        if not feasible:
            with pytest.raises(ValueError, match='infeasible'):
                project_output(np.array(sources[0]), previous, 2, slew, dc, anchor)
            continue
        for source in sources + feasible:
            result = project_output(np.array(source), previous, 2, slew, dc, anchor).tolist()
            assert legal(result, previous, 2, slew, dc, anchor), (source, result)
            if source in feasible:
                assert result == list(source)


@pytest.mark.parametrize('ola,slew,dc', list(product([False, True], repeat=3)))
def test_final_invariants_and_history_for_adversarial_streams(execution_path, ola, slew, dc):
    engine = safety.SafetyEngine()
    rng = random.Random(20261030)
    previous = (0, 0)
    for block in range(80):
        patterns = [[32767]*64, [-32768]*64, [-32768,32767,32767,32767]*16,
                    [0]*63+[32767], [32767]*63+[-32768],
                    [rng.randrange(-32768,32768) for _ in range(64)]]
        left = patterns[block % len(patterns)]
        right = patterns[(block+1) % len(patterns)]
        output = engine.step_block(np.array(left, dtype=np.int16), np.array(right, dtype=np.int16),
                                   ParamFrame(K_mode=block % 4), apply_master_ola=ola,
                                   apply_slew_limit=slew, apply_dc_removal=dc)
        for channel, before in zip(output, previous):
            assert legal(channel, before, slew=8192 if slew else -1,
                         dc=2048 if dc else -1, anchor=ola and block > 0)
        assert engine._last_block_prev_L == previous[0]
        assert engine._last_block_prev_R == previous[1]
        previous = tuple(channel[-1] for channel in output)
        assert (engine.prev_sample_L, engine.prev_sample_R) == previous
        assert engine.master_ola_L == output[0][-8:]
        assert engine.master_ola_R == output[1][-8:]
        if slew and dc:
            assert all(result.is_safe for result in engine.check_audio_safety(*output))


def test_reenabled_stages_use_actual_emitted_history(execution_path):
    engine = safety.SafetyEngine()
    previous = (0, 0)
    for block in range(36):
        enabled = block % 2 == 1
        value = 32767 if block % 3 == 0 else -32768
        output = engine.step_block([value]*64, [value]*64, ParamFrame(K_mode=3),
                                   apply_master_ola=enabled, apply_slew_limit=enabled,
                                   apply_dc_removal=enabled)
        for values, p in zip(output, previous):
            assert legal(values, p, slew=8192 if enabled else -1,
                         dc=2048 if enabled else -1, anchor=enabled)
        previous = tuple(v[-1] for v in output)


def test_lookahead_reacts_to_late_peak_and_release_is_slow():
    engine = safety.SafetyEngine()
    left = [10000]*63+[60000]  # Wide DC-blocker intermediate, before limiting.
    right = [10000]*64
    output = engine.apply_limiter(left, right)
    gain = (32512 << 15)//60000
    assert output[0][0] == round(Fraction(10000*gain, 32768))
    assert output[1][0] == output[0][0]  # Same block-peak target for both channels.
    assert max(abs(x) for channel in output for x in channel) <= 32512
    assert engine.lim_L.gain == engine.lim_R.gain == gain
    engine.B = 1
    engine.apply_limiter([1000], [1000])
    assert engine.lim_L.gain == gain + engine.RELEASE_RATE


def test_dc_blocker_handles_opposed_channels_without_premature_clipping():
    engine = safety.SafetyEngine()
    left = [32767]*63+[-32768]
    right = [-x for x in left]
    out_l, out_r = engine.remove_dc(left, right)
    assert abs(sum(out_l)) <= 64*2048
    assert out_r == [-x for x in out_l]
    assert out_l[-1] < -32768  # The following block lookahead handles headroom.
    assert len(set(a-b for a,b in zip(left,out_l))) == 1


@pytest.mark.parametrize('previous', [-32512, 32512])
def test_projection_from_extreme_history_retains_boundary_and_all_bounds(previous):
    rng = random.Random(81)
    for dc, slew in ((2048,8192), (0,8192)):
        for _ in range(16):
            source = np.array([rng.randrange(-32768,32768) for _ in range(64)])
            result = project_output(source, previous, 32512, slew, dc, True)
            assert legal(result, previous, 32512, slew, dc, True)
            mirrored = project_output(-source, -previous, 32512, slew, dc, True)
            assert mirrored.tolist() == (-result).tolist()


def test_lookahead_bound_does_not_wait_for_gain_attack():
    engine = safety.SafetyEngine()
    engine.ATTACK_RATE = 0
    output = engine.apply_limiter([60000]*64, [-60000]*64)
    assert engine.lim_L.gain == engine.lim_R.gain == 32767
    assert all(abs(x) <= 32512 for channel in output for x in channel)


@pytest.mark.parametrize('count', [1, 3, 7, 8, 17, 64, 65])
def test_supported_block_lengths_are_safe_and_match_across_paths(count):
    original = hot.jit_enabled()
    engines = [safety.SafetyEngine(count), safety.SafetyEngine(count)]
    try:
        for block in range(12):
            output = []
            for index, engine in enumerate(engines):
                hot.set_jit_enabled(bool(index))
                output.append(engine.step_block([2000 if block % 2 else -2000]*count,
                                                [1000]*count, ParamFrame()))
                assert all(r.is_safe for r in engine.check_audio_safety(*output[-1]))
            assert output[0] == output[1]
    finally:
        hot.set_jit_enabled(original)


def test_infeasible_short_block_is_rejected_before_state_changes(execution_path):
    engine = safety.SafetyEngine(1)
    engine.step_block([32767], [32767], ParamFrame(),
                      apply_master_ola=False, apply_slew_limit=False, apply_dc_removal=False)
    before = deepcopy(vars(engine))
    with pytest.raises(ValueError, match='infeasible'):
        engine.step_block([0], [0], ParamFrame())
    # Use representations for the identity-equality SymbolTail object.
    for name in ('rg', 'lim_L', 'lim_R', 'q_gain', 'q_hf_cut', 'prev_sample_L',
                 'prev_sample_R', 'master_ola_L', 'master_ola_R', '_last_block_prev_L',
                 '_last_block_prev_R'):
        assert getattr(engine, name) == before[name], name


@pytest.mark.parametrize('value', [2048, -2048])
def test_exact_dc_predicate_rejects_fraction_above_limit(value):
    block = [value]*64
    block[-1] += 1 if value > 0 else -1
    result = safety.check_audio_safe(np.array(block, dtype=np.int16), value)
    assert result.a1_peak_ok and result.a2_slew_ok
    assert not result.a3_dc_ok


@pytest.mark.filterwarnings('error')
def test_predicate_widens_numpy_and_checks_previous_sample():
    result = safety.check_audio_safe(np.array([-32768,32767], dtype=np.int16), 32767)
    assert result.peak == 32768 and not result.a1_peak_ok
    assert result.max_slew == 65535 and not result.a2_slew_ok
    assert safety.check_audio_safe(np.array([], dtype=np.int16)).is_safe


def test_ola_uses_last_emitted_sample_and_preserves_constant_unity():
    engine = safety.SafetyEngine()
    engine.master_ola_L, engine.master_ola_R = list(range(8)), [1000]*8
    left, right = engine.apply_master_ola([7]*64, [1000]*64)
    assert left == [7]*64
    assert right == [1000]*64


def test_documented_stage_order(monkeypatch):
    monkeypatch.setattr(safety, '_tail_jit_active', lambda: False)
    engine = safety.SafetyEngine()
    observed = []
    for name in ('remove_dc', 'apply_limiter', 'apply_slew_limit', 'apply_master_ola'):
        original = getattr(engine, name)
        def spy(*args, _name=name, _original=original, **kwargs):
            observed.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(engine, name, spy)
    engine.step_block([32767]*64, [-32768]*64, ParamFrame())
    assert observed == ['remove_dc', 'apply_limiter', 'apply_slew_limit', 'apply_master_ola']


def test_real_synth_output_after_material_and_gain_is_safe(tmp_path, execution_path):
    synth = SynthMaster(SynthConfig(output_dir=str(tmp_path)))
    synth.configure_safety(SafetyConfig(dc_removal=True, master_ola=True))
    synth.disable('cockpit')
    previous = (0,0)
    heard = False
    for block in range(48):
        packet = VoiceFramePacketV1(voice_id=0, frame_seq=block+1, flags=VoiceFlags.GATE,
                                    gain_q15=32767, carrier_role_u8=1)
        params = ParamFrame(K_mode=3, drive_q15=32767, bias_q15=16000 if block % 2 else -16000,
                            gate_q15=32767 if block % 3 else 16384)
        left, right, _ = synth.render_frames([packet.to_voice_frame()], params=params)
        for values, p in zip((left,right), previous):
            assert legal(values, p, anchor=block > 0)
            heard |= any(values)
        previous = (left[-1],right[-1])
    assert heard


@pytest.mark.parametrize('left,right', [([0]*63,[0]*64), ([0]*64,[0]*65),
                                       ([32768]*64,[0]*64), ([0.5]*64,[0]*64)])
def test_bad_input_is_rejected_atomically(execution_path, left, right):
    engine = safety.SafetyEngine()
    with pytest.raises((ValueError, TypeError)):
        engine.step_block(left, right, ParamFrame())
    assert engine.rg.E_hat == engine.rg.dc_hat == 0
    assert engine.master_ola_L == engine.master_ola_R == []
    assert engine.prev_sample_L == engine.prev_sample_R == 0
