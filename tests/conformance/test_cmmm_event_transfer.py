"""Independent event semantics and the CMMM -> VoiceFrame -> rendered audio path."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from symcrash_python_ap.frames import CMMMEventPayload, VoiceFrame
from symcrash_python_ap.operator_library import BAND_SLICES
from symcrash_python_ap.voice import VoiceEngine
from symcrash_python_ap.voice.voice_engine import (
    ARC_SPECS, ArcSpec, GestureBasis, Regime, VoiceFrameABI,
)
from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.vm.cmmm_events import apply_cmmm_events
from symcrash_python_ap.vm.decode import BackboneEntry, decode_frame
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.lanes import dispatch_coeff_op, lane_inject
from symcrash_python_ap.vm.state import VMState


@pytest.fixture(autouse=True)
def restore_jit():
    before = hot.jit_enabled()
    yield
    hot.set_jit_enabled(before)


def state(value=10000):
    result = VMState.create()
    result.coeff_state[:] = value
    return result


@pytest.mark.parametrize('jit', [False, True])
@pytest.mark.parametrize('op_id,ratios', [
    (15, [Fraction(-3, 100), Fraction(-3, 100), Fraction(1, 10), Fraction(1, 10), Fraction(-3, 100), Fraction(-3, 100)]),
    (20, [Fraction(-6, 100), Fraction(15, 100), Fraction(30, 100), Fraction(15, 100), Fraction(-6, 100), Fraction(-6, 100)]),
])
@pytest.mark.parametrize('strength', [0, 1, 16384, 32767])
def test_pressure_is_bounded_diagonal_scaling_with_zero_identity(jit, op_id, ratios, strength):
    hot.set_jit_enabled(jit)
    original = np.resize(np.array([-32768, -20000, -10, 0, 10, 10000, 20000, 32767], dtype=np.int16), 256)
    s = state()
    s.coeff_state = original.copy()
    expected = original.astype(np.int64)
    mask = 0xF0FF55FF
    for (start, end), ratio in zip(BAND_SLICES, ratios):
        gain_delta = round(Fraction(round(ratio * 32768) * strength, 32768))
        for i in range(start, end):
            if mask & (1 << (i // 8)):
                expected[i] = max(-32768, min(32767, int(original[i]) + round(Fraction(int(original[i]) * gain_delta, 32768))))
    dispatch_coeff_op(s, BackboneEntry(op_id, strength), mask)
    assert np.array_equal(s.coeff_state, expected)
    if strength == 0:
        assert np.array_equal(s.coeff_state, original)


def test_pressure_operators_change_quiet_signals_in_distinct_authored_bands():
    ramp, fold = state(10000), state(10000)
    dispatch_coeff_op(ramp, BackboneEntry(15), 0xFFFFFFFF)
    dispatch_coeff_op(fold, BackboneEntry(20), 0xFFFFFFFF)
    assert 10999 <= ramp.coeff_state[24] <= 11001
    assert 12999 <= fold.coeff_state[24] <= 13001
    assert ramp.coeff_state[8] < 10000 < fold.coeff_state[8]
    assert ramp.coeff_state[168] < 10000 and fold.coeff_state[168] < 10000


def test_abi_transfers_semantics_without_using_wire_lane_numbers():
    engine = VoiceEngine()
    engine.allocate_voice()
    abi = VoiceFrameABI(event_mask=255, event_strengths=[100, 200, 300, 0x15, 400, 0x3C, 500, 600],
                        impulse_band=2, noise_band=4)
    frame = engine._abi_to_voice_frame(abi, 0)
    events = frame.cmmm_events
    assert events == CMMMEventPayload(2, 100, 4, 200, 300, 0x15, 400, 0x3C, 500, 600, 255)
    plan = decode_frame(frame)
    assert plan.cmmm_events == events and not plan.event_instrs
    # Conversion copies its payload instead of retaining mutable ABI storage.
    abi.event_strengths[0] = 999
    assert events.impulse_strength_q15 == 100


@pytest.mark.parametrize('band', range(6))
def test_impulse_targets_selected_band_and_respects_node_mask(band):
    s = state(0)
    payload = CMMMEventPayload(active_mask=1, impulse_band=band, impulse_strength_q15=10000)
    start = BAND_SLICES[band][0]
    apply_cmmm_events(s, payload, 0xFFFFFFFF ^ (1 << (start // 8)))
    assert not np.any(s.coeff_state)
    before_rng = s.prng_state
    apply_cmmm_events(s, payload, 0xFFFFFFFF)
    expected = np.zeros(256, dtype=np.int16)
    expected[start] = 10000
    assert np.array_equal(s.coeff_state, expected)
    assert s.prng_state == before_rng


def test_noise_targets_band_and_consumes_exactly_one_draw_per_active_coefficient():
    s = state(0)
    seed = s.prng_state
    payload = CMMMEventPayload(active_mask=2, noise_band=5, noise_amount_q15=16384)
    apply_cmmm_events(s, payload, (1 << 21) | (1 << 0))
    expected = np.zeros(256, dtype=np.int16)
    for i in range(168, 176):
        seed ^= (seed << 13) & 0xFFFFFFFF
        seed ^= seed >> 17
        seed ^= (seed << 5) & 0xFFFFFFFF
        expected[i] = round(Fraction(((seed & 0xFFFF) - 32768) * 16384, 32768))
    assert s.prng_state == seed
    assert np.array_equal(s.coeff_state, expected)


@pytest.mark.parametrize('active,amount,mask', [(0, 30000, 0xFFFFFFFF), (2, 0, 0xFFFFFFFF), (2, 1000, 1)])
def test_inactive_zero_or_fully_masked_noise_does_not_consume_rng(active, amount, mask):
    s = state()
    before, seed = s.coeff_state.copy(), s.prng_state
    apply_cmmm_events(s, CMMMEventPayload(active_mask=active, noise_band=5, noise_amount_q15=amount), mask)
    assert np.array_equal(s.coeff_state, before) and s.prng_state == seed


def test_gate_bitmap_is_not_an_amplitude_and_inactive_nodes_are_untouched():
    s = state(12345)
    mask = 0xFFFFFFFF ^ (1 << 1)
    apply_cmmm_events(s, CMMMEventPayload(active_mask=8, gate_mask=0x15), mask)
    expected = np.full(256, 12345, dtype=np.int16)
    for b, (start, end) in enumerate(BAND_SLICES):
        if not 0x15 & (1 << b):
            expected[start:end] = 0
    expected[8:16] = 12345
    assert np.array_equal(s.coeff_state, expected)


def test_swallowtail_bitmap_limits_operator_to_requested_bands():
    s = state(12345)
    apply_cmmm_events(s, CMMMEventPayload(active_mask=32, swallowtail_bands=1 << 3), 0xFFFFFFFF)
    expected = np.full(256, 12345, dtype=np.int16)
    for i in range(56, 104):
        if ((i // 8) * 3) % 5 >= 3:
            expected[i] = 0
    assert np.array_equal(s.coeff_state, expected)


def test_cusp_widens_before_addition_and_saturates_both_signs():
    s = state()
    s.coeff_state[0], s.coeff_state[8] = 30000, -30000
    with np.errstate(over='raise'):
        apply_cmmm_events(s, CMMMEventPayload(active_mask=16, cusp_amount_q15=32767), 3)
    assert s.coeff_state[0] == 32767 and s.coeff_state[8] == -32768


def test_sheaf_threshold_projects_only_incompatible_active_boundaries_with_rne():
    s = state(100)
    s.coeff_state[7:9] = [-101, 100]  # Average -0.5 rounds to even zero.
    s.coeff_state[15:17] = [-20000, 20000]  # Node 2 inactive: untouched.
    apply_cmmm_events(s, CMMMEventPayload(active_mask=128, sheaf_threshold_q15=200), 3)
    assert list(s.coeff_state[7:9]) == [0, 0]
    assert list(s.coeff_state[15:17]) == [-20000, 20000]


@pytest.mark.parametrize('lane,kwargs', [
    (0, dict(impulse_strength_q15=10000)), (1, dict(noise_amount_q15=10000)),
    (2, dict(pulse_pressure_q15=32767)), (3, dict(gate_mask=0)),
    (4, dict(cusp_amount_q15=32767)), (5, dict(swallowtail_bands=0x3F)),
    (6, dict(fold_pressure_q15=32767)), (7, dict(sheaf_threshold_q15=100)),
])
def test_each_semantic_lane_reaches_coefficient_execution(lane, kwargs):
    s = state()
    s.coeff_state[7:9] = [-20000, 20000]
    original, seed = s.coeff_state.copy(), s.prng_state
    frame = VoiceFrame(cmmm_events=CMMMEventPayload(active_mask=1 << lane, **kwargs))
    lane_inject(s, decode_frame(frame))
    assert not np.array_equal(s.coeff_state, original)
    assert (s.prng_state != seed) == (lane == 1)


def test_catastrophe_emits_all_four_documented_corridor_events():
    e = VoiceEngine()
    e.allocate_voice()
    s = e.voices[0]
    s.regime = Regime.R5_CATASTROPHE
    s.arc.active = False
    mask, values = e._compute_events_with_regime(s)
    assert mask == 0xF0 and values[5] == 0x3C and values[7] > 0


def test_arc_masks_do_not_become_q15_strengths(monkeypatch):
    monkeypatch.setitem(ARC_SPECS, 99, ArcSpec(arc_id=99, arc_events=0x28))
    e = VoiceEngine()
    e.allocate_voice()
    s = e.voices[0]
    s.regime = Regime.R2_STABLE
    s.arc.active, s.arc.arc_id = True, 99
    mask, values = e._compute_events_with_regime(s)
    assert mask == 0x28 and values[3] == 0xFF and values[5] == 0x3C


def test_actual_cmmm_events_change_audio_and_replay_identically():
    e = VoiceEngine(max_voices=1)
    e.allocate_voice()
    e.note_on(0, GestureBasis(gate=True, pressure_q15=32767, speed_q15=32767))
    frames = [e.step_block().render_batch[0].frame for _ in range(20)]
    assert any(f.cmmm_events.active_mask for f in frames)
    def render(with_events, jit):
        hot.set_jit_enabled(jit)
        s = VMState.create()
        samples = []
        for original in frames:
            frame = deepcopy(original)
            if not with_events:
                frame.cmmm_events = None
            s, output = vm_step_block(s, frame)
            samples.extend(output.tolist())
        return samples, s.snapshot()
    reference = render(True, False)
    assert reference == render(True, True)
    assert reference[0] != render(False, False)[0]
    assert any(reference[0])


def test_latched_node_mask_restricts_events_after_the_command_block():
    s = VMState.create()
    payload = CMMMEventPayload(active_mask=2, noise_band=5, noise_amount_q15=10000)
    seed = s.prng_state
    vm_step_block(s, VoiceFrame(cmmm_node_mask=1, cmmm_events=payload))
    vm_step_block(s, VoiceFrame(cmmm_events=payload))
    assert s.prng_state == seed  # Band 5 has no active nodes in either block.
