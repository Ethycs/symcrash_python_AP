"""
Hot-Path JIT Equivalence Tests (Packet E3)
==========================================

Verifies that the numba hot-path kernels (vm/hotpath_jit.py) are BIT-EXACT
with the pure-Python reference implementations in vm/executor.py and
vm/lanes.py.

Strategy: run the same inputs through both paths (toggling
hotpath_jit.set_jit_enabled) and require identical outputs AND identical
final state. Randomized inputs use a fixed seed for determinism.
"""

import numpy as np
import pytest

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.vm.state import VMState as VMStateCls
from symcrash_python_ap.vm.decode import BackboneEntry
from symcrash_python_ap.vm.lanes import dispatch_coeff_op
from symcrash_python_ap.frames import VoiceFrame

pytestmark = pytest.mark.skipif(
    not hot.NUMBA_AVAILABLE, reason="numba not available"
)

# Operators covered by the JIT dispatcher (op_id -> name, for test ids)
_JIT_OPS = {
    1: "Tilt",
    3: "IntraMix",
    4: "DiffuseUp",
    5: "DiffuseDown",
    6: "BandShear",
    7: "LowRankMorph",
    9: "SwapBands",
    10: "RotateNodes",
    12: "GluePair",
    15: "PressureRamp",
    16: "GateMask",
    17: "QuasiRotate",
    20: "FoldPressure",
    22: "SwallowtailMask",
}

_MASKS = [0xFFFFFFFF, 0x0000FFFF, 0xAAAAAAAA, 0x00FF00FF, 0x00000001]
_STRENGTHS = [0, 1, 1024, 16384, 20000, 32767]


@pytest.fixture(autouse=True)
def _restore_jit_flag():
    """Always leave the global JIT flag as we found it."""
    before = hot.jit_enabled()
    yield
    hot.set_jit_enabled(before)


def _random_coeffs(rng, scale=32767):
    return rng.integers(-scale, scale + 1, size=256).astype(np.int16)


@pytest.mark.parametrize(
    "op_id", sorted(_JIT_OPS), ids=[_JIT_OPS[k] for k in sorted(_JIT_OPS)]
)
def test_operator_kernels_bit_exact(op_id):
    """Every JIT operator kernel matches the reference on random states."""
    rng = np.random.default_rng(0xE3 + op_id)
    for mask in _MASKS:
        for strength in _STRENGTHS:
            coeffs = _random_coeffs(rng)
            instr = BackboneEntry(op_id=op_id, strength_q15=strength)

            state_ref = VMStateCls()
            state_ref.coeff_state = coeffs.copy()
            state_jit = VMStateCls()
            state_jit.coeff_state = coeffs.copy()

            hot.set_jit_enabled(False)
            dispatch_coeff_op(state_ref, instr, mask)
            hot.set_jit_enabled(True)
            dispatch_coeff_op(state_jit, instr, mask)

            assert np.array_equal(state_ref.coeff_state, state_jit.coeff_state), (
                f"op {op_id} ({_JIT_OPS[op_id]}) diverged "
                f"(mask=0x{mask:08X}, strength={strength})"
            )


def test_tail_ccs_soft_limit_bit_exact():
    """JIT tail limiter matches reference, including the rms trigger edge."""
    from symcrash_python_ap.vm.lanes import _tail_ccs_soft_limit

    rng = np.random.default_rng(0xE3)
    for scale in (1000, 20000, 23170, 24000, 32767):
        coeffs = _random_coeffs(rng, scale)

        state_ref = VMStateCls()
        state_ref.coeff_state = coeffs.copy()
        _tail_ccs_soft_limit(state_ref)

        jit_coeffs = coeffs.copy()
        hot.tail_ccs_soft_limit(jit_coeffs)

        assert np.array_equal(state_ref.coeff_state, jit_coeffs), (
            f"tail limiter diverged at scale {scale}"
        )


def test_idwpt_polyphase_bit_exact():
    """JIT polyphase IDWPT matches idwpt_polyphase_block."""
    from symcrash_python_ap.vm.tables import COEFF_LEN, VM_K
    from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block

    rng = np.random.default_rng(0xE3)
    for _ in range(20):
        coeffs = rng.integers(-32768, 32768, size=COEFF_LEN).astype(np.int16)

        coeff_2d = coeffs[:COEFF_LEN].reshape(64, VM_K).astype(np.int16)
        ref = idwpt_polyphase_block(coeff_2d, headroom_bits=2)

        out = np.zeros(64, dtype=np.int16)
        hot.idwpt_polyphase(coeffs, out, 2)

        assert np.array_equal(ref, out)


def test_fwpt_64_bit_exact():
    """JIT array-native FWPT matches fwpt_jit."""
    from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit

    rng = np.random.default_rng(0xE3)
    for _ in range(20):
        audio = rng.integers(-32768, 32768, size=64).astype(np.int16)
        ref = fwpt_jit(audio.tolist(), headroom_bits=1)
        out = np.empty(64, dtype=np.int64)
        hot.fwpt_64(audio, 1, out)
        assert np.array_equal(np.asarray(ref, dtype=np.int64), out)


def test_sincos_kernel_bit_exact():
    """JIT quarter-wave sin/cos matches sine_core.sin_cos_q15."""
    from symcrash_python_ap.wavelet.sine_core import sin_cos_q15

    rng = np.random.default_rng(0xE3)
    phases = rng.integers(0, 1 << 32, size=(64, 4), dtype=np.int64)
    # Include quadrant boundaries and LUT edge cases
    edge = [0, 0x3FFFFFFF, 0x40000000, 0x40000001, 0x7FFFFFFF,
            0x80000000, 0xBFFFFFFF, 0xC0000000, 0xFFFFFFFF]
    phases[0, :4] = edge[:4]
    phases[1, :4] = edge[4:8]
    phases[2, 0] = edge[8]

    sin_out = np.zeros(phases.shape, dtype=np.int32)
    cos_out = np.zeros(phases.shape, dtype=np.int32)
    hot.sincos_hk(hot._get_sine_lut_np(), phases, sin_out, cos_out)

    for h in range(phases.shape[0]):
        for k in range(phases.shape[1]):
            s_ref, c_ref = sin_cos_q15(int(phases[h, k]))
            assert sin_out[h, k] == s_ref, f"sin mismatch at phase {phases[h, k]:#x}"
            assert cos_out[h, k] == c_ref, f"cos mismatch at phase {phases[h, k]:#x}"


def _run_blocks(frame_fn, num_blocks, jit, note=69):
    hot.set_jit_enabled(jit)
    state = VMState.create(7)
    state.reset_for_note_on(note=note, cents=0)
    outputs = []
    for i in range(num_blocks):
        state, out = vm_step_block(state, frame_fn(i))
        outputs.append(out.copy())
    return outputs, state.coeff_state.copy()


def _assert_paths_match(frame_fn, num_blocks=32, note=69):
    ref_out, ref_cs = _run_blocks(frame_fn, num_blocks, jit=False, note=note)
    jit_out, jit_cs = _run_blocks(frame_fn, num_blocks, jit=True, note=note)
    for i, (a, b) in enumerate(zip(ref_out, jit_out)):
        assert np.array_equal(a, b), f"block {i} audio diverged"
    assert np.array_equal(ref_cs, jit_cs), "final coeff_state diverged"


def test_full_block_mode_a():
    def frame(i):
        return VoiceFrame(
            voice_id=7, note=60, f0_hz=261.63,
            gain_q15=min(32767, 4096 * (i + 1)),
            beta_target_q15=16384,
            partial_amps_q15=[32767, 16384, 8192, 4096],
            ops_enable=['tilt', 'intramix'],
            families={'tilt': 0.7, 'intramix': 0.5},
            is_active=True,
        )
    _assert_paths_match(frame, note=60)


def test_full_block_mode_b_partials():
    def frame(i):
        return VoiceFrame(
            voice_id=7, note=69, f0_hz=440.0,
            coeff_osc_mode=1,
            gain_q15=24576, beta_target_q15=16384,
            partial_amps_q15=[32767, 16384, 10923, 8192, 6554, 5461],
            ops_enable=['tilt', 'diffusedown', 'gluepair'],
            families={'tilt': 0.4},
            is_active=True,
        )
    _assert_paths_match(frame)


def test_full_block_mode_b_pwm():
    def frame(i):
        return VoiceFrame(
            voice_id=7, note=48, f0_hz=130.81,
            coeff_osc_mode=1,
            waveform_type=2,  # PULSE
            pwm_enabled=True,
            pwm_width_q15=8192 + (i * 512) % 16384,
            H_request=32,
            gain_q15=24576, beta_target_q15=16384,
            is_active=True,
        )
    _assert_paths_match(frame, note=48)


@pytest.mark.parametrize("alg_id", [1, 5, 16, 22, 32])
def test_full_block_fm6(alg_id):
    def frame(i):
        return VoiceFrame(
            voice_id=7, note=57, f0_hz=220.0,
            coeff_osc_mode=3,
            fm_algorithm_id=alg_id,
            fm_op_ratios=[65536, 131072, 65536, 98304, 65536, 131072],
            fm_op_levels=[32767, 16384, 24576, 8192, 12288, 16384],
            fm_feedback_depth=5,
            gain_q15=24576, beta_target_q15=16384,
            is_active=True,
        )
    _assert_paths_match(frame, num_blocks=24, note=57)


def test_full_block_cmmm_events():
    """CMMM mask / tail-mode events drive identical behavior on both paths."""
    def frame(i):
        f = VoiceFrame(
            voice_id=7, note=64, f0_hz=329.63,
            coeff_osc_mode=1,
            gain_q15=24576, beta_target_q15=16384,
            partial_amps_q15=[32767, 12000, 8000],
            ops_enable=['tilt', 'intramix', 'quasirotate', 'lowrankmorph'],
            families={'tilt': 0.6, 'intramix': 0.9},
            is_active=True,
        )
        if i == 8:
            f.cmmm_node_mask = 0x0000FFFF
        if i == 16:
            f.cmmm_tail_mode = 2
            f.cmmm_traction_q15 = 16384
        if i == 24:
            f.cmmm_node_mask = 0xFFFFFFFF
        return f
    _assert_paths_match(frame, num_blocks=40, note=64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
