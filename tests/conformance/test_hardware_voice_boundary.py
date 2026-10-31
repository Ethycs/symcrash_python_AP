"""Establish the coefficient lifetime before assigning hardware state storage.

At a successful service boundary the final vector also owns next block's slew
history. This is a candidate storage sharing invariant, not a full page codec.
The current Python VM is K4; the tail alone is also exercised at K2 and K8.
"""
import numpy as np
import pytest

from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.coefficient_tail import apply_coefficient_tail
from symcrash_python_ap.vm.decode import DecodedPlan
from symcrash_python_ap.vm.scheduling import ExecutionPolicy
from symcrash_python_ap.vm.tables import VM_K
from tests.conformance.test_vm_snapshot_replay import prepared, assert_state_equal, execution_path


KINDS = ('a', 'b', 'pwm', 'fm6', 'morph', 'sample', 'sample_coeff', 'wavetable', 'granular')


@pytest.mark.parametrize('kind', KINDS)
def test_boundary_history_can_be_reconstructed_without_changing_future_audio(kind, execution_path):
    state, frame = prepared(kind)
    assert VM_K == 4, 'Extend this boundary qualification when the VM layout changes'
    np.testing.assert_array_equal(state.coeff_state, state.tail_previous_coeffs)
    restored = VMState.restore(state.snapshot())
    # Reconstruct a detached history from the single committed vector. Both
    # arrays are still required while processing, so aliasing them is forbidden.
    restored.tail_previous_coeffs = restored.coeff_state.copy()
    assert not np.shares_memory(restored.coeff_state, restored.tail_previous_coeffs)
    nonzero = False
    for block in range(12):
        if block == 3:
            frame.note, frame.f0_hz = 74, 587.33
        if block == 7:
            state.trigger_release()
            restored.trigger_release()
        policy = ExecutionPolicy(serviced=block % 4 != 0)
        before = state.coeff_state.copy()
        expected = vm_step_block(state, frame, policy=policy)[1].copy()
        actual = vm_step_block(restored, frame, policy=policy)[1].copy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(state.coeff_state, state.tail_previous_coeffs)
        np.testing.assert_array_equal(restored.coeff_state, restored.tail_previous_coeffs)
        if not policy.serviced:
            np.testing.assert_array_equal(state.coeff_state, before)
        nonzero |= bool(np.any(expected))
    assert nonzero
    assert_state_equal(state, restored)


@pytest.mark.parametrize('k', [2, 4, 8])
@pytest.mark.parametrize('mode', [0, 1, 2, 3])
def test_tail_commit_preserves_every_int16_slot_at_each_hardware_k(k, mode):
    state = VMState.create(1)
    rng = np.random.default_rng(48000 + k)
    state.coeff_state = rng.integers(-32768, 32768, 64 * k, dtype=np.int16)
    state.tail_previous_coeffs = rng.integers(-32768, 32768, 64 * k, dtype=np.int16)
    cfg = DecodedPlan()
    cfg.tail_mode = mode
    apply_coefficient_tail(state, cfg)
    assert state.coeff_state.shape == (64 * k,)
    assert state.coeff_state.dtype == np.int16
    np.testing.assert_array_equal(state.tail_previous_coeffs, state.coeff_state)
    assert not np.shares_memory(state.coeff_state, state.tail_previous_coeffs)


def test_arbitrary_restored_state_does_not_imply_the_commit_invariant():
    # Exporters must check the invariant at their boundary: a general Python
    # checkpoint is permitted to contain independently authored history.
    state = VMState.create(1)
    state.coeff_state.fill(123)
    state.tail_previous_coeffs.fill(-456)
    restored = VMState.restore(state.snapshot())
    np.testing.assert_array_equal(restored.coeff_state, 123)
    np.testing.assert_array_equal(restored.tail_previous_coeffs, -456)
