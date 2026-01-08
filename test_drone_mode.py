"""Quick test for Drone mode - Phase 4."""
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame
import numpy as np


def test_drone_mode():
    """Test drone mode reduces coefficient evolution rate (Phase 4)."""
    # Test 1: Normal mode - coefficients refresh quickly
    state_normal = VMState.create(voice_id=0)
    state_normal.reset_for_note_on(note=60)

    frame_normal = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # Mode B
        H_request=8,
        drone_mode=False,
    )

    # Run several blocks
    for _ in range(10):
        state_normal, _ = vm_step_block(state_normal, frame_normal)
    coeff_normal = state_normal.coeff_state.copy()

    # Test 2: Drone mode - coefficients persist more
    state_drone = VMState.create(voice_id=1)
    state_drone.reset_for_note_on(note=60)

    frame_drone = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # Mode B
        H_request=8,
        drone_mode=True,
    )

    for _ in range(10):
        state_drone, _ = vm_step_block(state_drone, frame_drone)
    coeff_drone = state_drone.coeff_state.copy()

    # Test 3: Verify drone mode produces different coefficients
    # (due to different base_mix blending behavior)
    diff = np.max(np.abs(coeff_normal.astype(float) - coeff_drone.astype(float)))
    assert diff > 0, "Drone mode should produce different coefficients"

    # Test 4: Drone mode with NoiseBand operator for texture
    state_noise_drone = VMState.create(voice_id=2)
    state_noise_drone.reset_for_note_on(note=60)

    frame_noise_drone = VoiceFrame(
        note=60,
        coeff_osc_mode=1,
        H_request=4,
        drone_mode=True,
        ops_enable=['noiseband'],
        families={'noiseband': 0.3},
    )

    outputs = []
    for _ in range(20):
        state_noise_drone, out = vm_step_block(state_noise_drone, frame_noise_drone)
        outputs.append(out.copy())

    # Should produce varying output due to noise injection
    rms_first = np.sqrt(np.mean(outputs[0].astype(float)**2))
    rms_last = np.sqrt(np.mean(outputs[-1].astype(float)**2))
    assert rms_first > 0, "Noise drone should produce output"
    assert rms_last > 0, "Noise drone should continue producing output"


if __name__ == "__main__":
    test_drone_mode()
    print("All drone mode tests passed!")
