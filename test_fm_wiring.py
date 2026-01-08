"""Quick test for FM wiring - Phase 3."""
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState, OSC_MODE_B
from symcrash_python_ap.vm.decode import FM_SOURCE_FEEDBACK
from symcrash_python_ap.frames import VoiceFrame
import numpy as np

def test_fm_wiring():
    """Test FM modulator feedback wiring (Phase 3 - WSE-STD-210 Section 8.3)."""
    # Create voice state for Mode B
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    # Test 1: Basic Mode B without FM (should work)
    frame_no_fm = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B (0=Mode A, 1=Mode B)
        H_request=8,
        fm_depth_q15=0,
        fm_source=0,
    )

    state, output = vm_step_block(state, frame_no_fm)
    rms1 = np.sqrt(np.mean(output.astype(float)**2))
    assert rms1 > 0, "Mode B should produce output"

    # Test 2: Mode B with FM feedback (should produce different output)
    state2 = VMState.create(voice_id=1)
    state2.reset_for_note_on(note=60)
    frame_with_fm = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B (0=Mode A, 1=Mode B)
        H_request=8,
        fm_depth_q15=16384,  # 50% depth
        fm_source=1,  # FM_SOURCE_FEEDBACK
    )
    # Run a few blocks to build up feedback
    for i in range(5):
        state2, output2 = vm_step_block(state2, frame_with_fm)
    rms2 = np.sqrt(np.mean(output2.astype(float)**2))
    assert rms2 > 0, "Mode B with FM should produce output"
    assert np.any(state2.prev_fm_mod_buffer != 0), "FM buffer should be populated"

    # Test 3: Verify FM changes output after several blocks
    state3 = VMState.create(voice_id=2)
    state3.reset_for_note_on(note=60)
    outputs_no_fm = []
    for i in range(10):
        frame = VoiceFrame(note=60, coeff_osc_mode=1, H_request=8, fm_depth_q15=0, fm_source=0)
        state3, out = vm_step_block(state3, frame)
        outputs_no_fm.append(out.copy())

    state4 = VMState.create(voice_id=3)
    state4.reset_for_note_on(note=60)
    outputs_with_fm = []
    for i in range(10):
        frame = VoiceFrame(note=60, coeff_osc_mode=1, H_request=8, fm_depth_q15=16384, fm_source=1)
        state4, out = vm_step_block(state4, frame)
        outputs_with_fm.append(out.copy())

    # After a few blocks, outputs should differ
    diff = np.max(np.abs(outputs_no_fm[9].astype(float) - outputs_with_fm[9].astype(float)))
    assert diff > 0, f"FM should change output, but diff={diff}"

if __name__ == "__main__":
    test_fm_wiring()
