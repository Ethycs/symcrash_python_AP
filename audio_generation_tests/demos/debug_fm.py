"""Debug FM feedback synthesis."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame

# Test FM feedback
state = VMState.create(voice_id=0)
state.reset_for_note_on(note=60)

frame = VoiceFrame(
    note=60,
    coeff_osc_mode=1,  # MODE_B
    H_request=8,
    fm_depth_q15=16384,  # 50% FM depth
    fm_source=1,  # FM_SOURCE_FEEDBACK
)

print("=" * 60)
print("FM FEEDBACK DEBUG")
print("=" * 60)
print(f"fm_depth_q15 = {frame.fm_depth_q15}")
print(f"fm_source = {frame.fm_source}")
print()

# Run a few blocks and check the FM buffer
for i in range(10):
    # Check buffer before execution
    fm_buf_before = state.prev_fm_mod_buffer.copy()
    mb_state = state.mode_b_state

    state, output = vm_step_block(state, frame)

    fm_buf_after = state.prev_fm_mod_buffer

    print(f"Block {i}:")
    print(f"  fm_mod_buffer before: max={np.max(np.abs(fm_buf_before))}")
    print(f"  fm_mod_buffer after:  max={np.max(np.abs(fm_buf_after))}, values={fm_buf_after[:4]}")
    print(f"  mode_b fm_depth: {mb_state.fm_depth_q15}")
    print(f"  mode_b fm_mod_buffer: {mb_state.fm_mod_buffer is not None}")
    print(f"  output range: {np.min(output)} to {np.max(output)}")
    print()
