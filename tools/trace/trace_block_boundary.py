"""Trace block boundaries to find where discontinuities are introduced."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.vm import vm_step_block, VMState, DecodedPlan, Tables

Q15_ONE = 32767
BLOCK_SIZE = 64
SAMPLE_RATE = 48000

def check_boundary_discontinuity(samples, block_size):
    """Check for discontinuities at block boundaries."""
    samples = np.array(samples, dtype=np.int32)
    issues = []

    for b in range(1, len(samples) // block_size):
        boundary = b * block_size
        if boundary >= len(samples):
            break

        # Get samples around boundary
        prev = samples[boundary - 1]
        curr = samples[boundary]

        # Also get first derivatives
        if boundary >= 2 and boundary < len(samples) - 1:
            d1_before = samples[boundary - 1] - samples[boundary - 2]
            d1_after = samples[boundary + 1] - samples[boundary]
            d1_at = curr - prev

            # Check for sudden jump in derivative
            d1_jump = abs(d1_at - d1_before)
            if d1_jump > 500:  # Threshold for detecting discontinuity
                issues.append({
                    'block': b,
                    'sample': boundary,
                    'prev': prev,
                    'curr': curr,
                    'jump': curr - prev,
                    'd1_before': d1_before,
                    'd1_at': d1_at,
                    'd1_after': d1_after,
                    'd1_jump': d1_jump
                })

    return issues


# Create a pure sine VoiceFrame
frame = VoiceFrame(
    voice_id=0,
    note=60,
    cents=0,
    f0_hz=261.63,
    gain_q15=24576,
    pan_q15=0,
    tree_a_id=0,
    tree_b_id=0,
    tree_blend_q15=0,
    is_active=True,
    beta_target_q15=0,
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
)

print("=" * 70)
print("STAGE 1: Pure VM output (no safety, no pan)")
print("=" * 70)

# Direct VM output
tables = Tables()
vm_state = VMState.create(0)
vm_state.reset_for_note_on(f0_hz=261.63)

vm_blocks = []
for i in range(20):
    vm_state, output = vm_step_block(vm_state, frame, tables)
    vm_blocks.extend(output.tolist())

issues = check_boundary_discontinuity(vm_blocks, BLOCK_SIZE)
if issues:
    print(f"Found {len(issues)} boundary discontinuities in VM output")
    for issue in issues[:5]:
        print(f"  Block {issue['block']}: d1_jump={issue['d1_jump']}")
else:
    print("No boundary discontinuities in VM output - VM is clean!")

print("\n" + "=" * 70)
print("STAGE 2: Full pipeline (VM + pan + safety)")
print("=" * 70)

stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
stage.voice_on(0, frame.note, frame.cents)

pipeline_L = []
pipeline_R = []
for i in range(20):
    L, R = stage.render_block(frame)
    pipeline_L.extend(L)
    pipeline_R.extend(R)

issues_L = check_boundary_discontinuity(pipeline_L, BLOCK_SIZE)
if issues_L:
    print(f"Found {len(issues_L)} boundary discontinuities in L channel")
    for issue in issues_L[:5]:
        print(f"  Block {issue['block']}: d1_jump={issue['d1_jump']}, jump={issue['jump']}")
else:
    print("No boundary discontinuities in L channel - pipeline is clean!")

# Try to isolate: check if it's pan or safety causing it
print("\n" + "=" * 70)
print("STAGE 3: Isolating - VM + pan only (no safety)")
print("=" * 70)

from symcrash_python_ap.wavelet.sine_core import sine_q15, cosine_q15, PHASE_MASK
from symcrash_python_ap.fixed import mul_q15_sat

# Reset VM state
vm_state = VMState.create(0)
vm_state.reset_for_note_on(f0_hz=261.63)

Q15_SCALE = 32768
pan_q15 = 0
pan_norm = (pan_q15 + Q15_SCALE) / (2 * Q15_SCALE)
pan_phase = int(pan_norm * 0x40000000) & PHASE_MASK
left_q15 = cosine_q15(pan_phase)
right_q15 = sine_q15(pan_phase)

pan_only_L = []
for i in range(20):
    vm_state, output = vm_step_block(vm_state, frame, tables)
    L = [mul_q15_sat(s, left_q15) for s in output.tolist()]
    pan_only_L.extend(L)

issues_pan = check_boundary_discontinuity(pan_only_L, BLOCK_SIZE)
if issues_pan:
    print(f"Found {len(issues_pan)} boundary discontinuities (pan only)")
    for issue in issues_pan[:5]:
        print(f"  Block {issue['block']}: d1_jump={issue['d1_jump']}")
else:
    print("No boundary discontinuities with pan only - safety is the culprit!")

print("\n" + "=" * 70)
print("STAGE 4: Safety engine step-by-step")
print("=" * 70)

from symcrash_python_ap.safety.safety_engine import SafetyEngine
from symcrash_python_ap.frames import ParamFrame

# Fresh VM state and safety engine
vm_state = VMState.create(0)
vm_state.reset_for_note_on(f0_hz=261.63)

safety = SafetyEngine(block_size=BLOCK_SIZE)
safety.voice_on(0)
params = ParamFrame()

# Test without OLA
safety_no_ola_L = []
for i in range(20):
    vm_state, output = vm_step_block(vm_state, frame, tables)
    L = [mul_q15_sat(s, left_q15) for s in output.tolist()]
    R = [mul_q15_sat(s, right_q15) for s in output.tolist()]
    L_out, R_out = safety.step_block(L, R, params, apply_master_ola=False, apply_dc_removal=False)
    safety_no_ola_L.extend(L_out)

issues_no_ola = check_boundary_discontinuity(safety_no_ola_L, BLOCK_SIZE)
print(f"Safety WITHOUT OLA: {len(issues_no_ola)} boundary discontinuities")

# Reset for OLA test
vm_state = VMState.create(0)
vm_state.reset_for_note_on(f0_hz=261.63)

safety2 = SafetyEngine(block_size=BLOCK_SIZE)
safety2.voice_on(0)

safety_with_ola_L = []
for i in range(20):
    vm_state, output = vm_step_block(vm_state, frame, tables)
    L = [mul_q15_sat(s, left_q15) for s in output.tolist()]
    R = [mul_q15_sat(s, right_q15) for s in output.tolist()]
    L_out, R_out = safety2.step_block(L, R, params, apply_master_ola=True, apply_dc_removal=False)
    safety_with_ola_L.extend(L_out)

issues_with_ola = check_boundary_discontinuity(safety_with_ola_L, BLOCK_SIZE)
print(f"Safety WITH OLA: {len(issues_with_ola)} boundary discontinuities")

# Check just gain/limiter (bypass OLA entirely)
print("\n" + "=" * 70)
print("STAGE 5: Checking each safety component")
print("=" * 70)

# Check raw limiter effect
vm_state = VMState.create(0)
vm_state.reset_for_note_on(f0_hz=261.63)

safety3 = SafetyEngine(block_size=BLOCK_SIZE)
safety3.voice_on(0)

# Disable HF filter to isolate limiter
safety3.q_hf_cut = 32767  # Pass everything
safety3.q_hf_blend = 0   # No blending

raw_safety_L = []
for i in range(20):
    vm_state, output = vm_step_block(vm_state, frame, tables)
    L = [mul_q15_sat(s, left_q15) for s in output.tolist()]
    R = [mul_q15_sat(s, right_q15) for s in output.tolist()]
    L_out, R_out = safety3.step_block(L, R, params, apply_master_ola=False, apply_dc_removal=False)
    raw_safety_L.extend(L_out)

issues_raw = check_boundary_discontinuity(raw_safety_L, BLOCK_SIZE)
print(f"Safety (HF disabled, no OLA): {len(issues_raw)} boundary discontinuities")
