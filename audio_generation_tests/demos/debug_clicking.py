"""Debug clicking/discontinuities in Mode B synthesis."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState, OSC_MODE_B
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas

# Clear cached basis frames
clear_basis_atlas()

FS = 48000
BLOCK_SIZE = 64

def analyze_discontinuities(samples: np.ndarray, threshold: int = 1000) -> list:
    """Find sample-to-sample jumps exceeding threshold."""
    diffs = np.abs(np.diff(samples.astype(np.int32)))
    discontinuities = []
    for i, d in enumerate(diffs):
        if d > threshold:
            discontinuities.append({
                'sample': i,
                'block': i // BLOCK_SIZE,
                'in_block_pos': i % BLOCK_SIZE,
                'jump': int(d),
                'val_before': int(samples[i]),
                'val_after': int(samples[i + 1]),
            })
    return discontinuities

print("=" * 70)
print("DEBUG: Mode B Clicking Analysis")
print("=" * 70)

# Test 1: Basic Mode B (no FM) - single sustained note
print("\n--- Test 1: Basic Mode B (C4, 1 second) ---")
state1 = VMState.create(voice_id=0)
state1.reset_for_note_on(note=60)

frame1 = VoiceFrame(note=60, coeff_osc_mode=1, H_request=8)

samples1 = []
for _ in range(750):  # 1 second
    state1, output = vm_step_block(state1, frame1)
    samples1.extend(output.tolist())

samples1 = np.array(samples1, dtype=np.int16)
discs1 = analyze_discontinuities(samples1, threshold=2000)
print(f"Total samples: {len(samples1)}")
print(f"Discontinuities > 2000: {len(discs1)}")
if discs1:
    print("First 5 discontinuities:")
    for d in discs1[:5]:
        print(f"  Sample {d['sample']} (block {d['block']}, pos {d['in_block_pos']}): "
              f"{d['val_before']} -> {d['val_after']} (jump={d['jump']})")

# Check block boundaries specifically
print("\nBlock boundary analysis (every 64 samples):")
for block in range(min(10, len(samples1) // BLOCK_SIZE)):
    boundary = block * BLOCK_SIZE - 1
    if boundary >= 0 and boundary < len(samples1) - 1:
        jump = abs(int(samples1[boundary + 1]) - int(samples1[boundary]))
        if jump > 500:
            print(f"  Block {block}: {samples1[boundary]} -> {samples1[boundary + 1]} (jump={jump})")


# Test 2: Mode B with FM feedback
print("\n--- Test 2: Mode B with FM Feedback (C4, 1 second) ---")
state2 = VMState.create(voice_id=0)
state2.reset_for_note_on(note=60)

frame2 = VoiceFrame(note=60, coeff_osc_mode=1, H_request=8, fm_depth_q15=16384, fm_source=1)

samples2 = []
for _ in range(750):
    state2, output = vm_step_block(state2, frame2)
    samples2.extend(output.tolist())

samples2 = np.array(samples2, dtype=np.int16)
discs2 = analyze_discontinuities(samples2, threshold=2000)
print(f"Total samples: {len(samples2)}")
print(f"Discontinuities > 2000: {len(discs2)}")
if discs2:
    print("First 5 discontinuities:")
    for d in discs2[:5]:
        print(f"  Sample {d['sample']} (block {d['block']}, pos {d['in_block_pos']}): "
              f"{d['val_before']} -> {d['val_after']} (jump={d['jump']})")

# Check block boundaries
print("\nBlock boundary analysis (every 64 samples):")
boundary_jumps = []
for block in range(len(samples2) // BLOCK_SIZE):
    boundary = block * BLOCK_SIZE - 1
    if boundary >= 0 and boundary < len(samples2) - 1:
        jump = abs(int(samples2[boundary + 1]) - int(samples2[boundary]))
        boundary_jumps.append(jump)
        if jump > 500:
            print(f"  Block {block}: {samples2[boundary]} -> {samples2[boundary + 1]} (jump={jump})")

print(f"\nMax boundary jump: {max(boundary_jumps) if boundary_jumps else 0}")
print(f"Avg boundary jump: {np.mean(boundary_jumps) if boundary_jumps else 0:.1f}")


# Test 3: Note transitions (melody)
print("\n--- Test 3: Note Transitions (C4 -> E4 -> G4) ---")
melody = [(60, 0.2), (64, 0.2), (67, 0.2)]
all_samples3 = []
transition_points = []

for note, duration in melody:
    # Create fresh state for each note (like the demo does)
    state3 = VMState.create(voice_id=0)
    state3.reset_for_note_on(note=note)

    frame3 = VoiceFrame(note=note, coeff_osc_mode=1, H_request=8)

    note_start = len(all_samples3)
    num_blocks = int(duration * 750)
    for _ in range(num_blocks):
        state3, output = vm_step_block(state3, frame3)
        all_samples3.extend(output.tolist())

    transition_points.append(note_start)

samples3 = np.array(all_samples3, dtype=np.int16)
discs3 = analyze_discontinuities(samples3, threshold=3000)
print(f"Total samples: {len(samples3)}")
print(f"Discontinuities > 3000: {len(discs3)}")

print("\nAnalyzing note transition boundaries:")
for i, tp in enumerate(transition_points[1:], 1):
    if tp > 0 and tp < len(samples3):
        jump = abs(int(samples3[tp]) - int(samples3[tp - 1]))
        print(f"  Transition {i} at sample {tp}: {samples3[tp-1]} -> {samples3[tp]} (jump={jump})")


# Test 4: Continuous state across notes (proper way)
print("\n--- Test 4: Continuous State Note Transitions ---")
state4 = VMState.create(voice_id=0)
all_samples4 = []
transition_points4 = []

for note, duration in melody:
    # Don't recreate state - just update the note
    state4.reset_for_note_on(note=note)

    frame4 = VoiceFrame(note=note, coeff_osc_mode=1, H_request=8)

    note_start = len(all_samples4)
    num_blocks = int(duration * 750)
    for _ in range(num_blocks):
        state4, output = vm_step_block(state4, frame4)
        all_samples4.extend(output.tolist())

    transition_points4.append(note_start)

samples4 = np.array(all_samples4, dtype=np.int16)
discs4 = analyze_discontinuities(samples4, threshold=3000)
print(f"Total samples: {len(samples4)}")
print(f"Discontinuities > 3000: {len(discs4)}")

print("\nAnalyzing note transition boundaries (continuous state):")
for i, tp in enumerate(transition_points4[1:], 1):
    if tp > 0 and tp < len(samples4):
        jump = abs(int(samples4[tp]) - int(samples4[tp - 1]))
        print(f"  Transition {i} at sample {tp}: {samples4[tp-1]} -> {samples4[tp]} (jump={jump})")


# Test 5: Legato mode - proper way (no phase reset on pitch change)
print("\n--- Test 5: Legato Mode Note Transitions ---")
state5 = VMState.create(voice_id=0)
state5.reset_for_note_on(note=melody[0][0])  # Only reset on first note
all_samples5 = []
transition_points5 = []

for note, duration in melody:
    # Use legato_mode=True for smooth pitch transitions
    frame5 = VoiceFrame(note=note, coeff_osc_mode=1, H_request=8, legato_mode=True)

    note_start = len(all_samples5)
    num_blocks = int(duration * 750)
    for _ in range(num_blocks):
        state5, output = vm_step_block(state5, frame5)
        all_samples5.extend(output.tolist())

    transition_points5.append(note_start)

samples5 = np.array(all_samples5, dtype=np.int16)
discs5 = analyze_discontinuities(samples5, threshold=3000)
print(f"Total samples: {len(samples5)}")
print(f"Discontinuities > 3000: {len(discs5)}")

print("\nAnalyzing note transition boundaries (legato mode):")
for i, tp in enumerate(transition_points5[1:], 1):
    if tp > 0 and tp < len(samples5):
        jump = abs(int(samples5[tp]) - int(samples5[tp - 1]))
        print(f"  Transition {i} at sample {tp}: {samples5[tp-1]} -> {samples5[tp]} (jump={jump})")


# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Test 1 (Basic Mode B):      {len(discs1)} discontinuities")
print(f"Test 2 (Mode B + FM):       {len(discs2)} discontinuities")
print(f"Test 3 (Fresh state/note):  {len(discs3)} discontinuities")
print(f"Test 4 (Continuous state):  {len(discs4)} discontinuities")
print(f"Test 5 (Legato mode):       {len(discs5)} discontinuities")
