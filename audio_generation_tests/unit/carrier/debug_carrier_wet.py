"""Debug carrier vs wet output quality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame
import numpy as np

stage = UnifiedOutputStage()

# Render A4 (440 Hz) with pure carrier (beta=0)
note = 69
stage.voice_on(0, note=note)

frame = VoiceFrame(
    voice_id=0,
    note=note,
    f0_hz=440.0,
    gain_q15=32767,
    beta_target_q15=0,  # Pure carrier, no wet
)

# Render several blocks to let it stabilize
print("=" * 60)
print("CARRIER VS WET DEBUG")
print("=" * 60)

print("\n=== Pure Carrier (beta=0) ===")
for i in range(5):
    L, R = stage.render_block(frame)
    arr = np.array(L)
    print(f"Block {i}: Peak={np.max(np.abs(arr)):5d}, RMS={np.sqrt(np.mean(arr.astype(float)**2)):7.1f}, ZeroCross={np.sum(np.diff(np.sign(arr)) != 0):2d}")

# Now switch to wet path
print("\n=== With Wet Path (beta=0.5) ===")
frame_wet = VoiceFrame(
    voice_id=0,
    note=note,
    f0_hz=440.0,
    gain_q15=32767,
    beta_target_q15=16384,  # 50% wet
)

for i in range(10):
    L, R = stage.render_block(frame_wet)
    arr = np.array(L)
    print(f"Block {i}: Peak={np.max(np.abs(arr)):5d}, RMS={np.sqrt(np.mean(arr.astype(float)**2)):7.1f}, ZeroCross={np.sum(np.diff(np.sign(arr)) != 0):2d}")

# Analyze the last block in detail
print("\n=== Waveform Analysis (last block) ===")
print(f"First 20 samples: {arr[:20].tolist()}")
print(f"Sample range: [{np.min(arr)}, {np.max(arr)}]")

# Check periodicity - for 440Hz at 48kHz, period is ~109 samples
expected_period = 48000 / 440
print(f"\nExpected period for 440Hz: {expected_period:.1f} samples")

# Find peaks to measure actual period
peaks = []
for i in range(1, len(arr)-1):
    if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > 1000:
        peaks.append(i)

if len(peaks) >= 2:
    actual_period = np.mean(np.diff(peaks))
    print(f"Measured period: {actual_period:.1f} samples")
else:
    print(f"Could not measure period (found {len(peaks)} peaks)")

# Now test fresh voice with brass_stab preset
print("\n" + "=" * 60)
print("BRASS STAB PRESET TEST")
print("=" * 60)

from symcrash_python_ap.library.preset_data import load_preset
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude

ir = load_preset("brass_stab")
stable = ir.layers.L3_cmmm.emit_templates.get("Stable")
ops_enable = stable.ops_enable if stable else []
families = stable.families if stable else {}
beta_q15 = int(0.5 * 32767)  # 50% wet

partial_amps = [get_partial_amplitude(WaveType.SAW, k + 1) for k in range(16)]

# Fresh pipeline
stage2 = UnifiedOutputStage()
stage2.voice_on(0, note=69)

print("\nRendering brass_stab with operators...")
for i in range(10):
    frame = VoiceFrame(
        voice_id=0,
        note=69,
        f0_hz=440.0,
        gain_q15=int(32767 * 0.85),
        beta_target_q15=beta_q15,
        partial_amps_q15=partial_amps,
        ops_enable=ops_enable,
        families=families,
    )
    L, R = stage2.render_block(frame)
    arr = np.array(L)
    print(f"Block {i}: Peak={np.max(np.abs(arr)):5d}, RMS={np.sqrt(np.mean(arr.astype(float)**2)):7.1f}, ZeroCross={np.sum(np.diff(np.sign(arr)) != 0):2d}")

print(f"\nLast block first 20 samples: {arr[:20].tolist()}")
