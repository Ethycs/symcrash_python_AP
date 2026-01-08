"""Render a single long note and save for visual inspection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.library.preset_data import load_preset
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude
import numpy as np

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)

Q15_ONE = 32767

# Load brass_stab preset
ir = load_preset("brass_stab")
stable = ir.layers.L3_cmmm.emit_templates.get("Stable")
ops_enable = stable.ops_enable if stable else []
families = stable.families if stable else {}
beta_q15 = int(0.5 * Q15_ONE)

partial_amps = [get_partial_amplitude(WaveType.SAW, k + 1) for k in range(16)]

# Create pipeline
stage = UnifiedOutputStage()

# Render A4 (440 Hz) for 1 second
note = 69
duration_sec = 1.0
num_blocks = int(duration_sec * 750)  # 750 blocks/sec at 48kHz/64

stage.voice_on(0, note=note)

samples_L = []
samples_R = []

print(f"Rendering {num_blocks} blocks ({duration_sec}s) of A4 (440 Hz)...")

for block_idx in range(num_blocks):
    # Simple envelope: fast attack, sustain
    if block_idx < 2:
        env = 0.5 + 0.5 * (block_idx + 1) / 2
    else:
        env = 1.0

    frame = VoiceFrame(
        voice_id=0,
        note=note,
        f0_hz=440.0,
        gain_q15=int(Q15_ONE * 0.85 * env),
        beta_target_q15=beta_q15,
        partial_amps_q15=partial_amps,
        ops_enable=ops_enable,
        families=families,
    )

    L, R = stage.render_block(frame)
    samples_L.extend(L)
    samples_R.extend(R)

stage.voice_off(0)

# Analyze
arr = np.array(samples_L, dtype=np.float64)
peak = int(np.max(np.abs(arr)))
rms = np.sqrt(np.mean(arr ** 2))

print(f"Peak: {peak} ({100*peak/32767:.1f}%)")
print(f"RMS: {rms:.0f}")

# Save
filepath = audio_io.write_audio(
    "single_note_A4_brass_stab", samples_L, samples_R,
    subdirectory="debug",
    timestamp=False
).filepath
print(f"Saved: {filepath}")

# Also save a pure carrier version for comparison
print("\n--- Pure Carrier Version ---")
stage2 = UnifiedOutputStage()
stage2.voice_on(0, note=note)

samples_L2 = []
samples_R2 = []

for block_idx in range(num_blocks):
    if block_idx < 2:
        env = 0.5 + 0.5 * (block_idx + 1) / 2
    else:
        env = 1.0

    frame = VoiceFrame(
        voice_id=0,
        note=note,
        f0_hz=440.0,
        gain_q15=int(Q15_ONE * 0.85 * env),
        beta_target_q15=0,  # Pure carrier
        partial_amps_q15=partial_amps,
    )

    L, R = stage2.render_block(frame)
    samples_L2.extend(L)
    samples_R2.extend(R)

stage2.voice_off(0)

arr2 = np.array(samples_L2, dtype=np.float64)
peak2 = int(np.max(np.abs(arr2)))
rms2 = np.sqrt(np.mean(arr2 ** 2))

print(f"Peak: {peak2} ({100*peak2/32767:.1f}%)")
print(f"RMS: {rms2:.0f}")

filepath2 = audio_io.write_audio(
    "single_note_A4_pure_carrier", samples_L2, samples_R2,
    subdirectory="debug",
    timestamp=False
).filepath
print(f"Saved: {filepath2}")
