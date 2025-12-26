"""Quick script to re-render just brass_stab and analyze."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from demo_fur_elise import render_melody_with_preset, save_audio
import numpy as np

print("Rendering brass_stab with fixed envelope...")
L, R, config = render_melody_with_preset("brass_stab", bpm=120)

arr = np.array(L, dtype=np.float64)
duration = len(L) / 48000
peak = int(np.max(np.abs(arr)))
rms = np.sqrt(np.mean(arr ** 2))

print(f"Duration: {duration:.2f}s")
print(f"Peak: {peak} ({100*peak/32767:.1f}%)")
print(f"RMS: {rms:.0f}")
adsr = config["adsr"]
print(f"ADSR: A={adsr['a_ms']}ms D={adsr['d_ms']}ms S={adsr['s']} R={adsr['r_ms']}ms")

filepath = save_audio("fur_elise_brass_stab", L, R)
print(f"Saved: {filepath}")
