#!/usr/bin/env python3
"""
Generate test WAV with anticlick bypass.

This uses the legacy path (not VM executor) to test the BYPASS_ANTICLICK flag.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import wave
import numpy as np

# Disable VM executor to use legacy path with BYPASS_ANTICLICK
import symcrash_python_ap.safety.unified_renderer as ur
ur.USE_VM_EXECUTOR = False

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

def main():
    stage = UnifiedOutputStage(block_size=64)
    stage.voice_on(0, 69)  # A4

    # Full wet (eta=32767) with bypass
    all_samples = []
    for i in range(1000):  # ~1.3 seconds
        frame = VoiceFrame(
            voice_id=0,
            note=69,  # A4
            cents=0,
            gain_q15=24576,
            pan_q15=0,
            beta_target_q15=32767,  # 100% wet
            is_active=True,
            partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
        )
        left, right = stage.render_block(frame)
        all_samples.extend(left)

    # Write to WAV
    samples = np.array(all_samples, dtype=np.int16)
    output_path = Path(__file__).parent.parent / "test_wet_bypass.wav"
    with wave.open(str(output_path), 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes(samples.tobytes())

    print("Wrote {} samples to {}".format(len(all_samples), output_path))
    print("RMS: {:.1f}".format(np.sqrt(np.mean(samples.astype(np.float64)**2))))
    print("Max: {}".format(np.max(np.abs(samples))))


if __name__ == "__main__":
    main()
