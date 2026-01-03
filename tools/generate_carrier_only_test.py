#!/usr/bin/env python3
"""
Generate test WAV with pure carrier (no wavelet/wet path).

This tests if the carrier itself is clean.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

# Disable VM executor to use legacy path
import symcrash_python_ap.safety.unified_renderer as ur
ur.USE_VM_EXECUTOR = False

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import write_audio

def main():
    stage = UnifiedOutputStage(block_size=64)
    stage.voice_on(0, 69)  # A4

    # Pure carrier (eta=0, no wet path)
    all_samples = []
    for i in range(1000):  # ~1.3 seconds
        frame = VoiceFrame(
            voice_id=0,
            note=69,  # A4
            cents=0,
            gain_q15=24576,
            pan_q15=0,
            beta_target_q15=0,  # 0% wet = pure carrier
            is_active=True,
            partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
        )
        left, right = stage.render_block(frame)
        all_samples.extend(left)

    # Write using centralized audio_io
    samples = np.array(all_samples, dtype=np.int16)
    meta = write_audio("test_carrier_only", samples, timestamp=False, subdirectory="carrier_tests")

    print("Wrote {} samples to {}".format(len(all_samples), meta.filepath))
    print("RMS: {:.1f}".format(np.sqrt(np.mean(samples.astype(np.float64)**2))))
    print("Max: {}".format(np.max(np.abs(samples))))


if __name__ == "__main__":
    main()
