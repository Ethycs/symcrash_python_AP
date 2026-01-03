#!/usr/bin/env python3
"""
Generate test WAV with pure carrier - matching generate_test_audio.py pattern.

Key differences from v1:
- Uses SAME frame object for all blocks (like generate_test_audio.py)
- Uses VM executor path (default)
- Uses note=60 (Middle C) like original
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import write_audio

def main():
    stage = UnifiedOutputStage(block_size=64)

    # Create a SINGLE frame object (reused for all blocks)
    frame = VoiceFrame(
        voice_id=0,
        note=60,  # Middle C (like generate_test_audio.py)
        cents=0,
        gain_q15=24576,
        pan_q15=0,
        beta_target_q15=0,  # 0% wet = pure carrier
        is_active=True,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
    )

    stage.voice_on(0, frame.note, frame.cents)

    # Render using the SAME frame object (like generate_test_audio.py)
    all_samples = []
    for i in range(1000):  # ~1.3 seconds
        left, right = stage.render_block(frame)
        all_samples.extend(left)

    # Write using centralized audio_io
    samples = np.array(all_samples, dtype=np.int16)
    meta = write_audio("test_carrier_only_v2", samples, timestamp=False, subdirectory="carrier_tests")

    print("Wrote {} samples to {}".format(len(all_samples), meta.filepath))
    print("RMS: {:.1f}".format(np.sqrt(np.mean(samples.astype(np.float64)**2))))
    print("Max: {}".format(np.max(np.abs(samples))))


if __name__ == "__main__":
    main()
