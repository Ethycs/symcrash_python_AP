"""
Debug: Is dc_hat bouncing causing discontinuities?
===================================================
"""

import numpy as np
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame, ParamFrame
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, Q15_HALF


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def main():
    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    params = ParamFrame()

    dc_hats = []
    dc_offsets = []

    for block_idx in range(int(0.5 * BLOCKS_PER_SEC)):  # 0.5 seconds
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)

        # Measure metrics (which computes dc_offset)
        m = pipeline.safety.measure_metrics(L, R)
        dc_offsets.append(m.dc_offset_q15)

        # Update RG (which smooths dc_hat)
        pipeline.safety.update_rg(m, params.K_mode)
        dc_hats.append(pipeline.safety.rg.dc_hat)

    print("DC Analysis (first 50 blocks):")
    print("Block |  DC offset  |  DC hat (smoothed)")
    print("-" * 45)
    for i in range(min(50, len(dc_hats))):
        print(f"{i:5d} | {dc_offsets[i]:10d} | {dc_hats[i]:10d}")

    print(f"\nStats:")
    print(f"  DC offset range: {min(dc_offsets)} to {max(dc_offsets)}")
    print(f"  DC hat range: {min(dc_hats)} to {max(dc_hats)}")
    print(f"  DC hat bouncing: {max(dc_hats) - min(dc_hats)}")


if __name__ == "__main__":
    main()
