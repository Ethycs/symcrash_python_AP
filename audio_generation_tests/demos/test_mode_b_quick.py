"""Quick Mode B Test - Verifies WSE-STD-210 Mode B oscillator works."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC
from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter

Q15_ONE = 32767
Q15_HALF = 16384

def main():
    print("=" * 50)
    print("Mode B Quick Test - WSE-STD-210")
    print("=" * 50)

    # Create pipeline
    pipeline = CarrierLiftPipeline()

    # Simple A4 note test with Mode B
    samples_L = []
    samples_R = []

    pipeline.voice_on(0, note=69)

    # Render 2 seconds of A440
    for block in range(int(2 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=69,
            f0_hz=440.0,
            gain_q15=int(Q15_ONE * 0.7),
            beta_target_q15=Q15_HALF,
            partial_amps_q15=[Q15_ONE, Q15_HALF, Q15_ONE // 3, Q15_ONE // 4, 0, 0, 0, 0],
            coeff_osc_mode=1,  # Mode B!
            H_request=16,
        )
        L, R = pipeline.render_block(frame)
        samples_L.extend(L)
        samples_R.extend(R)

    pipeline.voice_off(0)

    # Analyze
    arr = np.array(samples_L, dtype=np.float64)
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr ** 2))

    print(f"Mode B Test - A440 (2s)")
    print(f"  Peak: {peak}, RMS: {rms:.0f}")
    print(f"  Samples: {len(samples_L)}")
    print(f"  Duration: {len(samples_L) / FS:.2f}s")

    # Save
    OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
    audio_io = AudioWriter(output_dir=OUTPUT_DIR)
    metadata = audio_io.write_audio(
        "mode_b_test_a440", samples_L, samples_R,
        subdirectory="mode_b_test", timestamp=False
    )
    print(f"  Saved: {metadata.filepath}")
    print("SUCCESS!")


if __name__ == "__main__":
    main()
