"""Debug dc_hat values during processing."""
import numpy as np

from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

BLOCK_SIZE = 64


def test_dc_hat():
    """Track dc_hat across blocks."""
    frame = VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        gain_q15=24576,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=16384,  # Non-zero to trigger DC removal
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )

    stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
    stage.voice_on(0, frame.note, frame.cents)

    dc_hat_values = []
    dc_offset_values = []

    for i in range(100):
        samples = stage.render_voice(frame)

        # Track dc_hat before processing
        dc_hat_before = stage.safety.rg.dc_hat

        # Apply pan (center)
        from symcrash_python_ap.wavelet.sine_core import cosine_q15, PHASE_MASK
        from symcrash_python_ap.fixed import mul_q15_sat
        pan_phase = (32768 << 14) & PHASE_MASK
        left_q15 = cosine_q15(pan_phase)
        L = [mul_q15_sat(s, left_q15) for s in samples]
        R = L.copy()

        # Measure actual DC offset of input signal
        dc_offset = sum(L) // len(L)
        dc_offset_values.append(dc_offset)

        # Apply safety with DC removal
        L_out, R_out = stage.safety.step_block(
            L, R, stage.params,
            apply_master_ola=False,
            apply_dc_removal=True
        )

        # Track dc_hat after processing
        dc_hat_after = stage.safety.rg.dc_hat
        dc_hat_values.append(dc_hat_after)

        if i < 10 or i % 10 == 0:
            # Measure discontinuity at block boundary
            if i > 0:
                diff = abs(L_out[0] - prev_last)
            else:
                diff = 0
            print(f"Block {i}: dc_hat={dc_hat_after:5d}, dc_offset={dc_offset:5d}, boundary_diff={diff}")

        prev_last = L_out[-1]

    print(f"\ndc_hat range: {min(dc_hat_values)} to {max(dc_hat_values)}")
    print(f"dc_offset range: {min(dc_offset_values)} to {max(dc_offset_values)}")
    print(f"Mean dc_offset: {np.mean(dc_offset_values):.1f}")


test_dc_hat()
