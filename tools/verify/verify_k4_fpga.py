"""
Verify K=4 FPGA-compliant configuration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, ModeBState,
    clear_basis_atlas, BLOCK_SIZE, SAMPLE_RATE, VM_K, K_STRIDE, VM_K_RECONSTRUCT
)
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

def main():
    print("K=4 FPGA Compliance Verification")
    print("=" * 50)
    print(f"VM_K (storage slots): {VM_K}")
    print(f"VM_K_RECONSTRUCT:     {VM_K_RECONSTRUCT}")
    print(f"K_STRIDE:             {K_STRIDE}")
    print()

    # Check configuration for K=4 consecutive block polyphase
    assert VM_K == 4, f"VM_K should be 4, got {VM_K}"
    assert VM_K_RECONSTRUCT == 4, f"VM_K_RECONSTRUCT should be 4, got {VM_K_RECONSTRUCT}"
    assert K_STRIDE == 16, f"K_STRIDE should be 16 for K=4, got {K_STRIDE}"

    print("Configuration VALID for K=4 FPGA:")
    print("  - Storage uses 64x4=256 coefficients")
    print("  - Reconstruction uses 4 consecutive-block polyphase")
    print("  - Slot k handles samples [k*16 : (k+1)*16]")
    print("  - Basis frames use 16-sample phase offsets (k_stride=16)")
    print()

    # Quick render test
    note = 69
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_sine()
    mb_state.H_active = 1

    # Render 10 blocks
    coeff_len = 64 * VM_K  # 256 for K=4, 512 for K=8
    samples = []
    for _ in range(10):
        carrier = np.zeros(coeff_len, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier)

        coeff_2d = carrier.reshape(64, VM_K)
        block_samples = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        samples.extend(block_samples.tolist())
        mb_state.advance_block()

    samples = np.array(samples, dtype=np.float64)
    print(f"Render test: {len(samples)} samples")
    print(f"Max amplitude: {np.max(np.abs(samples)):.0f}")
    print()
    print("K=4 FPGA-COMPLIANT CONFIGURATION VERIFIED")

if __name__ == "__main__":
    main()
