"""Debug table generation to find frequency error source."""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

BLOCK_SIZE = 64
NUM_LEAVES = 64
SAMPLE_RATE = 48000
K = 8
P = 64
S = 8


def analyze_table_gen(target_freq_hz: float):
    """Trace through table generation."""
    print(f"\n{'='*60}")
    print(f"Table Generation for {target_freq_hz:.2f} Hz")
    print(f"{'='*60}")

    L = NUM_LEAVES
    M = P * K * S  # = 64 * 8 * 8 = 4096

    # Phase increment for target frequency
    phase_inc = int((target_freq_hz / SAMPLE_RATE) * (1 << 32))

    print(f"  M = {M} samples")
    print(f"  phase_inc = {phase_inc}")
    print(f"  phase_inc * M = {phase_inc * M}")
    print(f"  cycles in M = {(phase_inc * M) / (1 << 32):.2f}")

    # Generate waveform
    waveform = np.zeros(M, dtype=np.float32)
    phase = 0
    for n in range(M):
        waveform[n] = sine_q15(phase)
        phase = (phase + phase_inc) & 0xFFFFFFFF

    print(f"  Waveform range: [{waveform.min():.0f}, {waveform.max():.0f}]")

    # Zero crossings to verify frequency
    crossings = 0
    for i in range(1, M):
        if waveform[i-1] <= 0 < waveform[i]:
            crossings += 1
    print(f"  Zero crossings (rising): {crossings}")
    print(f"  Implied cycles: {crossings}")
    print(f"  Implied freq: {crossings * SAMPLE_RATE / M:.2f} Hz")

    # Analysis blocks
    num_analysis_blocks = M // BLOCK_SIZE
    print(f"  Analysis blocks: {num_analysis_blocks}")

    subband_streams = np.zeros((L, num_analysis_blocks), dtype=np.float32)
    for block_idx in range(num_analysis_blocks):
        start = block_idx * BLOCK_SIZE
        audio_block = waveform[start:start + BLOCK_SIZE].astype(np.int16).tolist()
        coeffs = fwpt_jit(audio_block, headroom_bits=1)
        for l in range(L):
            subband_streams[l, block_idx] = coeffs[l]

    print(f"  Subband stream shape: {subband_streams.shape}")
    print(f"  Leaf 1 stream: {subband_streams[1, :]}")

    # Calculate mapping parameters
    samples_per_osc_cycle = SAMPLE_RATE / target_freq_hz
    blocks_per_osc_cycle = samples_per_osc_cycle / BLOCK_SIZE

    print(f"\n  samples_per_osc_cycle = {samples_per_osc_cycle:.1f}")
    print(f"  blocks_per_osc_cycle = {blocks_per_osc_cycle:.2f}")

    # Show mapping for key (p, k) values
    print(f"\n  Phase step mapping:")
    for p in [0, 16, 32, 48, 63]:
        k = 0
        frac_in_cycle = (p * K + k) / (P * K)
        block_idx_float = frac_in_cycle * blocks_per_osc_cycle
        idx0 = int(block_idx_float) % num_analysis_blocks

        print(f"    p={p:2d}, k={k}: frac={frac_in_cycle:.3f} -> block_idx={block_idx_float:.2f} -> idx0={idx0}")

    # Now trace what happens at runtime
    print(f"\n  Runtime phase progression:")
    phase_inc_runtime = int((target_freq_hz / SAMPLE_RATE) * (1 << 32))
    phase = 0
    for block in range(5):
        phase_idx = (phase >> 26) & 0x3F
        frac_26bit = phase & 0x3FFFFFF
        alpha = frac_26bit / float(1 << 26)

        print(f"    Block {block}: phase_idx={phase_idx:2d}, alpha={alpha:.3f}")

        phase = (phase + phase_inc_runtime * BLOCK_SIZE) & 0xFFFFFFFF

    # The key question: does phase_idx advance at the same rate as block_idx_float?
    # phase_idx advances by (phase_inc * BLOCK_SIZE) >> 26 per block
    phase_idx_advance = (phase_inc_runtime * BLOCK_SIZE) / (1 << 26)
    block_idx_advance = blocks_per_osc_cycle / P  # Because each phase step is 1/P of a cycle

    print(f"\n  phase_idx advance per block: {phase_idx_advance:.2f}")
    print(f"  Expected table index advance per block: {block_idx_advance * P:.2f}")

    # These should match!
    # phase_idx_advance = phase_inc * BLOCK_SIZE / 2^26
    # block_idx_advance = blocks_per_osc_cycle / P
    #                   = (SAMPLE_RATE / target_freq_hz / BLOCK_SIZE) / P
    #                   = SAMPLE_RATE / (target_freq_hz * BLOCK_SIZE * P)

    # Meanwhile: phase_inc = (target_freq_hz / SAMPLE_RATE) * 2^32
    # So: phase_idx_advance = target_freq_hz / SAMPLE_RATE * 2^32 * BLOCK_SIZE / 2^26
    #                       = target_freq_hz * BLOCK_SIZE * 2^6 / SAMPLE_RATE
    #                       = target_freq_hz * BLOCK_SIZE * 64 / SAMPLE_RATE

    # And: block_idx_advance * P = blocks_per_osc_cycle
    #                            = SAMPLE_RATE / (target_freq_hz * BLOCK_SIZE)

    # So phase_idx_advance / (block_idx_advance * P)
    #    = target_freq_hz * BLOCK_SIZE * 64 / SAMPLE_RATE * target_freq_hz * BLOCK_SIZE / SAMPLE_RATE
    #    = target_freq_hz^2 * BLOCK_SIZE^2 * 64 / SAMPLE_RATE^2

    # For 261.63 Hz:
    # = 261.63^2 * 64^2 * 64 / 48000^2
    # = 68450 * 4096 * 64 / 2304000000
    # = 17944576000 / 2304000000 = 7.79

    # Hmm, this doesn't match up. The rates are different.

    # Actually I think the issue is that the table uses blocks_per_osc_cycle
    # but the runtime uses phase index (0-63) which maps P steps to one cycle.

    # The table should be indexed such that p=0..P-1 covers exactly one oscillator cycle.
    # At runtime, phase goes 0..2^32-1 for one cycle.
    # Top 6 bits of phase give phase_idx 0-63, which is P=64 steps per cycle.

    # So the mapping is:
    # phase_idx (0-63) -> table_idx (0-63) directly!
    # The blocks_per_osc_cycle should not affect the runtime lookup,
    # only the table generation.

    print(f"\n  Conclusion:")
    print(f"  The table is generated with blocks_per_osc_cycle = {blocks_per_osc_cycle:.2f}")
    print(f"  But runtime expects P={P} phase steps per cycle")
    print(f"  Blocks in one cycle: {blocks_per_osc_cycle:.2f}")
    print(f"  Table phase steps: {P}")

    # The issue: if blocks_per_osc_cycle != some integer multiple of P / K,
    # then the table doesn't properly represent one complete cycle.

    # For 261.63 Hz: blocks_per_osc_cycle = 2.87
    # If we want P=64 phase steps to cover this, each phase step covers
    # 2.87/64 = 0.045 blocks.

    # But the table was generated with:
    # block_idx_float = frac_in_cycle * blocks_per_osc_cycle
    # where frac_in_cycle = (p*K + k) / (P*K) = (p*8 + k) / 512

    # So for p=63, k=7: frac = 0.998, block_idx = 2.86
    # The table covers up to block 2.86, which is 183.5 samples = one cycle. Correct!

    # But wait - the issue is that at runtime, phase_idx advances by more than 1 per block.
    # For 261.63 Hz: phase_idx_advance = 22.4 per block
    # So after ~3 blocks, we've gone through all 64 phase steps (one cycle).

    # This is correct! The problem must be elsewhere.


if __name__ == '__main__':
    analyze_table_gen(261.63)
