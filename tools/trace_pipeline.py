"""Trace the full audio pipeline to find where spikes are introduced."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.wavelet.sine_core import sine_q15_vectorized
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage

Q15_ONE = 32767
BLOCK_SIZE = 64
SAMPLE_RATE = 48000

def count_spikes(signal, threshold_sigma=3):
    """Count spikes in second derivative."""
    signal = np.array(signal, dtype=np.int32)
    d2 = np.diff(np.diff(signal))
    if len(d2) == 0 or d2.std() == 0:
        return 0, 0
    threshold = d2.std() * threshold_sigma
    spikes = np.where(np.abs(d2) > threshold)[0]
    return len(spikes), d2.max()


# Create a pure sine VoiceFrame (harm=0)
frame = VoiceFrame(
    voice_id=0,
    note=60,  # Middle C
    cents=0,
    f0_hz=261.63,
    gain_q15=24576,
    pan_q15=0,  # Center
    tree_a_id=0,
    tree_b_id=0,
    tree_blend_q15=0,
    is_active=True,
    beta_target_q15=0,  # No wet/wavelet
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
)

# Create output stage
stage = UnifiedOutputStage(block_size=BLOCK_SIZE)
stage.voice_on(0, frame.note, frame.cents)

# Render multiple blocks and concatenate
num_blocks = 20
all_L = []
all_R = []

for i in range(num_blocks):
    L, R = stage.render_block(frame)
    all_L.extend(L)
    all_R.extend(R)

# Check for spikes
samples_per_cycle = int(SAMPLE_RATE / frame.f0_hz)
num_cycles = len(all_L) / samples_per_cycle

print(f"Rendered {len(all_L)} samples ({num_cycles:.1f} cycles at {frame.f0_hz}Hz)")
print(f"Sample range: {min(all_L)} to {max(all_L)}")

spikes_L, max_d2_L = count_spikes(all_L)
spikes_R, max_d2_R = count_spikes(all_R)

print(f"\nLeft channel: {spikes_L} spikes ({spikes_L/num_cycles:.1f}/cycle), max d2={max_d2_L}")
print(f"Right channel: {spikes_R} spikes ({spikes_R/num_cycles:.1f}/cycle), max d2={max_d2_R}")

if spikes_L > 0:
    # Find where spikes occur
    d2 = np.diff(np.diff(np.array(all_L, dtype=np.int32)))
    threshold = d2.std() * 3
    spike_indices = np.where(np.abs(d2) > threshold)[0]

    # Check if spikes are at block boundaries
    block_boundary_spikes = [i for i in spike_indices if (i + 2) % BLOCK_SIZE < 3 or (i + 2) % BLOCK_SIZE > BLOCK_SIZE - 3]
    print(f"\nSpikes near block boundaries: {len(block_boundary_spikes)} / {len(spike_indices)}")

    # Check cycle positions
    cycle_positions = (spike_indices % samples_per_cycle) / samples_per_cycle
    print(f"Cycle positions of first 10 spikes: {cycle_positions[:10]}")
