"""
Convergence Crescendo
=====================

An original composition inspired by the concept of chaotic tones
converging to a powerful chord. Uses the wavelet synthesis pipeline.

Structure:
- 30 voices start at random frequencies (chaos)
- Over ~25 seconds, converge to a massive D major chord
- Build intensity throughout
- Final sustain on the resolved chord

Uses the symcrash_python_ap audio pipeline.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Import from the symcrash_python_ap package
from symcrash_python_ap.fixed import sat16
from symcrash_python_ap.wavelet.wavelet_packet_trees import COEFF_LEN
from symcrash_python_ap.audio_io import write_audio


# =============================================================================
# CONFIGURATION
# =============================================================================

FS = 48000
BLOCK_SIZE = 64
DURATION = 30.0  # seconds
TOTAL_SAMPLES = int(DURATION * FS)
TOTAL_BLOCKS = TOTAL_SAMPLES // BLOCK_SIZE

N_VOICES = 30
N_LEAVES = 32
NODE_SIZE = 8

# Target chord: D major across multiple octaves
# D2, A2, D3, F#3, A3, D4, F#4, A4, D5, F#5, A5, D6
TARGET_FREQS = [
    # Bass foundation
    73.42,   # D2
    110.0,   # A2
    # Lower mid
    146.83,  # D3
    185.0,   # F#3
    220.0,   # A3
    # Mid
    293.66,  # D4
    369.99,  # F#4
    440.0,   # A4
    # Upper mid
    587.33,  # D5
    739.99,  # F#5
    880.0,   # A5
    # High
    1174.66, # D6
    1479.98, # F#6
    1760.0,  # A6
    # Very high (shimmer)
    2349.32, # D7
    2959.96, # F#7
]


@dataclass
class Voice:
    """A single voice in the convergence."""
    voice_id: int
    start_freq: float      # Random starting frequency
    target_freq: float     # Target frequency in final chord
    current_freq: float    # Current frequency
    phase: float = 0.0     # Oscillator phase
    amplitude: float = 0.0 # Current amplitude
    pan: float = 0.0       # Stereo position (-1 to 1)

    # Trajectory parameters (randomized per voice)
    convergence_start: float = 0.0   # When to start converging (0-1)
    convergence_speed: float = 1.0   # How fast to converge


def build_voices() -> List[Voice]:
    """Create voices with random starting positions."""
    voices = []

    # Distribute target frequencies among voices
    targets = TARGET_FREQS.copy()

    # Add more voices on bass and mid frequencies for power
    while len(targets) < N_VOICES:
        # Duplicate lower frequencies for more bass power
        targets.append(targets[len(targets) % len(TARGET_FREQS)])

    np.random.seed(42)  # Reproducible randomness

    for i in range(N_VOICES):
        target = targets[i % len(targets)]

        # Random starting frequency (anywhere from 100 Hz to 1500 Hz)
        # Biased toward mid-range for initial chaos
        start_freq = 100 + np.random.random() * 1400

        # Random pan position (will converge toward center for bass)
        pan = (np.random.random() - 0.5) * 2  # -1 to 1

        # Random convergence timing
        # Some voices start converging early, some late
        conv_start = 0.1 + np.random.random() * 0.4  # 10% to 50% into piece
        conv_speed = 0.5 + np.random.random() * 1.5  # Variable speed

        voices.append(Voice(
            voice_id=i,
            start_freq=start_freq,
            target_freq=target,
            current_freq=start_freq,
            phase=np.random.random() * 2 * np.pi,
            amplitude=0.0,
            pan=pan,
            convergence_start=conv_start,
            convergence_speed=conv_speed,
        ))

    return voices


# =============================================================================
# SYNTHESIS
# =============================================================================

def update_voice(voice: Voice, t: float, block: int) -> None:
    """
    Update voice parameters based on time position.

    t: 0-1 progress through piece
    """
    # Amplitude envelope
    # Fade in during first 10%, full during middle, slight swell at end
    if t < 0.1:
        amp_env = t / 0.1
    elif t < 0.85:
        amp_env = 1.0 + 0.3 * (t - 0.1)  # Gradual build
    else:
        amp_env = 1.3 + 0.5 * ((t - 0.85) / 0.15)  # Final swell

    voice.amplitude = amp_env * 0.7

    # Frequency convergence
    if t < voice.convergence_start:
        # Still in chaos phase - add some random wandering
        wander = np.sin(block * 0.01 * (voice.voice_id + 1)) * 50
        voice.current_freq = voice.start_freq + wander
    else:
        # Converging toward target
        conv_progress = (t - voice.convergence_start) / (1.0 - voice.convergence_start)
        conv_progress = min(1.0, conv_progress * voice.convergence_speed)

        # Smooth convergence curve (ease-in-out)
        conv_curve = 0.5 - 0.5 * np.cos(conv_progress * np.pi)

        voice.current_freq = (
            voice.start_freq * (1 - conv_curve) +
            voice.target_freq * conv_curve
        )

    # Pan convergence (bass moves to center)
    if voice.target_freq < 200:
        # Bass frequencies converge to center
        pan_target = 0.0
    else:
        # Higher frequencies keep some stereo spread
        pan_target = voice.pan * 0.5

    if t > 0.5:
        pan_progress = (t - 0.5) / 0.5
        voice.pan = voice.pan * (1 - pan_progress) + pan_target * pan_progress


def synthesize_block(voices: List[Voice], block: int) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize one block of audio (optimized with numpy)."""
    L = np.zeros(BLOCK_SIZE, dtype=np.float64)
    R = np.zeros(BLOCK_SIZE, dtype=np.float64)

    t_samples = np.arange(BLOCK_SIZE)

    for voice in voices:
        if voice.amplitude < 0.001:
            continue

        omega = 2 * np.pi * voice.current_freq / FS
        phase_array = voice.phase + omega * t_samples

        # Harmonics (vectorized)
        sample = np.zeros(BLOCK_SIZE)

        # Fundamental
        sample += np.sin(phase_array)

        # Add harmonics based on frequency
        if voice.current_freq < FS / 4:
            sample += 0.5 * np.sin(phase_array * 2)
        if voice.current_freq < FS / 6:
            sample += 0.25 * np.sin(phase_array * 3)
        if voice.current_freq < FS / 8:
            sample += 0.15 * np.sin(phase_array * 4)
        if voice.current_freq < FS / 10:
            sample += 0.1 * np.sin(phase_array * 5)

        sample *= voice.amplitude * 1500

        # Stereo panning
        left_gain = np.sqrt(0.5 * (1 - voice.pan))
        right_gain = np.sqrt(0.5 * (1 + voice.pan))

        L += sample * left_gain
        R += sample * right_gain

        # Update phase
        voice.phase += omega * BLOCK_SIZE
        voice.phase = voice.phase % (2 * np.pi)

    # Soft clip
    def soft_clip_array(x):
        result = np.copy(x)
        mask = np.abs(x) > 20000
        result[mask] = np.sign(x[mask]) * (20000 + (np.abs(x[mask]) - 20000) * 0.3)
        return result

    L = soft_clip_array(L)
    R = soft_clip_array(R)

    L_int = np.clip(L, -32768, 32767).astype(np.int16)
    R_int = np.clip(R, -32768, 32767).astype(np.int16)

    return L_int, R_int


# =============================================================================
# MAIN
# =============================================================================

def generate_convergence() -> Tuple[List[int], List[int]]:
    """Generate the convergence crescendo."""

    print("=" * 60)
    print("CONVERGENCE CRESCENDO")
    print("=" * 60)
    print(f"Duration: {DURATION:.1f}s")
    print(f"Voices: {N_VOICES}")
    print(f"Target chord: D major (multiple octaves)")
    print()

    voices = build_voices()

    print("Voice distribution:")
    print(f"  Starting freqs: {min(v.start_freq for v in voices):.0f} - {max(v.start_freq for v in voices):.0f} Hz")
    print(f"  Target freqs: {min(v.target_freq for v in voices):.0f} - {max(v.target_freq for v in voices):.0f} Hz")
    print()

    # Preallocate output arrays
    all_L = np.zeros(TOTAL_SAMPLES, dtype=np.int16)
    all_R = np.zeros(TOTAL_SAMPLES, dtype=np.int16)

    for block in range(TOTAL_BLOCKS):
        t = block / TOTAL_BLOCKS

        # Update all voices
        for voice in voices:
            update_voice(voice, t, block)

        # Synthesize
        L, R = synthesize_block(voices, block)

        # Store in preallocated array
        start_idx = block * BLOCK_SIZE
        all_L[start_idx:start_idx + BLOCK_SIZE] = L
        all_R[start_idx:start_idx + BLOCK_SIZE] = R

        # Progress
        if block % (TOTAL_BLOCKS // 10) == 0:
            pct = 100 * block / TOTAL_BLOCKS
            avg_freq = np.mean([v.current_freq for v in voices])
            print(f"  {pct:3.0f}% - avg freq: {avg_freq:.0f} Hz")

    print(f"  100% - complete")
    print()

    # Write WAV using audio_io
    write_audio("convergence_crescendo", all_L, all_R)

    # Analysis
    L_arr = all_L.astype(np.float32)
    R_arr = all_R.astype(np.float32)

    print()
    print("Analysis:")
    print(f"  Peak: {max(np.max(np.abs(L_arr)), np.max(np.abs(R_arr))):.0f} ({100*np.max(np.abs(L_arr))/32768:.1f}%)")
    print(f"  RMS: {np.sqrt(np.mean(L_arr**2)):.0f}")

    # Frequency analysis at different points
    print()
    print("Frequency evolution:")
    for t_pct in [0.1, 0.3, 0.5, 0.7, 0.9]:
        start_idx = int(t_pct * len(L_arr))
        seg = L_arr[start_idx:start_idx + FS]  # 1 second window

        fft = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1/FS)

        # Find dominant frequencies
        top_indices = np.argsort(fft)[-5:][::-1]
        top_freqs = [freqs[i] for i in top_indices if fft[i] > fft.max() * 0.1]

        print(f"  t={t_pct*100:.0f}%: {', '.join(f'{f:.0f}' for f in top_freqs[:4])} Hz")

    return all_L.tolist(), all_R.tolist()


if __name__ == "__main__":
    generate_convergence()
