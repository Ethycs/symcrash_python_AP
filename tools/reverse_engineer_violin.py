"""
Reverse-engineer reference violin.wav using FWPT
=================================================

Uses Forward Wavelet Packet Transform to analyze the spectral structure
of the reference violin sample, extracting leaf coefficients that can
be used to drive the IDWPT synthesis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import wave

from symcrash_python_ap.wavelet.fast_lifting import fwpt_balanced, idwpt_balanced
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter

# Constants
BLOCK_SIZE = 64
FS = 48000
BLOCKS_PER_SEC = FS // BLOCK_SIZE


def load_wav_blocks(filepath, target_fs=48000):
    """Load WAV file and split into 64-sample blocks."""
    with wave.open(filepath, 'rb') as w:
        params = w.getparams()
        raw = w.readframes(params.nframes)

        if params.sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16)
        elif params.sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32)
            samples = (samples / 2147483648 * 32767).astype(np.int16)
        else:
            samples = np.frombuffer(raw, dtype=np.int8).astype(np.int16) * 256

        # Take mono
        if params.nchannels == 2:
            samples = samples[::2]

        # Resample if needed
        if params.framerate != target_fs:
            from scipy.signal import resample
            new_len = int(len(samples) * target_fs / params.framerate)
            samples = resample(samples, new_len).astype(np.int16)

        # Split into blocks
        num_blocks = len(samples) // BLOCK_SIZE
        blocks = []
        for i in range(num_blocks):
            block = samples[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].tolist()
            blocks.append(block)

        return blocks, params.framerate


def analyze_violin_coefficients(blocks):
    """Analyze coefficient structure across blocks."""
    print(f"\nAnalyzing {len(blocks)} blocks...")

    # Take blocks from sustained portion (skip attack)
    start_block = len(blocks) // 4
    end_block = 3 * len(blocks) // 4
    analysis_blocks = blocks[start_block:end_block]

    print(f"Using blocks {start_block}-{end_block} (sustained portion)")

    # Accumulate leaf statistics
    num_leaves = 64  # 2^6 for depth 6
    leaf_energies = np.zeros(num_leaves)
    leaf_means = np.zeros(num_leaves)
    num_samples = 0

    all_coeffs = []

    for block in analysis_blocks:
        # Forward transform
        leaves = fwpt_balanced(block, depth=6)

        # Each leaf has 1 coefficient for depth 6
        for i, leaf in enumerate(leaves):
            if len(leaf) > 0:
                coeff = leaf[0]
                leaf_means[i] += coeff
                leaf_energies[i] += coeff * coeff

        all_coeffs.append([l[0] if len(l) > 0 else 0 for l in leaves])
        num_samples += 1

    leaf_means /= num_samples
    leaf_energies /= num_samples
    leaf_rms = np.sqrt(leaf_energies)

    return leaf_means, leaf_rms, np.array(all_coeffs)


def print_leaf_structure(leaf_rms):
    """Print which leaves carry energy."""
    print("\n=== Leaf Energy Distribution ===")
    print("Leaf | RMS Energy | Bar")
    print("-" * 50)

    max_rms = np.max(leaf_rms)
    active_leaves = []

    for i, rms in enumerate(leaf_rms):
        if rms > max_rms * 0.05:  # Show leaves with >5% of max energy
            bar_len = int(rms / max_rms * 30)
            bar = "#" * bar_len
            print(f"{i:4d} | {rms:10.0f} | {bar}")
            active_leaves.append((i, rms))

    print(f"\nActive leaves (>5% energy): {len(active_leaves)}")
    return active_leaves


def extract_representative_coeffs(all_coeffs, leaf_rms):
    """Extract a representative coefficient set."""
    # Average the coefficients, weighted by energy
    avg_coeffs = np.mean(all_coeffs, axis=0)

    # Normalize to Q15 range
    max_abs = np.max(np.abs(avg_coeffs))
    if max_abs > 0:
        scaled = (avg_coeffs / max_abs * 16384).astype(np.int16)
    else:
        scaled = np.zeros(64, dtype=np.int16)

    return scaled.tolist()


def render_from_coeffs(coeffs, num_blocks=150, note=69):
    """Render audio by injecting the extracted coefficients."""
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)

    samples = []

    for block_idx in range(num_blocks):
        # Create frame with coefficients injected
        frame = VoiceFrame(
            note=note,
            coeff_osc_mode=1,  # Mode B
            H_request=8,
            gain_q15=24576,
            beta_target_q15=0,  # Pure carrier to test
            coeff_q15=coeffs,   # Inject the extracted coefficients
            waveform_type=0,    # Sine base
        )

        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())

    return np.array(samples, dtype=np.int16)


def main():
    print("=" * 60)
    print("REVERSE ENGINEERING VIOLIN.WAV")
    print("=" * 60)

    # Load reference
    blocks, orig_fs = load_wav_blocks("violin.wav")
    print(f"\nLoaded violin.wav: {len(blocks)} blocks, original {orig_fs} Hz")

    # Analyze coefficients
    leaf_means, leaf_rms, all_coeffs = analyze_violin_coefficients(blocks)

    # Print structure
    active_leaves = print_leaf_structure(leaf_rms)

    # Extract representative coefficients
    rep_coeffs = extract_representative_coeffs(all_coeffs, leaf_rms)

    print("\n=== Representative Coefficients (Q15) ===")
    print("coeffs = [")
    for i in range(0, 64, 8):
        row = rep_coeffs[i:i+8]
        print(f"    {', '.join(f'{c:6d}' for c in row)},")
    print("]")

    # Show which leaves map to which frequencies
    print("\n=== Leaf -> Frequency Mapping (for 440 Hz fundamental) ===")
    # At 48kHz, each leaf covers ~375 Hz (48000/2/64)
    leaf_bw = 48000 / 2 / 64
    for leaf_idx, rms in active_leaves[:10]:
        freq_low = leaf_idx * leaf_bw
        freq_high = (leaf_idx + 1) * leaf_bw
        print(f"Leaf {leaf_idx:2d}: {freq_low:.0f}-{freq_high:.0f} Hz, RMS={rms:.0f}")

    # Test render
    print("\n=== Test Render ===")
    test_samples = render_from_coeffs(rep_coeffs, num_blocks=150, note=69)

    # Save
    OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_generation_tests" / "audio_out")
    audio_io = AudioWriter(output_dir=OUTPUT_DIR)

    L = test_samples.tolist()
    R = test_samples.tolist()

    metadata = audio_io.write_audio(
        "violin_reverse_engineered",
        L, R,
        subdirectory="presets",
        timestamp=False
    )

    print(f"\nSaved: {metadata.filepath}")

    # Also print the leaf energies as harmonic-like structure
    print("\n=== Recommended Harmonic Structure ===")
    print("Based on active leaf analysis, use these partial amplitudes:")

    # Map leaf indices to harmonic-like structure
    # Leaves 0-3 are low freq, 4-7 are mid-low, etc.
    harmonic_groups = [
        np.mean(leaf_rms[0:4]),    # H1 region
        np.mean(leaf_rms[4:8]),    # H2 region
        np.mean(leaf_rms[8:12]),   # H3 region
        np.mean(leaf_rms[12:16]),  # H4 region
        np.mean(leaf_rms[16:24]),  # H5-6 region
        np.mean(leaf_rms[24:32]),  # H7-8 region
        np.mean(leaf_rms[32:48]),  # Upper harmonics
        np.mean(leaf_rms[48:64]),  # Very high
    ]

    max_h = max(harmonic_groups)
    print("harmonic_amps = [")
    for i, h in enumerate(harmonic_groups):
        q15_val = int(h / max_h * 32767) if max_h > 0 else 0
        pct = h / max_h * 100 if max_h > 0 else 0
        print(f"    {q15_val:5d},  # H{i+1}: {pct:.1f}%")
    print("]")


if __name__ == "__main__":
    main()
