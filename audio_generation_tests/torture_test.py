"""
Cathedral of Stress - Main Torture Test
=======================================

5-second comprehensive test exercising all pipeline components:
- Symbolic algebra (all operator families)
- Wavelet packet trees (all 5 types + crossfades)
- Leaf atom matrices (I, DCT8, chirp, spike)
- Material engine (tape, rank, steel, catastrophes)
- Safety engine (limiter under stress)

Run this file to generate the torture test WAV.

Uses the symcrash_python_ap audio pipeline.
"""

import numpy as np
from typing import List, Tuple, Dict

from test_config import (
    FS, BLOCK_SIZE, TOTAL_BLOCKS, BLOCKS_PER_SEC,
    N_VOICES, N_LEAVES, NODE_SIZE,
    SEGMENTS, get_current_segment,
    TreeID, MaterialID, AtomID,
)
from test_segments import (
    TestState, update_segment_state,
    get_material_program, get_voice_configs,
)
from test_operators import get_operator_program, OpID
from test_atoms import (
    get_complex_leaf_mask, apply_atom_transform,
    get_blended_matrix,
)

# Import from the symcrash_python_ap package
from symcrash_python_ap.pipeline import Pipeline
from symcrash_python_ap.symbol_engine import SymbolEngine
from symcrash_python_ap.wavelet_engine import WaveletEngine
from symcrash_python_ap.material_engine import MaterialEngine
from symcrash_python_ap.safety_engine import SafetyEngine
from symcrash_python_ap.frames import ParamFrame, VoiceFrame
from symcrash_python_ap.fixed import sat16, mul_q15_sat
from symcrash_python_ap.wavelet_packet_trees import PacketTreeSet, COEFF_LEN
from symcrash_python_ap.audio_io import write_audio


# =============================================================================
# FULL PIPELINE SYNTHESIS
# =============================================================================

class TortureTestPipeline:
    """
    Full pipeline wrapper for torture test.

    Uses real symbol engine, wavelet synthesis, material processing, and safety.
    """

    def __init__(self):
        self.fs = FS
        self.B = BLOCK_SIZE

        # Core engines
        self.symbol = SymbolEngine(n_voices=N_VOICES, coeff_len=COEFF_LEN)
        self.wavelet = WaveletEngine(block_size=BLOCK_SIZE)
        self.material = MaterialEngine(block_size=BLOCK_SIZE)
        self.safety = SafetyEngine(block_size=BLOCK_SIZE)

        # Tree set for reference
        self.trees = PacketTreeSet()

        # Current parameters
        self.params = ParamFrame()

        # Voice phase states for richer synthesis
        self.voice_phases: Dict[int, Dict[int, float]] = {i: {} for i in range(N_VOICES)}

        # Audio buffer
        self.output_L: List[int] = []
        self.output_R: List[int] = []

    def inject_voice(self, voice_id: int, config, tree_id: int,
                     atom_mix: float, amplitude: int):
        """
        Inject coefficients for a voice with rich harmonic content.
        Uses logarithmic frequency mapping for better distribution.
        """
        coeffs = [0] * COEFF_LEN

        freq = config.base_freq
        nyquist = self.fs / 2

        # Logarithmic frequency-to-leaf mapping
        # Maps 20Hz-20kHz to leaves 0-31 logarithmically
        def freq_to_leaf(f):
            if f < 20:
                return 0
            if f > 20000:
                return N_LEAVES - 1
            log_pos = np.log2(f / 20) / np.log2(20000 / 20)
            return int(log_pos * (N_LEAVES - 1))

        # Generate rich harmonic series
        for harmonic in range(1, 24):
            h_freq = freq * harmonic
            if h_freq >= nyquist * 0.95:
                break

            # Map to leaf using log scale
            leaf_idx = freq_to_leaf(h_freq)
            leaf_idx = max(0, min(N_LEAVES - 1, leaf_idx))

            # Harmonic amplitude with gentler rolloff
            h_amp = amplitude // (harmonic ** 0.7)

            # Create oscillating pattern in node
            node_start = leaf_idx * NODE_SIZE

            # Get or create phase
            phase_key = (leaf_idx, harmonic)
            if phase_key not in self.voice_phases[voice_id]:
                self.voice_phases[voice_id][phase_key] = np.random.random() * 2 * np.pi

            phase = self.voice_phases[voice_id][phase_key]

            for i in range(NODE_SIZE):
                t = i / NODE_SIZE
                val = h_amp * (
                    0.6 * np.sin(2 * np.pi * t * (1 + harmonic * 0.1) + phase) +
                    0.3 * np.sin(4 * np.pi * t + phase * 1.3) +
                    0.1 * np.sin(6 * np.pi * t + phase * 0.7)
                )
                coeffs[node_start + i] = sat16(coeffs[node_start + i] + int(val))

            # Advance phase
            omega = 2 * np.pi * h_freq / self.fs * BLOCK_SIZE
            self.voice_phases[voice_id][phase_key] = (phase + omega) % (2 * np.pi)

        # Spread to adjacent leaves
        spread_coeffs = list(coeffs)
        for leaf_idx in range(N_LEAVES):
            node_start = leaf_idx * NODE_SIZE
            node_energy = sum(abs(coeffs[node_start + i]) for i in range(NODE_SIZE))

            if node_energy > 100:
                for neighbor_offset in [-1, 1]:
                    neighbor_leaf = leaf_idx + neighbor_offset
                    if 0 <= neighbor_leaf < N_LEAVES:
                        neighbor_start = neighbor_leaf * NODE_SIZE
                        for i in range(NODE_SIZE):
                            spread_val = coeffs[node_start + i] // 4
                            spread_coeffs[neighbor_start + i] = sat16(
                                spread_coeffs[neighbor_start + i] + spread_val
                            )

        coeffs = spread_coeffs

        # Apply atom transform to complex leaves
        complex_mask = get_complex_leaf_mask(tree_id)

        for leaf_idx in range(N_LEAVES):
            if complex_mask[leaf_idx] and atom_mix > 0.01:
                node_start = leaf_idx * NODE_SIZE
                node = coeffs[node_start:node_start + NODE_SIZE]

                # Choose atom based on tree
                if tree_id == TreeID.HARMONIC:
                    atom_id = AtomID.DCT8
                elif tree_id == TreeID.TEXTURE:
                    atom_id = AtomID.CHIRP
                elif tree_id == TreeID.IMPACT:
                    atom_id = AtomID.SPIKE
                else:
                    atom_id = AtomID.DCT8

                transformed = apply_atom_transform(node, atom_id, atom_mix)
                for i, v in enumerate(transformed):
                    coeffs[node_start + i] = v

        # Inject into symbol engine
        self.symbol.inject_note(voice_id, coeffs)
        self.symbol.voice_trees[voice_id] = tree_id

    def step(self, state: TestState) -> Tuple[List[int], List[int]]:
        """
        Process one block through the full pipeline.
        """
        # Update params from state
        material_prog = get_material_program(state)
        self.params.drive_q15 = material_prog.drive_q15
        self.params.pressure_q15 = material_prog.pressure_q15
        self.params.bias_q15 = material_prog.bias_q15
        self.params.tone_q15 = material_prog.tone_q15

        # Symbol engine step
        vframes = self.symbol.step_block(self.params)

        # Wavelet synthesis for each voice
        voice_blocks = []
        for vf in vframes:
            audio = self.wavelet.render_voice(vf)
            voice_blocks.append((vf, audio))

        # Mix to stereo
        L, R = self.wavelet.mix_block(voice_blocks)

        # Material processing
        L, R = self.material.step_block(
            L, R, self.params,
            material_id=material_prog.material_id,
            cat_id=0
        )

        # Safety limiting
        L, R = self.safety.step_block(L, R, self.params)

        return L, R


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_torture_test() -> Tuple[List[int], List[int]]:
    """
    Run the full torture test with real pipeline.
    """

    print("=" * 60)
    print("CATHEDRAL OF STRESS - TORTURE TEST (FULL PIPELINE)")
    print("=" * 60)
    print(f"Duration: 5.00s ({TOTAL_BLOCKS} blocks)")
    print(f"Voices: {N_VOICES}")
    print()

    # Initialize
    state = TestState()
    pipeline = TortureTestPipeline()

    all_L = []
    all_R = []

    current_segment = None
    inject_interval = 8  # Re-inject voices every 8 blocks for sustain

    for block in range(TOTAL_BLOCKS):
        state.block = block

        # Update segment state
        update_segment_state(state)

        # Log segment changes
        seg_name, seg = get_current_segment(block)
        if seg_name != current_segment:
            current_segment = seg_name
            time_sec = block / BLOCKS_PER_SEC
            print(f"[{time_sec:.2f}s] Segment {seg_name}: {seg.description}")

        # Inject voices periodically for sustain
        if block % inject_interval == 0:
            voice_configs = get_voice_configs(state)
            for idx, (config, tree_id, crossfade, amplitude) in enumerate(voice_configs):
                if amplitude > 100:
                    pipeline.inject_voice(
                        idx, config, tree_id,
                        state.atom_mix, amplitude
                    )

        # Process block
        L, R = pipeline.step(state)

        all_L.extend(L)
        all_R.extend(R)

        # Progress
        if block % 750 == 0:
            print(f"  Progress: {block}/{TOTAL_BLOCKS} ({100*block/TOTAL_BLOCKS:.0f}%)")

    print(f"  Progress: {TOTAL_BLOCKS}/{TOTAL_BLOCKS} (100%)")
    print()

    # Write WAV using audio_io
    write_audio("cathedral_of_stress", all_L, all_R)

    # Analysis
    analyze_output(all_L, all_R)

    return all_L, all_R


def analyze_output(L: List[int], R: List[int]):
    """Analyze the output audio."""
    L_arr = np.array(L, dtype=np.float32)
    R_arr = np.array(R, dtype=np.float32)

    print()
    print("=" * 60)
    print("OUTPUT ANALYSIS")
    print("=" * 60)

    print(f"Peak L: {np.max(np.abs(L_arr)):.0f} ({100*np.max(np.abs(L_arr))/32768:.1f}%)")
    print(f"Peak R: {np.max(np.abs(R_arr)):.0f} ({100*np.max(np.abs(R_arr))/32768:.1f}%)")
    print(f"RMS L: {np.sqrt(np.mean(L_arr**2)):.0f}")
    print(f"RMS R: {np.sqrt(np.mean(R_arr**2)):.0f}")
    print(f"DC L: {np.mean(L_arr):.1f}")
    print(f"DC R: {np.mean(R_arr):.1f}")

    # Zero crossings
    zc_L = np.sum(np.abs(np.diff(np.sign(L_arr))) > 0)
    print(f"Zero crossings: {zc_L}")

    # Frequency analysis
    print()
    print("Frequency content per segment:")

    for seg_name, seg in SEGMENTS.items():
        start = int(seg.start_sec * FS)
        end = int(seg.end_sec * FS)
        seg_L = L_arr[start:end]

        # FFT
        fft = np.abs(np.fft.rfft(seg_L))
        freqs = np.fft.rfftfreq(len(seg_L), 1/FS)

        # Band energy
        bands = [(20, 200), (200, 1000), (1000, 4000), (4000, 12000)]
        total = np.sum(fft**2)

        band_str = []
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            energy = np.sum(fft[mask]**2) / total * 100 if total > 0 else 0
            band_str.append(f"{energy:4.0f}%")

        # Top frequency
        top_idx = np.argmax(fft[10:]) + 10
        top_freq = freqs[top_idx]

        print(f"  {seg_name}: bands=[{', '.join(band_str)}] peak={top_freq:.0f}Hz")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_torture_test()
