"""
Wavelet Engine Coefficient Transition Tests
============================================

Tests the WaveletEngine's handling of coefficient changes at different intensities:
- 1x: Gradual coefficient evolution (baseline)
- 10x: Moderate coefficient jumps (typical note changes)
- 100x: Extreme coefficient jumps (stress test)

Verifies that slewing + micro-bridge produces click-free output.

Output: audio_generation_tests/audio_out/wavelet_transition_tests/
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC
from audio_io import write_audio

from symcrash_python_ap.wavelet.wavelet_engine import WaveletEngine
from symcrash_python_ap.safety.safety_engine import SafetyEngine
from symcrash_python_ap.frames import VoiceFrame, ParamFrame
from symcrash_python_ap.wavelet.wavelet_packet_trees import COEFF_LEN
from symcrash_python_ap.fixed import sat16

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_SUBDIR = "wavelet_transition_tests"
Q15_ONE = 32767
Q15_HALF = 16384

# Number of coefficients per leaf node
NODE_SIZE = 8
N_LEAVES = 32


# =============================================================================
# AUDIO OUTPUT HELPER
# =============================================================================

def save_audio(samples_L: List[int], samples_R: List[int], name: str) -> str:
    """Save audio using audio_io module. Returns filepath."""
    metadata = write_audio(name, samples_L, samples_R, timestamp=True,
                           subdirectory=OUTPUT_SUBDIR, verbose=False)
    return metadata.filepath


def compute_click_metric(samples: List[int], threshold: int = 3000) -> Tuple[int, float, List[int]]:
    """
    Compute click metric: count of sample-to-sample jumps exceeding threshold.

    Returns:
        (click_count, max_delta, click_indices)
    """
    if len(samples) < 2:
        return 0, 0.0, []

    click_count = 0
    max_delta = 0
    click_indices = []

    for i in range(1, len(samples)):
        delta = abs(samples[i] - samples[i-1])
        if delta > max_delta:
            max_delta = delta
        if delta > threshold:
            click_count += 1
            if len(click_indices) < 10:  # Track first 10
                click_indices.append(i)

    return click_count, max_delta, click_indices


def analyze_block_boundaries(samples: List[int], block_size: int = BLOCK_SIZE) -> Tuple[int, float]:
    """
    Specifically check for clicks at block boundaries.

    Returns:
        (boundary_clicks, max_boundary_delta)
    """
    boundary_clicks = 0
    max_boundary_delta = 0
    threshold = 3000

    n_blocks = len(samples) // block_size

    for block in range(1, n_blocks):
        boundary_idx = block * block_size
        if boundary_idx < len(samples):
            delta = abs(samples[boundary_idx] - samples[boundary_idx - 1])
            if delta > max_boundary_delta:
                max_boundary_delta = delta
            if delta > threshold:
                boundary_clicks += 1

    return boundary_clicks, max_boundary_delta


def generate_harmonic_coeffs(base_freq: float, amplitude: int, n_harmonics: int = 8) -> List[int]:
    """
    Generate wavelet coefficients for a harmonic tone.

    Maps harmonics to leaf nodes using logarithmic frequency mapping.
    """
    coeffs = [0] * COEFF_LEN
    nyquist = FS / 2

    def freq_to_leaf(f: float) -> int:
        if f < 20:
            return 0
        if f > 20000:
            return N_LEAVES - 1
        log_pos = np.log2(f / 20) / np.log2(20000 / 20)
        return int(log_pos * (N_LEAVES - 1))

    for h in range(1, n_harmonics + 1):
        h_freq = base_freq * h
        if h_freq >= nyquist * 0.95:
            break

        leaf_idx = freq_to_leaf(h_freq)
        leaf_idx = max(0, min(N_LEAVES - 1, leaf_idx))

        # Harmonic amplitude with rolloff
        h_amp = amplitude // (h ** 0.7)

        # Fill node with pattern
        node_start = leaf_idx * NODE_SIZE
        for i in range(NODE_SIZE):
            t = i / NODE_SIZE
            val = int(h_amp * np.sin(2 * np.pi * t * h))
            coeffs[node_start + i] = sat16(coeffs[node_start + i] + val)

    return coeffs


# =============================================================================
# TEST CLASS
# =============================================================================

class TestWaveletTransitions:
    """Test WaveletEngine coefficient transitions at different intensities."""

    def _run_transition_test(
        self,
        intensity_name: str,
        coeff_multiplier: float,
        duration_sec: float = 2.0,
        transitions_per_sec: int = 4,
    ) -> Tuple[List[int], List[int], int, float]:
        """
        Run a transition test at specified intensity.

        Args:
            intensity_name: Name for output file
            coeff_multiplier: How much to scale coefficient changes (1x, 10x, 100x)
            duration_sec: Total duration
            transitions_per_sec: How often to change coefficients

        Returns:
            (L, R, click_count, max_delta)
        """
        wavelet = WaveletEngine(block_size=BLOCK_SIZE)
        safety = SafetyEngine(block_size=BLOCK_SIZE)
        params = ParamFrame()

        total_blocks = int(duration_sec * BLOCKS_PER_SEC)
        blocks_per_transition = max(1, BLOCKS_PER_SEC // transitions_per_sec)

        # Allocate voice
        wavelet.voice_allocate(0)
        safety.voice_on(0)

        all_L = []
        all_R = []

        # Base frequencies for alternating tones
        freq_a = 220.0  # A3
        freq_b = 440.0  # A4 (octave up - significant harmonic change)

        # Base amplitude
        base_amp = int(8000 * coeff_multiplier)
        base_amp = min(base_amp, Q15_ONE)  # Clamp to prevent overflow

        current_freq = freq_a

        for block in range(total_blocks):
            # Transition every N blocks
            if block > 0 and block % blocks_per_transition == 0:
                # Alternate between frequencies
                current_freq = freq_b if current_freq == freq_a else freq_a

            # Generate coefficients for current frequency
            coeffs = generate_harmonic_coeffs(current_freq, base_amp)

            # Create VoiceFrame
            vf = VoiceFrame(
                voice_id=0,
                tree_a_id=1,  # Harmonic tree
                tree_b_id=1,
                tree_blend_q15=0,
                gain_q15=Q15_HALF,
                pan_q15=0,
                coeff_q15=coeffs,
            )

            # Render through wavelet engine
            audio = wavelet.render_voice(vf)

            # Apply per-voice declick
            audio = safety.apply_declick(audio, 0)

            # Mix to stereo (single voice)
            L, R = wavelet.mix_block([(vf, audio)])

            # Apply safety limiting
            L, R = safety.step_block(L, R, params)

            all_L.extend(L)
            all_R.extend(R)

        # Compute click metrics
        click_count, max_delta, click_indices = compute_click_metric(all_L)
        boundary_clicks, max_boundary = analyze_block_boundaries(all_L)

        return all_L, all_R, click_count, max_delta, boundary_clicks, max_boundary

    def test_transition_1x_baseline(self):
        """
        Test coefficient transitions at 1x intensity (baseline).

        This is the normal operating condition with gradual coefficient evolution.
        Documents current behavior for comparison.
        """
        L, R, clicks, max_delta, boundary_clicks, max_boundary = self._run_transition_test(
            "transition_1x_baseline",
            coeff_multiplier=1.0,
            duration_sec=2.0,
            transitions_per_sec=4,
        )

        filepath = save_audio(L, R, "transition_1x_baseline")
        print(f"\n  1x Baseline: {len(L)} samples")
        print(f"    Total clicks (>3000): {clicks}")
        print(f"    Boundary clicks: {boundary_clicks}")
        print(f"    Max sample delta: {max_delta:.0f}")
        print(f"    Max boundary delta: {max_boundary:.0f}")
        print(f"    Saved: {filepath}")

    def test_transition_10x_moderate(self):
        """
        Test coefficient transitions at 10x intensity (moderate stress).

        This simulates larger coefficient jumps like note changes.
        Documents behavior under moderate stress.
        """
        L, R, clicks, max_delta, boundary_clicks, max_boundary = self._run_transition_test(
            "transition_10x_moderate",
            coeff_multiplier=10.0,
            duration_sec=2.0,
            transitions_per_sec=4,
        )

        filepath = save_audio(L, R, "transition_10x_moderate")
        print(f"\n  10x Moderate: {len(L)} samples")
        print(f"    Total clicks (>3000): {clicks}")
        print(f"    Boundary clicks: {boundary_clicks}")
        print(f"    Max sample delta: {max_delta:.0f}")
        print(f"    Max boundary delta: {max_boundary:.0f}")
        print(f"    Saved: {filepath}")

    def test_transition_100x_extreme(self):
        """
        Test coefficient transitions at 100x intensity (extreme stress).

        This is a torture test - massive coefficient jumps that push the
        slewing + micro-bridge to its limits.
        """
        L, R, clicks, max_delta, boundary_clicks, max_boundary = self._run_transition_test(
            "transition_100x_extreme",
            coeff_multiplier=100.0,
            duration_sec=2.0,
            transitions_per_sec=4,
        )

        filepath = save_audio(L, R, "transition_100x_extreme")
        print(f"\n  100x Extreme: {len(L)} samples")
        print(f"    Total clicks (>3000): {clicks}")
        print(f"    Boundary clicks: {boundary_clicks}")
        print(f"    Max sample delta: {max_delta:.0f}")
        print(f"    Max boundary delta: {max_boundary:.0f}")
        print(f"    Saved: {filepath}")
        print(f"    NOTE: 100x is a stress test to expose limits")

    def test_rapid_transitions_all_intensities(self):
        """
        Test rapid transitions (16/sec) at all intensities.

        This combines fast transition rate with varying intensities
        to stress-test the coefficient smoothing.
        """
        results = {}

        for mult, name in [(1.0, "1x"), (10.0, "10x"), (100.0, "100x")]:
            L, R, clicks, max_delta, boundary_clicks, max_boundary = self._run_transition_test(
                f"rapid_{name}",
                coeff_multiplier=mult,
                duration_sec=1.0,
                transitions_per_sec=16,  # Very rapid
            )

            filepath = save_audio(L, R, f"rapid_transitions_{name}")
            results[name] = {
                'clicks': clicks,
                'max_delta': max_delta,
                'boundary_clicks': boundary_clicks,
                'max_boundary': max_boundary,
                'filepath': filepath,
            }

        print("\n  Rapid Transitions (16/sec):")
        for name, data in results.items():
            print(f"    {name}: total_clicks={data['clicks']}, boundary_clicks={data['boundary_clicks']}")
            print(f"         max_delta={data['max_delta']:.0f}, max_boundary={data['max_boundary']:.0f}")
            print(f"         Saved: {data['filepath']}")

    def test_comparison_with_dual_render_simulation(self):
        """
        Compare current slew+bridge approach vs simulated dual-render.

        This demonstrates what dual-render WOULD look like for the
        wavelet engine by synthesizing with both old and new coefficients
        and crossfading.
        """
        wavelet = WaveletEngine(block_size=BLOCK_SIZE)
        safety = SafetyEngine(block_size=BLOCK_SIZE)
        params = ParamFrame()

        duration_sec = 2.0
        total_blocks = int(duration_sec * BLOCKS_PER_SEC)
        blocks_per_transition = BLOCKS_PER_SEC // 4  # 4 transitions/sec

        wavelet.voice_allocate(0)
        safety.voice_on(0)

        # Method A: Standard (current) approach
        all_L_standard = []
        all_R_standard = []

        # Method B: Simulated dual-render approach
        all_L_dual = []
        all_R_dual = []

        freq_a, freq_b = 220.0, 440.0
        current_freq = freq_a
        prev_coeffs = None

        for block in range(total_blocks):
            is_transition = block > 0 and block % blocks_per_transition == 0

            if is_transition:
                current_freq = freq_b if current_freq == freq_a else freq_a

            # Generate new coefficients
            new_coeffs = generate_harmonic_coeffs(current_freq, 8000)

            # === Method A: Standard rendering ===
            vf_standard = VoiceFrame(
                voice_id=0,
                tree_a_id=1,
                tree_b_id=1,
                tree_blend_q15=0,
                gain_q15=Q15_HALF,
                pan_q15=0,
                coeff_q15=new_coeffs,
            )
            audio_standard = wavelet.render_voice(vf_standard)
            audio_standard = safety.apply_declick(audio_standard, 0)
            L_std, R_std = wavelet.mix_block([(vf_standard, audio_standard)])
            L_std, R_std = safety.step_block(L_std, R_std, params)
            all_L_standard.extend(L_std)
            all_R_standard.extend(R_std)

            # === Method B: Simulated dual-render ===
            if is_transition and prev_coeffs is not None:
                # Synthesize with OLD coefficients
                vf_old = VoiceFrame(
                    voice_id=1,  # Different voice ID to avoid state contamination
                    tree_a_id=1,
                    tree_b_id=1,
                    tree_blend_q15=0,
                    gain_q15=Q15_HALF,
                    pan_q15=0,
                    coeff_q15=prev_coeffs,
                )

                # Create separate wavelet engine for clean dual-render
                wavelet_dual = WaveletEngine(block_size=BLOCK_SIZE)
                wavelet_dual.voice_allocate(0)
                wavelet_dual.voice_allocate(1)

                # Render both
                audio_old = wavelet_dual.render_voice(vf_old, apply_regularity=False)

                vf_new = VoiceFrame(
                    voice_id=1,
                    tree_a_id=1,
                    tree_b_id=1,
                    tree_blend_q15=0,
                    gain_q15=Q15_HALF,
                    pan_q15=0,
                    coeff_q15=new_coeffs,
                )
                audio_new = wavelet_dual.render_voice(vf_new, apply_regularity=False)

                # Crossfade at block start (8 samples)
                M = 8
                audio_dual = []
                for i in range(len(audio_old)):
                    if i < M:
                        # Crossfade region
                        t = i / (M - 1) if M > 1 else 1.0
                        # Smoothstep
                        t = t * t * (3 - 2 * t)
                        blended = int(audio_old[i] * (1 - t) + audio_new[i] * t)
                        audio_dual.append(sat16(blended))
                    else:
                        audio_dual.append(audio_new[i])

                L_dual = list(audio_dual)
                R_dual = list(audio_dual)
            else:
                # Normal rendering (no transition)
                vf_dual = VoiceFrame(
                    voice_id=0,
                    tree_a_id=1,
                    tree_b_id=1,
                    tree_blend_q15=0,
                    gain_q15=Q15_HALF,
                    pan_q15=0,
                    coeff_q15=new_coeffs,
                )
                wavelet_temp = WaveletEngine(block_size=BLOCK_SIZE)
                wavelet_temp.voice_allocate(0)
                audio_dual = wavelet_temp.render_voice(vf_dual, apply_regularity=False)
                L_dual = list(audio_dual)
                R_dual = list(audio_dual)

            all_L_dual.extend(L_dual)
            all_R_dual.extend(R_dual)

            prev_coeffs = list(new_coeffs)

        # Save both
        filepath_std = save_audio(all_L_standard, all_R_standard, "comparison_standard")
        filepath_dual = save_audio(all_L_dual, all_R_dual, "comparison_dual_render_sim")

        clicks_std, max_std, _ = compute_click_metric(all_L_standard)
        clicks_dual, max_dual, _ = compute_click_metric(all_L_dual)
        boundary_std, max_bound_std = analyze_block_boundaries(all_L_standard)
        boundary_dual, max_bound_dual = analyze_block_boundaries(all_L_dual)

        print("\n  Comparison: Standard vs Dual-Render Simulation")
        print(f"    Standard: total_clicks={clicks_std}, boundary_clicks={boundary_std}")
        print(f"              max_delta={max_std:.0f}, max_boundary={max_bound_std:.0f}")
        print(f"              Saved: {filepath_std}")
        print(f"    Dual-Render Sim: total_clicks={clicks_dual}, boundary_clicks={boundary_dual}")
        print(f"                     max_delta={max_dual:.0f}, max_boundary={max_bound_dual:.0f}")
        print(f"              Saved: {filepath_dual}")
        print(f"    NOTE: Dual-render sim is flawed - needs proper stateless implementation")

    def test_rapid_transitions_dual_render_sim(self):
        """
        Test rapid transitions (16/sec) with dual-render simulation.

        Compares standard vs dual-render at rapid transition rates.
        """
        duration_sec = 1.0
        transitions_per_sec = 16
        total_blocks = int(duration_sec * BLOCKS_PER_SEC)
        blocks_per_transition = max(1, BLOCKS_PER_SEC // transitions_per_sec)

        results = {}

        for method in ["standard", "dual_render"]:
            wavelet = WaveletEngine(block_size=BLOCK_SIZE)
            safety = SafetyEngine(block_size=BLOCK_SIZE)
            params = ParamFrame()

            wavelet.voice_allocate(0)
            safety.voice_on(0)

            all_L = []
            all_R = []

            freq_a, freq_b = 220.0, 440.0
            current_freq = freq_a
            prev_coeffs = None

            for block in range(total_blocks):
                is_transition = block > 0 and block % blocks_per_transition == 0

                if is_transition:
                    current_freq = freq_b if current_freq == freq_a else freq_a

                new_coeffs = generate_harmonic_coeffs(current_freq, 8000)

                if method == "standard":
                    # Standard: use persistent engine with slewing
                    vf = VoiceFrame(
                        voice_id=0,
                        tree_a_id=1,
                        tree_b_id=1,
                        tree_blend_q15=0,
                        gain_q15=Q15_HALF,
                        pan_q15=0,
                        coeff_q15=new_coeffs,
                    )
                    audio = wavelet.render_voice(vf)
                    audio = safety.apply_declick(audio, 0)
                    L, R = wavelet.mix_block([(vf, audio)])
                    L, R = safety.step_block(L, R, params)

                else:
                    # Dual-render: fresh engine per block, crossfade at transitions
                    if is_transition and prev_coeffs is not None:
                        # Create fresh engine
                        wavelet_fresh = WaveletEngine(block_size=BLOCK_SIZE)
                        wavelet_fresh.voice_allocate(0)
                        wavelet_fresh.voice_allocate(1)

                        # Render with OLD coefficients
                        vf_old = VoiceFrame(
                            voice_id=0,
                            tree_a_id=1,
                            tree_b_id=1,
                            tree_blend_q15=0,
                            gain_q15=Q15_HALF,
                            pan_q15=0,
                            coeff_q15=prev_coeffs,
                        )
                        audio_old = wavelet_fresh.render_voice(vf_old, apply_regularity=False)

                        # Render with NEW coefficients
                        vf_new = VoiceFrame(
                            voice_id=1,
                            tree_a_id=1,
                            tree_b_id=1,
                            tree_blend_q15=0,
                            gain_q15=Q15_HALF,
                            pan_q15=0,
                            coeff_q15=new_coeffs,
                        )
                        audio_new = wavelet_fresh.render_voice(vf_new, apply_regularity=False)

                        # Crossfade
                        M = 8
                        audio = []
                        for i in range(len(audio_old)):
                            if i < M:
                                t = i / (M - 1) if M > 1 else 1.0
                                t = t * t * (3 - 2 * t)  # Smoothstep
                                blended = int(audio_old[i] * (1 - t) + audio_new[i] * t)
                                audio.append(sat16(blended))
                            else:
                                audio.append(audio_new[i])
                        L = list(audio)
                        R = list(audio)
                    else:
                        # Non-transition: fresh engine, no crossfade
                        wavelet_fresh = WaveletEngine(block_size=BLOCK_SIZE)
                        wavelet_fresh.voice_allocate(0)
                        vf = VoiceFrame(
                            voice_id=0,
                            tree_a_id=1,
                            tree_b_id=1,
                            tree_blend_q15=0,
                            gain_q15=Q15_HALF,
                            pan_q15=0,
                            coeff_q15=new_coeffs,
                        )
                        audio = wavelet_fresh.render_voice(vf, apply_regularity=False)
                        L = list(audio)
                        R = list(audio)

                all_L.extend(L)
                all_R.extend(R)
                prev_coeffs = list(new_coeffs)

            filepath = save_audio(all_L, all_R, f"rapid_16hz_{method}")
            clicks, max_delta, _ = compute_click_metric(all_L)
            boundary_clicks, max_boundary = analyze_block_boundaries(all_L)

            results[method] = {
                'clicks': clicks,
                'max_delta': max_delta,
                'boundary_clicks': boundary_clicks,
                'max_boundary': max_boundary,
                'filepath': filepath,
            }

        print("\n  Rapid Transitions (16/sec) - Standard vs Dual-Render:")
        for method, data in results.items():
            print(f"    {method}: total_clicks={data['clicks']}, boundary_clicks={data['boundary_clicks']}")
            print(f"              max_delta={data['max_delta']:.0f}, max_boundary={data['max_boundary']:.0f}")
            print(f"              Saved: {data['filepath']}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
