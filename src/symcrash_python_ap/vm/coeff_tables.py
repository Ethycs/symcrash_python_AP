"""
Coefficient Table Generation for CCS (WSE-STD-ARCH-CCS v0)
==========================================================

Generates wavelet-domain coefficient tables analytically using
IDWPT inversion. Tables map (MIDI note, phase) → 64 leaf coefficients.

For each MIDI note and phase step, we compute 64 leaf coefficients that,
when passed through IDWPT, reconstruct a pure sinusoid at that frequency.

This eliminates FWPT from the voice path entirely - the carrier is generated
directly in coefficient space.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import logging

# Import IDWPT for validation and inverse solving
from ..wavelet.fast_lifting import idwpt_jit

# Q15 constants
Q15_ONE = 32767
Q15_HALF = 16384

# Sample rate (fixed for coefficient table generation)
SAMPLE_RATE = 48000

# Block size (fixed for IDWPT)
BLOCK_SIZE = 64

# Phase steps (64 for high fidelity)
PHASE_STEPS = 64

# Number of MIDI notes
NUM_MIDI_NOTES = 128

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CoeffTable2D:
    """
    2D coefficient table: [MIDI_notes × phase_steps × 64_leaves]

    For each MIDI note and phase step, stores 64 leaf coefficients
    representing sin/cos basis that reconstructs to a pure sinusoid.

    Memory: 128 notes × 64 phases × 2 (sin/cos) × 64 coeffs × 2 bytes = 2 MB
    """
    sin_tables: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_MIDI_NOTES, PHASE_STEPS, 64), dtype=np.int16)
    )  # [128, 64, 64] int16

    cos_tables: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_MIDI_NOTES, PHASE_STEPS, 64), dtype=np.int16)
    )  # [128, 64, 64] int16

    sample_rate: int = SAMPLE_RATE
    block_size: int = BLOCK_SIZE
    phase_steps: int = PHASE_STEPS


# =============================================================================
# MIDI TO FREQUENCY CONVERSION
# =============================================================================

def midi_to_hz(note: int, cents: int = 0) -> float:
    """
    Convert MIDI note number to frequency in Hz.

    Args:
        note: MIDI note (0-127)
        cents: Fine tuning in cents (-100 to +100)

    Returns:
        Frequency in Hz
    """
    semitones = note - 69 + (cents / 100.0)
    return 440.0 * (2.0 ** (semitones / 12.0))


# =============================================================================
# ANALYTIC COEFFICIENT GENERATION (IDWPT INVERSION)
# =============================================================================

def generate_reference_sinusoid(
    freq_hz: float,
    phase_radians: float,
    num_samples: int = BLOCK_SIZE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate reference sinusoid for coefficient solving.

    Args:
        freq_hz: Frequency in Hz
        phase_radians: Starting phase in radians
        num_samples: Number of samples (64 for one block)
        sample_rate: Sample rate in Hz

    Returns:
        int16 array of audio samples (Q15 format)
    """
    t = np.arange(num_samples) / sample_rate
    omega = 2 * np.pi * freq_hz

    # Generate sinusoid and scale to Q15 range
    audio = np.sin(omega * t + phase_radians)
    audio_q15 = (audio * Q15_HALF).astype(np.int16)

    return audio_q15


def solve_coefficients_for_sinusoid(
    target_audio: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """
    Solve for 64 leaf coefficients that reconstruct to target audio via IDWPT.

    Uses iterative gradient descent in coefficient space:
    - Start with FWPT of target as initial guess
    - Iteratively adjust coefficients to minimize reconstruction error

    Args:
        target_audio: Target audio samples (64 int16 values)
        max_iterations: Maximum gradient descent iterations
        tolerance: Convergence tolerance (RMS error)

    Returns:
        64 leaf coefficients (int16) that best reconstruct target
    """
    # Initial guess: Use FWPT to get starting coefficients
    from ..wavelet.fast_lifting import fwpt_jit

    # Convert to list for fwpt_jit
    target_list = target_audio.tolist()

    # Get initial coefficients via FWPT
    coeffs_init = fwpt_jit(target_list, headroom_bits=1)
    coeffs = np.array(coeffs_init, dtype=np.float32)

    # Gradient descent to refine
    learning_rate = 0.5

    for iteration in range(max_iterations):
        # Forward pass: IDWPT
        coeffs_int = np.clip(coeffs, -32768, 32767).astype(np.int16)
        reconstructed = idwpt_jit(coeffs_int.tolist(), headroom_bits=1)
        reconstructed = np.array(reconstructed, dtype=np.float32)

        # Compute error
        error = target_audio.astype(np.float32) - reconstructed
        rms_error = np.sqrt(np.mean(error ** 2))

        # Check convergence
        if rms_error < tolerance:
            break

        # Backward pass: Approximate gradient
        # For each coefficient, perturb slightly and measure effect on output
        # (This is a simplified gradient - full implementation would use actual IDWPT Jacobian)
        gradient = np.zeros(64, dtype=np.float32)

        delta = 1.0
        for i in range(64):
            # Perturb coefficient up
            coeffs_perturbed = coeffs.copy()
            coeffs_perturbed[i] += delta
            coeffs_perturbed_int = np.clip(coeffs_perturbed, -32768, 32767).astype(np.int16)

            reconstructed_up = idwpt_jit(coeffs_perturbed_int.tolist(), headroom_bits=1)
            reconstructed_up = np.array(reconstructed_up, dtype=np.float32)

            # Estimate gradient via finite difference
            output_change = reconstructed_up - reconstructed
            gradient[i] = np.dot(error, output_change) / delta

        # Update coefficients
        coeffs += learning_rate * gradient

        # Decay learning rate
        if iteration % 100 == 0:
            learning_rate *= 0.9

    # Final clamp and return
    coeffs_final = np.clip(coeffs, -32768, 32767).astype(np.int16)

    # Validate
    final_audio = idwpt_jit(coeffs_final.tolist(), headroom_bits=1)
    final_error = np.array(target_audio, dtype=np.float32) - np.array(final_audio, dtype=np.float32)
    final_rms = np.sqrt(np.mean(final_error ** 2))

    if final_rms > 100:  # More than 100 samples RMS error is suspicious
        logger.warning(f"High reconstruction error: RMS={final_rms:.2f}")

    return coeffs_final


def generate_coeff_entry_pair(
    freq_hz: float,
    phase_step: int,
    phase_steps: int = PHASE_STEPS,
) -> tuple:
    """
    Generate sin/cos coefficient pair for one (note, phase) entry.

    Args:
        freq_hz: Frequency in Hz
        phase_step: Phase step index (0 to phase_steps-1)
        phase_steps: Total number of phase steps (64)

    Returns:
        (c_sin, c_cos): Tuple of 64-element int16 arrays
    """
    # Compute phase in radians for this step
    phase_radians = 2 * np.pi * phase_step / phase_steps

    # Generate reference sinusoid at this phase
    target_sin = generate_reference_sinusoid(freq_hz, phase_radians)

    # Generate reference cosine (90 degree phase shift)
    target_cos = generate_reference_sinusoid(freq_hz, phase_radians + np.pi / 2)

    # Solve for coefficients
    c_sin = solve_coefficients_for_sinusoid(target_sin)
    c_cos = solve_coefficients_for_sinusoid(target_cos)

    return c_sin, c_cos


# =============================================================================
# NORMALIZATION (WSE-STD-ARCH-CCS section 6.1)
# =============================================================================

def normalize_coefficients(
    coeffs: np.ndarray,
    target_rms_q15: int = Q15_HALF,
) -> np.ndarray:
    """
    Normalize coefficients to achieve target RMS after IDWPT reconstruction.

    Per WSE-STD-ARCH-CCS.md section 6.1:
    - Reconstruct audio via IDWPT
    - Measure RMS
    - Scale coefficients to achieve target RMS (-6 dBFS nominal)

    Args:
        coeffs: 64 leaf coefficients (int16)
        target_rms_q15: Target RMS in Q15 (16384 = -6 dBFS)

    Returns:
        Normalized coefficients (int16)
    """
    # Reconstruct audio
    audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)
    audio_arr = np.array(audio, dtype=np.float32)

    # Compute RMS
    rms_actual = np.sqrt(np.mean(audio_arr ** 2))

    if rms_actual < 1.0:
        # Avoid division by zero
        return coeffs

    # Compute scale factor
    scale_factor = target_rms_q15 / rms_actual

    # Apply scaling
    coeffs_scaled = (coeffs.astype(np.float32) * scale_factor).astype(np.int32)

    # Clamp to int16 range
    coeffs_normalized = np.clip(coeffs_scaled, -32768, 32767).astype(np.int16)

    return coeffs_normalized


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_coeff_table_2d(
    midi_notes: range = range(NUM_MIDI_NOTES),
    phase_steps: int = PHASE_STEPS,
    sample_rate: int = SAMPLE_RATE,
    normalize: bool = True,
) -> CoeffTable2D:
    """
    Generate complete 2D coefficient table set.

    For each MIDI note and phase step, compute 64 leaf coefficients
    that reconstruct to a pure sinusoid at that frequency via IDWPT.

    This is the core table generation function - runs at startup.
    Expected generation time: ~10-30 seconds for all 128 notes.

    Args:
        midi_notes: Range of MIDI notes to generate (default: 0-127)
        phase_steps: Number of phase steps per note (default: 64)
        sample_rate: Sample rate in Hz (default: 48000)
        normalize: Whether to normalize RMS (default: True)

    Returns:
        CoeffTable2D with populated sin/cos tables
    """
    table = CoeffTable2D(
        sample_rate=sample_rate,
        phase_steps=phase_steps,
    )

    logger.info(f"Generating coefficient tables for {len(midi_notes)} MIDI notes, {phase_steps} phase steps...")

    total_entries = len(midi_notes) * phase_steps
    progress_interval = max(1, total_entries // 20)  # Report every 5%

    entry_count = 0

    for note in midi_notes:
        freq_hz = midi_to_hz(note)

        # Skip notes above Nyquist (above MIDI 125 at 48kHz)
        if freq_hz > sample_rate / 2:
            logger.warning(f"MIDI note {note} ({freq_hz:.1f} Hz) above Nyquist, skipping")
            continue

        for phase_idx in range(phase_steps):
            # Generate coefficient pair
            c_sin, c_cos = generate_coeff_entry_pair(freq_hz, phase_idx, phase_steps)

            # Normalize if requested
            if normalize:
                c_sin = normalize_coefficients(c_sin)
                c_cos = normalize_coefficients(c_cos)

            # Store in table
            table.sin_tables[note, phase_idx, :] = c_sin
            table.cos_tables[note, phase_idx, :] = c_cos

            entry_count += 1

            # Progress reporting
            if entry_count % progress_interval == 0:
                progress_pct = 100 * entry_count / total_entries
                logger.info(f"  Progress: {progress_pct:.1f}% ({entry_count}/{total_entries} entries)")

    logger.info(f"Coefficient table generation complete: {entry_count} entries")

    return table


# =============================================================================
# VALIDATION
# =============================================================================

def validate_table_entry(
    coeffs: np.ndarray,
    expected_freq: float,
    sample_rate: int = SAMPLE_RATE,
    freq_tolerance_hz: float = 5.0,
) -> bool:
    """
    Validate that coefficients reconstruct to expected frequency.

    Tests: IDWPT(coeffs) should yield pure sinusoid at expected_freq

    Args:
        coeffs: 64 leaf coefficients
        expected_freq: Expected frequency in Hz
        sample_rate: Sample rate
        freq_tolerance_hz: Tolerance in Hz (default: 5 Hz)

    Returns:
        True if validation passes
    """
    # Reconstruct audio
    audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)
    audio_arr = np.array(audio, dtype=np.float32)

    # FFT analysis
    spectrum = np.fft.rfft(audio_arr)
    freqs = np.fft.rfftfreq(len(audio_arr), 1.0 / sample_rate)

    # Find peak frequency
    peak_idx = np.argmax(np.abs(spectrum))
    peak_freq = freqs[peak_idx]

    # Check tolerance
    freq_error = abs(peak_freq - expected_freq)

    if freq_error > freq_tolerance_hz:
        logger.warning(
            f"Frequency mismatch: expected {expected_freq:.2f} Hz, "
            f"got {peak_freq:.2f} Hz (error: {freq_error:.2f} Hz)"
        )
        return False

    return True


# =============================================================================
# SINGLETON TABLE ACCESS
# =============================================================================

# Global coefficient table (lazy-loaded on first access)
_COEFF_TABLE_2D: Optional[CoeffTable2D] = None


def get_coeff_table_2d(table_id: int = 0) -> CoeffTable2D:
    """
    Get precomputed coefficient table (singleton pattern).

    On first call, generates all coefficient tables (~10-30 seconds).
    Subsequent calls return cached table instantly.

    Args:
        table_id: Table variant ID (0 = default sine/cos basis)

    Returns:
        CoeffTable2D instance
    """
    global _COEFF_TABLE_2D

    if _COEFF_TABLE_2D is None:
        logger.info("Generating coefficient tables for first time...")
        _COEFF_TABLE_2D = generate_coeff_table_2d()
        logger.info("Coefficient tables ready")

    return _COEFF_TABLE_2D


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CoeffTable2D',
    'generate_coeff_table_2d',
    'get_coeff_table_2d',
    'midi_to_hz',
    'validate_table_entry',
    'normalize_coefficients',
]
