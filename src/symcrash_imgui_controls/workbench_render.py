"""
Workbench Render - Character-aware preset rendering for the sound design workbench.

Wraps the existing render pipeline to apply character control overrides
(brightness, density, energy, risk, dream, slip) as audible synthesis
modifications before rendering.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Ensure src/ is importable
_src = str(Path(__file__).parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from symcrash_python_ap.instrument_ir import load_instrument, ControlConfig, extract_partials_q15
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas
from symcrash_python_ap.vm.state import VMState, OSC_MODE_B, OSC_MODE_FM6
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.envelope_targets import EnvelopeTarget

FS = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SEC = FS // BLOCK_SIZE


from symcrash_python_ap.primitives import ms_to_blocks as _ms_to_blocks


# Canonical closed-form ADSR + binding evaluator (single owner in
# envelope_scheduler). BindingEnvelope additionally honors key_scale_rate /
# velocity_sens on bindings, which this module's old private copy ignored.
from symcrash_python_ap.envelope_scheduler import (
    adsr_value_q15,
    BindingEnvelope as _EnvelopeState,
)
from symcrash_python_ap.voice.frame_builder import IRFrameBuilder, NoteClock


def _compute_envelope(block_idx, attack_blocks, decay_blocks, sustain,
                      release_blocks, sustain_blocks):
    """Compute ADSR envelope value (Q15) at a given block."""
    return adsr_value_q15(
        attack_blocks, decay_blocks, int(sustain * 32767), release_blocks,
        block_idx, attack_blocks + sustain_blocks,
    )


# =========================================================================
# CHARACTER -> SONIC MODIFICATIONS
# =========================================================================

def _apply_brightness(harmonic_amps: list, brightness: float) -> list:
    """
    Scale high harmonics based on brightness.

    brightness=0.0: attenuate H4+ by 80%
    brightness=1.0: preserve natural rolloff
    """
    result = list(harmonic_amps)
    for i in range(3, len(result)):
        # Scale factor: at brightness=0 -> 0.2, at brightness=1 -> 1.0
        scale = 0.2 + 0.8 * brightness
        result[i] = int(result[i] * scale)
    return result


def _apply_density(harmonic_amps: list, density: float) -> list:
    """
    Boost low harmonics (H1-H3) based on density.

    density=0.0: natural levels
    density=1.0: H1-H3 boosted up to 1.5x (clamped to Q15)
    """
    result = list(harmonic_amps)
    for i in range(min(3, len(result))):
        boost = 1.0 + 0.5 * density
        result[i] = min(32767, int(result[i] * boost))
    return result


def _apply_energy_to_envelope(env: dict, energy: float) -> dict:
    """
    Modify envelope based on energy.

    High energy = fast attack, high sustain
    Low energy = slow attack, low sustain
    """
    result = dict(env)
    # Attack: energy=1.0 -> 25% of original, energy=0.0 -> 200% of original
    attack_scale = 2.0 - 1.75 * energy
    result['attack_blocks'] = max(1, int(result['attack_blocks'] * attack_scale))

    # Sustain: energy=1.0 -> boost toward 1.0, energy=0.0 -> reduce toward 0.5
    base_sustain = result['sustain']
    result['sustain'] = base_sustain + (1.0 - base_sustain) * 0.3 * energy

    return result


def _apply_risk_to_phases(block_count: int, risk: float) -> list:
    """
    Generate per-block phase noise amounts based on risk.

    Returns list of noise amplitudes (0.0-1.0) per block.
    risk=0.0: no noise. risk=1.0: up to 5% phase noise.
    """
    if risk < 0.01:
        return [0.0] * block_count
    rng = np.random.RandomState(42)
    return (rng.random(block_count) * 0.05 * risk).tolist()


# =========================================================================
# MAIN RENDER FUNCTION
# =========================================================================

def render_with_character(
    preset_path: str,
    note: int = 60,
    duration_sec: float = 1.5,
    brightness: float = 0.5,
    density: float = 0.5,
    energy: float = 0.5,
    risk: float = 0.0,
    dream: float = 0.0,
    slip: float = 0.0,
) -> Tuple[np.ndarray, str]:
    """
    Render a preset with character control overrides.

    Args:
        preset_path: Path to YAML preset file
        note: MIDI note number
        duration_sec: Note duration in seconds
        brightness: Harmonic brightness (0.0-1.0)
        density: Low-harmonic body (0.0-1.0)
        energy: Attack/sustain intensity (0.0-1.0)
        risk: Phase noise amount (0.0-1.0)
        dream: Pitch drift amount (0.0-1.0) [reserved]
        slip: Cross-harmonic diffusion (0.0-1.0) [reserved]

    Returns:
        (samples_int16, preset_name) tuple
    """
    clear_basis_atlas()
    ir = load_instrument(str(preset_path))

    # Shared frame builder (voice/frame_builder.py) owns frame
    # construction; this function only applies the character transforms
    # (energy -> envelope, brightness/density -> harmonics, brightness ->
    # FM6 modulator levels) on top of the extracted parameters.
    builder = IRFrameBuilder(ir, note)

    # --- Apply character modifications ---
    env = {
        'attack_blocks': builder.attack_blocks,
        'decay_blocks': builder.decay_blocks,
        'sustain': builder.sustain_q15 / 32767.0,
        'release_blocks': builder.release_blocks,
    }
    env = _apply_energy_to_envelope(env, energy)
    builder.attack_blocks = env['attack_blocks']
    builder.decay_blocks = env['decay_blocks']
    builder.sustain_q15 = int(env['sustain'] * 32767)
    builder.release_blocks = env['release_blocks']

    harmonic_amps = [32767, 16383, 10922, 8191, 6553, 5461, 4681, 4096]
    if hasattr(ir, 'carrier') and ir.carrier and hasattr(ir.carrier, 'harmonic_amps_q15'):
        harmonic_amps = list(ir.carrier.harmonic_amps_q15)

    # Custom partials (waveform: partials) take priority over the default profile
    partials = extract_partials_q15(ir)
    if partials:
        harmonic_amps = partials

    harmonic_amps = _apply_brightness(harmonic_amps, brightness)
    harmonic_amps = _apply_density(harmonic_amps, density)

    # For FM6 presets, apply brightness to operator levels
    fm6 = builder._fm6
    if fm6 is not None:
        levels = list(fm6['op_levels'])
        for i in range(2, len(levels)):
            scale = 0.3 + 0.7 * brightness
            levels[i] = int(levels[i] * scale)
        builder.set_fm6_op_levels(levels)

    # --- Render ---
    attack_blocks = builder.attack_blocks
    release_blocks = builder.release_blocks

    sustain_blocks = int(duration_sec * BLOCKS_PER_SEC) - attack_blocks
    total_blocks = attack_blocks + sustain_blocks + release_blocks
    sustain_until = attack_blocks + sustain_blocks

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)

    samples = []

    for block_idx in range(total_blocks):
        frame = builder.build(
            NoteClock(block_idx, sustain_until_block=sustain_until),
            amps_override=harmonic_amps,
        )
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())

    name = ir.meta.name or Path(preset_path).stem
    return np.array(samples, dtype=np.int16), name


def normalize_to_rms(samples: np.ndarray, target_rms_dbfs: float = -12.0) -> np.ndarray:
    """
    Normalize audio to target RMS level in dBFS.

    Args:
        samples: Audio samples (int16)
        target_rms_dbfs: Target RMS level (default -12 dBFS)

    Returns:
        Normalized samples (int16)
    """
    samples_f = samples.astype(np.float64)
    rms = np.sqrt(np.mean(samples_f ** 2))
    if rms < 1.0:
        return samples

    target_rms = 32767 * (10 ** (target_rms_dbfs / 20))
    gain = target_rms / rms
    normalized = samples_f * gain

    peak = np.max(np.abs(normalized))
    if peak > 32000:
        normalized = np.clip(normalized, -32000, 32000)

    return normalized.astype(np.int16)
