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


def _ms_to_blocks(ms: float) -> int:
    return max(1, int(ms / 1000.0 * BLOCKS_PER_SEC))


def _compute_envelope(block_idx, attack_blocks, decay_blocks, sustain,
                      release_blocks, sustain_blocks):
    """Compute ADSR envelope value (Q15) at a given block."""
    sustain_q15 = int(sustain * 32767)
    attack_q15 = 32767

    if block_idx < attack_blocks:
        t = block_idx / attack_blocks if attack_blocks > 0 else 1
        return int(attack_q15 * t)
    elif block_idx < attack_blocks + decay_blocks:
        t = (block_idx - attack_blocks) / decay_blocks if decay_blocks > 0 else 1
        return int(attack_q15 - (attack_q15 - sustain_q15) * t)
    elif block_idx < attack_blocks + sustain_blocks:
        return sustain_q15
    else:
        release_idx = block_idx - (attack_blocks + sustain_blocks)
        t = release_idx / release_blocks if release_blocks > 0 else 1
        return int(sustain_q15 * (1.0 - t))


class _EnvelopeState:
    """Tracks envelope state for a single binding."""

    def __init__(self, binding):
        adsr = binding.adsr
        self.attack_blocks = _ms_to_blocks(adsr.a_ms)
        self.decay_blocks = _ms_to_blocks(adsr.d_ms)
        self.sustain = adsr.s
        self.release_blocks = _ms_to_blocks(adsr.r_ms)
        self.target_ids = []
        for name in binding.targets:
            try:
                self.target_ids.append(int(EnvelopeTarget[name]))
            except KeyError:
                pass

    def get_value(self, block_idx, sustain_blocks):
        value = _compute_envelope(
            block_idx, self.attack_blocks, self.decay_blocks,
            self.sustain, self.release_blocks, sustain_blocks
        )
        return max(0, min(32767, value))


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

    # Extract base envelope and synthesis params from L0 graph
    env = {'attack_blocks': 7, 'decay_blocks': 75, 'sustain': 0.7, 'release_blocks': 150}
    waveform = 1  # saw default
    use_fm6 = False
    fm6_params = None

    for node in ir.layers.L0_graph.nodes:
        if node.type == "Amp" and node.params.adsr:
            adsr = node.params.adsr
            env = {
                'attack_blocks': _ms_to_blocks(adsr.a_ms),
                'decay_blocks': _ms_to_blocks(adsr.d_ms),
                'sustain': adsr.s,
                'release_blocks': _ms_to_blocks(adsr.r_ms),
            }
        if node.type == "Osc" and node.params.waveform:
            wf = node.params.waveform.lower()
            waveform = {
                'sine': 0, 'square': 1, 'pulse': 2,
                'saw': 3, 'tri': 5, 'triangle': 5,
            }.get(wf, 0)
        if node.type == "FM6":
            use_fm6 = True
            fm6_params = {
                'algorithm': getattr(node.params, 'algorithm', 1),
                'feedback': getattr(node.params, 'feedback', 0),
                'op_ratios': getattr(node.params, 'op_ratios', [65536] * 6),
                'op_levels': getattr(node.params, 'op_levels', [32767, 0, 0, 0, 0, 0]),
            }

    # Extract envelope bindings
    envelope_states = []
    if hasattr(ir, 'envelopes') and ir.envelopes:
        for binding in ir.envelopes:
            envelope_states.append(_EnvelopeState(binding))

    # --- Apply character modifications ---
    env = _apply_energy_to_envelope(env, energy)

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
    if use_fm6 and fm6_params:
        levels = list(fm6_params['op_levels'])
        for i in range(2, len(levels)):
            scale = 0.3 + 0.7 * brightness
            levels[i] = int(levels[i] * scale)
        fm6_params['op_levels'] = levels

    phase_noise = _apply_risk_to_phases(
        int(duration_sec * BLOCKS_PER_SEC) + env['release_blocks'] + 100,
        risk,
    )

    # --- Render ---
    attack_blocks = env['attack_blocks']
    decay_blocks = env['decay_blocks']
    sustain = env['sustain']
    release_blocks = env['release_blocks']

    sustain_blocks = int(duration_sec * BLOCKS_PER_SEC) - attack_blocks
    total_blocks = attack_blocks + sustain_blocks + release_blocks

    attack_level_q15 = 32767
    sustain_level_q15 = int(sustain * 32767)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)

    samples = []

    for block_idx in range(total_blocks):
        # Gain envelope
        if block_idx < attack_blocks:
            t = block_idx / attack_blocks if attack_blocks > 0 else 1
            gain_q15 = int(attack_level_q15 * t)
        elif block_idx < attack_blocks + decay_blocks:
            t = (block_idx - attack_blocks) / decay_blocks if decay_blocks > 0 else 1
            gain_q15 = int(attack_level_q15 - (attack_level_q15 - sustain_level_q15) * t)
        elif block_idx < attack_blocks + sustain_blocks:
            gain_q15 = sustain_level_q15
        else:
            release_idx = block_idx - (attack_blocks + sustain_blocks)
            t = release_idx / release_blocks if release_blocks > 0 else 1
            gain_q15 = int(sustain_level_q15 * (1.0 - t))

        gain_q15 = max(0, min(32767, gain_q15))

        # Matrix envelope targets
        env_targets_q15 = None
        if envelope_states:
            env_targets_q15 = {}
            for env_state in envelope_states:
                value = env_state.get_value(block_idx, sustain_blocks)
                for target_id in env_state.target_ids:
                    env_targets_q15[target_id] = value

        if use_fm6 and fm6_params:
            frame = VoiceFrame(
                note=note,
                coeff_osc_mode=OSC_MODE_FM6,
                gain_q15=gain_q15,
                beta_target_q15=0,
                fm_algorithm_id=fm6_params['algorithm'],
                fm_op_ratios=fm6_params['op_ratios'],
                fm_op_levels=fm6_params['op_levels'],
                fm_feedback_depth=fm6_params['feedback'],
                env_targets_q15=env_targets_q15,
            )
        else:
            frame = VoiceFrame(
                note=note,
                coeff_osc_mode=OSC_MODE_B,
                H_request=len(harmonic_amps),
                gain_q15=gain_q15,
                beta_target_q15=0,
                partial_amps_q15=harmonic_amps,
                waveform_type=waveform,
                env_targets_q15=env_targets_q15,
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
