"""
Classic L0 Synthesizer Presets
==============================

This module provides synthesizer presets as InstrumentIR v0 YAML files.

Preset authoring files are stored in the top-level presets/ directory:
    presets/
        instruments/   - Instrument presets (brass_stab, saw_lead, etc.)
        macros/        - Macro presets (vibrato, legato, etc.)

Usage:
    from symcrash_python_ap.library.preset_data import list_presets, load_preset

    # List available presets
    names = list_presets()  # ['saw_lead_v0', 'square_bass_v0', ...]

    # Load a specific preset
    ir = load_preset('saw_lead_v0')

    # Load all presets
    all_presets = get_all_presets()

Available instrument presets:
    - saw_lead_v0      - Bright, cutting lead with TILT + COMPRESS_K
    - square_bass_v0   - Fat, punchy bass with negative TILT + LOWRANK_MORPH
    - sync_lead_v0     - Hard sync resonance with SHEAF_PROJECT + COMPRESS_K
    - brass_stab_v0    - Classic brass hit with TILT + DIFFUSE_DOWN
    - string_pad_v0    - Lush ensemble with DIFFUSE_UP + INTRA_MIX
    - fm_bell_v0       - Metallic bell with SHEAF_PROJECT + FINE_CLAMP
    - jump_brass_v0    - Van Halen style with TILT + COMPRESS_K + DIFFUSE_DOWN
    - sub_bass_v0      - Deep sub with negative TILT + LOWRANK_MORPH
    - pwm_pad_v0       - Slow pulse width mod with DIFFUSE_UP + INTRA_MIX
    - pluck_v0         - Quick attack pluck with TILT + COMPRESS_K
    - wobble_bass_v0   - Dubstep wobble with TILT + LOWRANK_MORPH
    - theremin_lead_v0 - Ethereal theremin with IDENTITY + FINE_CLAMP
    - glitch_lead_v0   - Experimental glitchy lead
    - violin_ir_v1     - Violin with vibrato

Available macro presets:
    - vibrato_classic_v0  - Classic vibrato modulation
    - legato_smooth_v0    - Smooth legato transitions
    - tremolo_fast_v0     - Fast tremolo effect
    - portamento_slow_v0  - Slow pitch glide
"""
from typing import Dict, List

from ...instrument_ir import InstrumentIR, load_instrument
from ...paths import (
    PRESETS_DIR as PRESET_DIR,
    PRESET_INSTRUMENTS_DIR as INSTRUMENTS_DIR,
    PRESET_MACROS_DIR as MACROS_DIR,
)


def list_presets(category: str = None) -> List[str]:
    """
    List all available preset names.

    Args:
        category: Optional filter - 'instruments', 'macros', or None for all.

    Returns:
        List of preset names (without .yaml extension), sorted alphabetically.
    """
    presets = []

    if category is None or category == 'instruments':
        if INSTRUMENTS_DIR.exists():
            presets.extend(f.stem for f in INSTRUMENTS_DIR.glob("*.yaml"))

    if category is None or category == 'macros':
        if MACROS_DIR.exists():
            presets.extend(f.stem for f in MACROS_DIR.glob("*.yaml"))

    return sorted(presets)


def load_preset(name: str) -> InstrumentIR:
    """
    Load a preset by name.

    Args:
        name: Preset name (without .yaml extension), e.g. 'saw_lead_v0'

    Returns:
        InstrumentIR object with the preset configuration.

    Raises:
        ValueError: If preset not found.
    """
    # Check instruments first, then macros
    path = INSTRUMENTS_DIR / f"{name}.yaml"
    if not path.exists():
        path = MACROS_DIR / f"{name}.yaml"

    if not path.exists():
        available = list_presets()
        raise ValueError(
            f"Preset not found: {name}. "
            f"Available presets: {', '.join(available)}"
        )
    return load_instrument(str(path))


def get_all_presets() -> Dict[str, InstrumentIR]:
    """
    Load all presets as a dictionary.

    Returns:
        Dictionary mapping preset name to InstrumentIR object.
    """
    return {name: load_preset(name) for name in list_presets()}


def get_preset_info(name: str) -> Dict[str, any]:
    """
    Get summary info about a preset without loading the full IR.

    Args:
        name: Preset name (without .yaml extension)

    Returns:
        Dictionary with preset metadata.
    """
    ir = load_preset(name)
    stable = ir.layers.L3_cmmm.emit_templates.get('Stable')

    return {
        'name': ir.meta.name,
        'instrument_id': ir.meta.instrument_id,
        'author': ir.meta.author,
        'ops_enable': stable.ops_enable if stable else [],
        'families': stable.families if stable else {},
        'event_count': len(stable.events) if stable else 0,
    }


def get_preset_path(name: str) -> Path:
    """
    Get the file path for a preset.

    Args:
        name: Preset name (without .yaml extension)

    Returns:
        Path to the preset YAML file.

    Raises:
        ValueError: If preset not found.
    """
    path = INSTRUMENTS_DIR / f"{name}.yaml"
    if not path.exists():
        path = MACROS_DIR / f"{name}.yaml"

    if not path.exists():
        available = list_presets()
        raise ValueError(
            f"Preset not found: {name}. "
            f"Available presets: {', '.join(available)}"
        )
    return path


__all__ = [
    'list_presets',
    'load_preset',
    'get_all_presets',
    'get_preset_info',
    'get_preset_path',
    'PRESET_DIR',
    'INSTRUMENTS_DIR',
    'MACROS_DIR',
]
