"""
Classic L0 Synthesizer Presets
==============================

This module provides 12 classic synthesizer presets as InstrumentIR v0 JSON files.

Usage:
    from symcrash_python_ap.library.preset_data import list_presets, load_preset

    # List available presets
    names = list_presets()  # ['saw_lead', 'square_bass', ...]

    # Load a specific preset
    ir = load_preset('saw_lead')

    # Load all presets
    all_presets = get_all_presets()

Available presets:
    1. saw_lead      - Bright, cutting lead with TILT + COMPRESS_K
    2. square_bass   - Fat, punchy bass with negative TILT + LOWRANK_MORPH
    3. sync_lead     - Hard sync resonance with SHEAF_PROJECT + COMPRESS_K
    4. brass_stab    - Classic brass hit with TILT + DIFFUSE_DOWN
    5. string_pad    - Lush ensemble with DIFFUSE_UP + INTRA_MIX
    6. fm_bell       - Metallic bell with SHEAF_PROJECT + FINE_CLAMP
    7. jump_brass    - Van Halen style with TILT + COMPRESS_K + DIFFUSE_DOWN
    8. sub_bass      - Deep sub with negative TILT + LOWRANK_MORPH
    9. pwm_pad       - Slow pulse width mod with DIFFUSE_UP + INTRA_MIX
    10. pluck        - Quick attack pluck with TILT + COMPRESS_K
    11. wobble_bass  - Dubstep wobble with TILT + LOWRANK_MORPH
    12. theremin_lead - Ethereal theremin with IDENTITY + FINE_CLAMP
"""
from pathlib import Path
from typing import Dict, List

from ...instrument_ir import InstrumentIR, load_instrument

PRESET_DIR = Path(__file__).parent


def list_presets() -> List[str]:
    """
    List all available preset names.

    Returns:
        List of preset names (without .json extension), sorted alphabetically.
    """
    return sorted([f.stem for f in PRESET_DIR.glob("*.json")])


def load_preset(name: str) -> InstrumentIR:
    """
    Load a preset by name.

    Args:
        name: Preset name (without .json extension), e.g. 'saw_lead'

    Returns:
        InstrumentIR object with the preset configuration.

    Raises:
        ValueError: If preset not found.
    """
    path = PRESET_DIR / f"{name}.json"
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
        name: Preset name (without .json extension)

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


__all__ = [
    'list_presets',
    'load_preset',
    'get_all_presets',
    'get_preset_info',
    'PRESET_DIR',
]
