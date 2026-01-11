"""
MacroIR v0 (Performance Macro IR)
=================================

A simple IR for performance macros (vibrato, legato, tremolo, portamento).

These are NOT instruments - they are modifiers that can be applied to any
instrument to add expressive capabilities.

Format:
    macro_ir_v0:
      meta:
        macro_id: "vibrato-classic-v0"
        name: "Classic Vibrato"
        version: 0
        author: "symcrash"
      macro:
        macro_type: VIBRATO
        rate_hz: 5.5
        depth_pitch_cents: 25
        ...
      tags:
        - vibrato
        - expressive

Macro Types:
- VIBRATO: Pitch/amplitude/timbre LFO modulation with delayed onset
- TREMOLO: Amplitude-only LFO modulation
- LEGATO: Smooth pitch transition with phase preservation
- PORTAMENTO: Pitch glide between all notes
"""

from dataclasses import dataclass, field
from typing import List, Optional
import hashlib

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

from .library.capsule import (
    CapsuleV0,
    CapsuleKind,
    CapsuleIdentity,
    MacroProgram,
    MacroType,
    _create_macro_capsule,
)


# =============================================================================
# META
# =============================================================================

@dataclass
class MacroMeta:
    """Macro identity and metadata."""
    macro_id: str = ""
    name: str = ""
    version: int = 0
    author: str = ""
    description: str = ""


# =============================================================================
# MACRO IR
# =============================================================================

@dataclass
class MacroIR:
    """
    MacroIR v0 - Performance Macro Intermediate Representation.

    A minimal IR for defining performance macros (vibrato, legato, etc.)
    that can be compiled to CapsuleV0 with CapsuleKind.MACRO.
    """
    meta: MacroMeta = field(default_factory=MacroMeta)
    macro: MacroProgram = field(default_factory=MacroProgram)
    tags: List[str] = field(default_factory=list)

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate MacroIR structure.

        Returns:
            (is_valid, error_message)
        """
        if not self.meta.macro_id:
            return False, "meta.macro_id is required"
        if not self.meta.name:
            return False, "meta.name is required"
        return True, None

    def to_capsule(self, capsule_id: str = None) -> CapsuleV0:
        """
        Convert MacroIR to CapsuleV0.

        Args:
            capsule_id: Optional explicit capsule ID (generated if None)

        Returns:
            CapsuleV0 with CapsuleKind.MACRO
        """
        if capsule_id is None:
            capsule_id = self._generate_capsule_id()

        return _create_macro_capsule(
            capsule_id=capsule_id,
            name=self.meta.name,
            macro=self.macro,
            author=self.meta.author,
            tags=self.tags,
        )

    def _generate_capsule_id(self) -> str:
        """Generate stable capsule_id from macro identity."""
        identity_str = f"MACRO:{self.meta.macro_id}:{self.meta.name}:{self.meta.author}"
        hash_bytes = hashlib.sha256(identity_str.encode()).hexdigest()[:12]
        safe_name = self.meta.macro_id.lower().replace('_', '-')[:20]
        return f"{safe_name}-{hash_bytes}"


# =============================================================================
# PARSING
# =============================================================================

_MACRO_TYPE_MAP = {
    'VIBRATO': MacroType.VIBRATO,
    'TREMOLO': MacroType.TREMOLO,
    'LEGATO': MacroType.LEGATO,
    'PORTAMENTO': MacroType.PORTAMENTO,
}


def _parse_macro_type(value: str) -> MacroType:
    """Parse macro type from string."""
    if isinstance(value, int):
        return MacroType(value)
    upper = value.upper()
    if upper in _MACRO_TYPE_MAP:
        return _MACRO_TYPE_MAP[upper]
    raise ValueError(f"Unknown macro type: {value}")


def _parse_macro_program(data: dict) -> MacroProgram:
    """Parse MacroProgram from dict."""
    macro_type = _parse_macro_type(data.get('macro_type', 'VIBRATO'))

    return MacroProgram(
        macro_type=macro_type,
        rate_hz=float(data.get('rate_hz', 5.5)),
        depth_pitch_cents=int(data.get('depth_pitch_cents', 0)),
        depth_amp_q15=int(data.get('depth_amp_q15', 0)),
        depth_timbre_q15=int(data.get('depth_timbre_q15', 0)),
        delay_ms=float(data.get('delay_ms', 0.0)),
        ramp_ms=float(data.get('ramp_ms', 0.0)),
        glide_time_ms=float(data.get('glide_time_ms', 0.0)),
        glide_curve=int(data.get('glide_curve', 0)),
        preserve_phase=bool(data.get('preserve_phase', False)),
    )


def load_macro_from_dict(data: dict) -> MacroIR:
    """
    Load MacroIR from a dictionary.

    Supports both:
    - macro_ir_v0: format (canonical)
    - Legacy preset: format (for migration)

    Args:
        data: Dict from YAML/JSON

    Returns:
        MacroIR object

    Raises:
        ValueError: If format is unrecognized
    """
    # Handle top-level wrapper
    if 'macro_ir_v0' in data:
        data = data['macro_ir_v0']
    elif 'preset' in data and data.get('kind') == 'MACRO':
        # Legacy format - convert
        return _convert_legacy_macro(data)
    elif 'preset' in data:
        raise ValueError(
            "The 'preset:' format for instruments is deprecated. "
            "Use 'instrument_ir_v0:' format instead. "
            "See PRESET-AUTHORING-GUIDE.md."
        )

    # Parse canonical macro_ir_v0 format
    meta_data = data.get('meta', {})
    meta = MacroMeta(
        macro_id=meta_data.get('macro_id', ''),
        name=meta_data.get('name', ''),
        version=int(meta_data.get('version', 0)),
        author=meta_data.get('author', ''),
        description=meta_data.get('description', ''),
    )

    macro_data = data.get('macro', {})
    macro = _parse_macro_program(macro_data)

    tags = data.get('tags', [])

    return MacroIR(meta=meta, macro=macro, tags=tags)


def _convert_legacy_macro(data: dict) -> MacroIR:
    """
    Convert legacy preset: format to MacroIR.

    Legacy format:
        preset:
          preset_id: "VIBRATO_CLASSIC_V0"
          author: "symcrash"
          description: "..."
        kind: MACRO
        macro:
          macro_type: VIBRATO
          rate_hz: 5.5
          ...
        tags: [...]
    """
    preset = data.get('preset', {})
    macro_data = data.get('macro', {})
    tags = data.get('tags', [])

    # Extract meta from preset section
    preset_id = preset.get('preset_id', '')
    # Convert VIBRATO_CLASSIC_V0 to vibrato-classic-v0
    macro_id = preset_id.lower().replace('_', '-')

    meta = MacroMeta(
        macro_id=macro_id,
        name=preset.get('name', preset_id.replace('_', ' ').title()),
        version=0,
        author=preset.get('author', ''),
        description=preset.get('description', ''),
    )

    macro = _parse_macro_program(macro_data)

    return MacroIR(meta=meta, macro=macro, tags=tags)


def load_macro(path: str) -> MacroIR:
    """
    Load MacroIR from a YAML or JSON file.

    Args:
        path: Path to .yaml or .json file

    Returns:
        MacroIR object

    Raises:
        ValueError: If format is invalid
        FileNotFoundError: If file doesn't exist
    """
    with open(path, 'r') as f:
        content = f.read()

    if path.endswith('.yaml') or path.endswith('.yml'):
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML files: pip install pyyaml")
        import yaml
        data = yaml.safe_load(content)
    else:
        import json
        data = json.loads(content)

    return load_macro_from_dict(data)


def is_macro_preset(path: str) -> bool:
    """
    Check if a file is a macro preset (vs instrument preset).

    Args:
        path: Path to preset file

    Returns:
        True if the file is a macro preset
    """
    with open(path, 'r') as f:
        content = f.read()

    if path.endswith('.yaml') or path.endswith('.yml'):
        if not HAS_YAML:
            return False
        import yaml
        data = yaml.safe_load(content)
    else:
        import json
        data = json.loads(content)

    # Check for macro_ir_v0 format
    if 'macro_ir_v0' in data:
        return True

    # Check for legacy format with kind: MACRO
    if data.get('kind') == 'MACRO':
        return True

    return False


__all__ = [
    'MacroMeta',
    'MacroIR',
    'load_macro',
    'load_macro_from_dict',
    'is_macro_preset',
]
