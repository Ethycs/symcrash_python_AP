"""
Capsule Object Model (WSE-STD-CAPSULE-COMPAT v0 §2)
====================================================

Portable capsule objects for PresetLibrary storage and runtime loading.

Capsule Types (v0):
1. CapsuleInstrument - ProgramPreset + optional InstrumentIR references
2. CapsuleSynth - ProgramPreset + synth-focused defaults
3. CapsuleAsset - References only; no control semantics

Key design choices:
- Library stores capsules
- Pipeline owns morphing
- Capsules include control/macro defaults and engine defaults
- Capsules MUST NOT contain runtime histories (coeff_state, phase, PRNG, etc.)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


# =============================================================================
# ENUMS
# =============================================================================

class CapsuleKind(IntEnum):
    """Capsule type identifier (§1)."""
    INSTRUMENT = 0
    SYNTH = 1
    ASSET = 2


class AssetType(IntEnum):
    """Asset reference types (§2)."""
    IR = 0
    WAVETABLE = 1
    MATERIAL = 2
    TABLE_ATLAS = 3
    OTHER = 255


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EngineCompat:
    """
    Engine compatibility requirements (§2).

    Capsule MUST declare ABI versions for compatibility checking.
    """
    control_registry_abi: int = 0
    voice_registry_abi: int = 0
    voiceframe_abi: int = 0
    sample_rate_hz: int = 48000
    block_size: int = 64


@dataclass
class CapsuleIdentity:
    """
    Capsule identity metadata (§2).

    Unique identification and human-readable info.
    """
    capsule_id: str
    kind: CapsuleKind
    name: str
    author: str = ""
    tags: List[str] = field(default_factory=list)
    engine_compat: EngineCompat = field(default_factory=EngineCompat)


@dataclass
class ProgramReference:
    """
    Program identity (§2).

    Either program_id OR instrument_ir_id must be set.
    """
    program_id: Optional[int] = None
    instrument_ir_id: Optional[str] = None
    regime_graph_id: Optional[int] = None
    tree_family_id: Optional[int] = None


@dataclass
class AssetRef:
    """
    Asset reference (§2).

    Points to external resources like IRs, wavetables, materials.
    """
    asset_type: AssetType
    asset_id: str
    asset_hash: Optional[str] = None
    asset_path: Optional[str] = None


@dataclass
class SeedConfig:
    """
    Seed configuration for determinism (§2).

    Controls PRNG initialization for deterministic playback.
    """
    master_seed: int = 0x12345678
    derivation_rule_id: int = 0  # 0 = deterministic


@dataclass
class CapsuleDefaults:
    """
    Default value overrides (§3-4).

    Overrides ONLY default values, never:
    - control ranges
    - value types
    - slew policy / slew param
    - binding targets
    """
    # control_id -> default value (§3)
    control_defaults: Dict[int, int] = field(default_factory=dict)

    # param_id -> default value (§4)
    voice_param_defaults: Dict[int, int] = field(default_factory=dict)


# =============================================================================
# CAPSULE V0
# =============================================================================

@dataclass
class CapsuleV0:
    """
    Complete capsule object per WSE-STD-CAPSULE-COMPAT v0 §2.

    Portable "instrument identity" that can be:
    - stored in a PresetLibrary
    - compiled-to by tools
    - loaded by the runtime pipeline
    - applied without ambiguity across engines

    MUST NOT contain runtime histories (coeff_state, phase, ADSR state, PRNG state).
    """
    identity: CapsuleIdentity
    program: ProgramReference = field(default_factory=ProgramReference)
    defaults: CapsuleDefaults = field(default_factory=CapsuleDefaults)
    assets: List[AssetRef] = field(default_factory=list)
    seeds: SeedConfig = field(default_factory=SeedConfig)

    def validate(self, strict: bool = False) -> List[str]:
        """
        Validate capsule structure.

        Args:
            strict: If True, unknown IDs cause errors. If False, warnings only.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Identity validation
        if not self.identity.capsule_id:
            issues.append("ERROR: capsule_id is required")

        if not self.identity.name:
            issues.append("ERROR: name is required")

        # Program validation
        if self.program.program_id is None and self.program.instrument_ir_id is None:
            issues.append("ERROR: Either program_id or instrument_ir_id must be set")

        if self.program.program_id is not None and self.program.instrument_ir_id is not None:
            issues.append("WARNING: Both program_id and instrument_ir_id set; program_id takes precedence")

        # Asset validation
        for asset in self.assets:
            if not asset.asset_id:
                issues.append("ERROR: Asset reference missing asset_id")

        return issues

    @property
    def is_instrument(self) -> bool:
        """Check if capsule is an instrument type."""
        return self.identity.kind == CapsuleKind.INSTRUMENT

    @property
    def is_synth(self) -> bool:
        """Check if capsule is a synth type."""
        return self.identity.kind == CapsuleKind.SYNTH

    @property
    def is_asset(self) -> bool:
        """Check if capsule is an asset type."""
        return self.identity.kind == CapsuleKind.ASSET


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_instrument_capsule(
    capsule_id: str,
    name: str,
    program_id: int,
    author: str = "",
    tags: List[str] = None,
    control_defaults: Dict[int, int] = None,
    voice_param_defaults: Dict[int, int] = None,
) -> CapsuleV0:
    """
    Create a CapsuleInstrument.

    Args:
        capsule_id: Unique identifier
        name: Human-readable name
        program_id: Voice engine program ID
        author: Optional author name
        tags: Optional list of tags
        control_defaults: Optional control default overrides
        voice_param_defaults: Optional voice param default overrides

    Returns:
        CapsuleV0 with INSTRUMENT kind
    """
    return CapsuleV0(
        identity=CapsuleIdentity(
            capsule_id=capsule_id,
            kind=CapsuleKind.INSTRUMENT,
            name=name,
            author=author,
            tags=tags or [],
        ),
        program=ProgramReference(program_id=program_id),
        defaults=CapsuleDefaults(
            control_defaults=control_defaults or {},
            voice_param_defaults=voice_param_defaults or {},
        ),
    )


def create_synth_capsule(
    capsule_id: str,
    name: str,
    program_id: int,
    author: str = "",
    tags: List[str] = None,
    control_defaults: Dict[int, int] = None,
    voice_param_defaults: Dict[int, int] = None,
) -> CapsuleV0:
    """
    Create a CapsuleSynth.

    Args:
        capsule_id: Unique identifier
        name: Human-readable name
        program_id: Voice engine program ID
        author: Optional author name
        tags: Optional list of tags
        control_defaults: Optional control default overrides
        voice_param_defaults: Optional voice param default overrides

    Returns:
        CapsuleV0 with SYNTH kind
    """
    return CapsuleV0(
        identity=CapsuleIdentity(
            capsule_id=capsule_id,
            kind=CapsuleKind.SYNTH,
            name=name,
            author=author,
            tags=tags or [],
        ),
        program=ProgramReference(program_id=program_id),
        defaults=CapsuleDefaults(
            control_defaults=control_defaults or {},
            voice_param_defaults=voice_param_defaults or {},
        ),
    )


def create_asset_capsule(
    capsule_id: str,
    name: str,
    assets: List[AssetRef],
    author: str = "",
    tags: List[str] = None,
) -> CapsuleV0:
    """
    Create a CapsuleAsset (references only, no control semantics).

    Args:
        capsule_id: Unique identifier
        name: Human-readable name
        assets: List of asset references
        author: Optional author name
        tags: Optional list of tags

    Returns:
        CapsuleV0 with ASSET kind
    """
    return CapsuleV0(
        identity=CapsuleIdentity(
            capsule_id=capsule_id,
            kind=CapsuleKind.ASSET,
            name=name,
            author=author,
            tags=tags or [],
        ),
        assets=assets,
    )


__all__ = [
    'CapsuleKind',
    'AssetType',
    'EngineCompat',
    'CapsuleIdentity',
    'ProgramReference',
    'AssetRef',
    'SeedConfig',
    'CapsuleDefaults',
    'CapsuleV0',
    'create_instrument_capsule',
    'create_synth_capsule',
    'create_asset_capsule',
]
