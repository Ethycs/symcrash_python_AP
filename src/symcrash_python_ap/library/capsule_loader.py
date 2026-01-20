"""
Capsule Loader (WSE-STD-CAPSULE-COMPAT v0 §6)
==============================================

Implements the "Capsule Apply" algorithm for loading capsules into the runtime.

Algorithm (§6):
1. Validate engine_compat (abi versions, sample_rate, block_size)
2. Load program identity (program_id / IR ref) into engine program selector
3. Apply voice_param_defaults into VoiceEngineRegistry defaults (clamped)
4. Apply control_defaults into ControlRegistry defaults (clamped)
5. Reset/allocate voices deterministically (no runtime histories)
6. Begin runtime

Compatibility Rules (§3-4):
- Capsule defaults override only default values (never ranges/types/slew/bindings)
- Unknown IDs are ignored deterministically unless strict_compat=True
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .capsule import CapsuleV0, CapsuleKind, EngineCompat
from .central_registry import get_central_registry, CentralRegistry
from ..control_registry import ControlRegistry, CONTROL_REGISTRY_V0
from ..voice.voice_engine_registry import VoiceEngineRegistry, get_registry


# =============================================================================
# COMPATIBILITY CHECK
# =============================================================================

@dataclass
class CompatResult:
    """Result of compatibility check."""
    compatible: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def check_compatibility(
    capsule: CapsuleV0,
    current_control_abi: int = 0,
    current_voice_abi: int = 0,
    current_voiceframe_abi: int = 0,
    current_sample_rate: int = 48000,
    current_block_size: int = 64,
    strict: bool = False,
) -> CompatResult:
    """
    Check if capsule is compatible with current engine configuration (§6 step 1).

    Args:
        capsule: Capsule to check
        current_control_abi: Current ControlRegistry ABI version
        current_voice_abi: Current VoiceEngineRegistry ABI version
        current_voiceframe_abi: Current VoiceFrame ABI version
        current_sample_rate: Current sample rate
        current_block_size: Current block size
        strict: If True, any mismatch is an error. If False, only breaking changes error.

    Returns:
        CompatResult with compatible flag, errors, and warnings
    """
    result = CompatResult(compatible=True)
    compat = capsule.identity.engine_compat

    # ABI version checks
    if compat.control_registry_abi != current_control_abi:
        if strict:
            result.errors.append(
                f"ControlRegistry ABI mismatch: capsule={compat.control_registry_abi}, "
                f"current={current_control_abi}"
            )
            result.compatible = False
        else:
            result.warnings.append(
                f"ControlRegistry ABI mismatch (may cause unknown ID warnings)"
            )

    if compat.voice_registry_abi != current_voice_abi:
        if strict:
            result.errors.append(
                f"VoiceEngineRegistry ABI mismatch: capsule={compat.voice_registry_abi}, "
                f"current={current_voice_abi}"
            )
            result.compatible = False
        else:
            result.warnings.append(
                f"VoiceEngineRegistry ABI mismatch (may cause unknown ID warnings)"
            )

    if compat.voiceframe_abi != current_voiceframe_abi:
        if strict:
            result.errors.append(
                f"VoiceFrame ABI mismatch: capsule={compat.voiceframe_abi}, "
                f"current={current_voiceframe_abi}"
            )
            result.compatible = False
        else:
            result.warnings.append(f"VoiceFrame ABI mismatch")

    # Sample rate/block size - always errors (fundamental mismatch)
    if compat.sample_rate_hz != current_sample_rate:
        result.errors.append(
            f"Sample rate mismatch: capsule={compat.sample_rate_hz}Hz, "
            f"current={current_sample_rate}Hz"
        )
        result.compatible = False

    if compat.block_size != current_block_size:
        result.errors.append(
            f"Block size mismatch: capsule={compat.block_size}, "
            f"current={current_block_size}"
        )
        result.compatible = False

    return result


# =============================================================================
# DEFAULT APPLICATION
# =============================================================================

@dataclass
class ApplyResult:
    """Result of applying capsule defaults."""
    success: bool
    applied_control_defaults: Dict[int, int] = field(default_factory=dict)
    applied_voice_params: Dict[int, int] = field(default_factory=dict)
    ignored_control_ids: List[int] = field(default_factory=list)
    ignored_param_ids: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def apply_control_defaults(
    defaults: Dict[int, int],
    registry: ControlRegistry,
    strict: bool = False,
) -> Tuple[Dict[int, int], List[int], List[str]]:
    """
    Apply control defaults with clamping (§3, §6 step 4).

    Args:
        defaults: Dict mapping control_id -> value
        registry: ControlRegistry to validate against
        strict: If True, unknown IDs cause errors

    Returns:
        (applied_defaults, ignored_ids, errors)
    """
    applied = {}
    ignored = []
    errors = []

    for control_id, value in defaults.items():
        if registry.has_control(control_id):
            # Clamp value to legal range
            clamped = registry.clamp(control_id, value)
            applied[control_id] = clamped
        else:
            # Unknown control ID
            ignored.append(control_id)
            if strict:
                errors.append(f"Unknown control_id: {control_id}")

    return applied, ignored, errors


def apply_voice_param_defaults(
    defaults: Dict[int, int],
    registry: VoiceEngineRegistry,
    strict: bool = False,
) -> Tuple[Dict[int, int], List[int], List[str]]:
    """
    Apply voice param defaults with clamping (§4, §6 step 3).

    Args:
        defaults: Dict mapping param_id -> value
        registry: VoiceEngineRegistry to validate against
        strict: If True, unknown IDs cause errors

    Returns:
        (applied_defaults, ignored_ids, errors)
    """
    applied = {}
    ignored = []
    errors = []

    for param_id, value in defaults.items():
        entry = registry.get(param_id)
        if entry is not None:
            # Clamp value to legal range
            clamped = registry.clamp(param_id, value)
            applied[param_id] = clamped
        else:
            # Unknown param ID
            ignored.append(param_id)
            if strict:
                errors.append(f"Unknown param_id: {param_id}")

    return applied, ignored, errors


# =============================================================================
# CAPSULE LOADER
# =============================================================================

class CapsuleLoader:
    """
    Loader for applying capsules to engine state.

    Implements WSE-STD-CAPSULE-COMPAT v0 §6 "Capsule Apply" algorithm.

    Uses CentralRegistry for unified access to all runtime registries.
    """

    def __init__(
        self,
        control_registry: ControlRegistry = None,
        voice_registry: VoiceEngineRegistry = None,
        central_registry: CentralRegistry = None,
        sample_rate: int = 48000,
        block_size: int = 64,
    ):
        """
        Initialize loader with registries and engine config.

        Args:
            control_registry: ControlRegistry instance (uses CentralRegistry if None)
            voice_registry: VoiceEngineRegistry instance (uses default if None)
            central_registry: CentralRegistry instance (uses global if None)
            sample_rate: Engine sample rate
            block_size: Engine block size
        """
        # Use CentralRegistry for unified access
        self.central_registry = central_registry or get_central_registry()

        # Individual registries (can be overridden or accessed via CentralRegistry)
        self.control_registry = control_registry or self.central_registry.controls
        self.voice_registry = voice_registry or get_registry()
        self.sample_rate = sample_rate
        self.block_size = block_size

        # Current ABI versions
        self.control_registry_abi = 0
        self.voice_registry_abi = 0
        self.voiceframe_abi = 0

        # Applied state (for introspection)
        self._current_capsule: Optional[CapsuleV0] = None
        self._applied_control_defaults: Dict[int, int] = {}
        self._applied_voice_params: Dict[int, int] = {}

    def check_compat(self, capsule: CapsuleV0, strict: bool = False) -> CompatResult:
        """
        Check capsule compatibility (§6 step 1).

        Args:
            capsule: Capsule to check
            strict: If True, ABI mismatches are errors

        Returns:
            CompatResult
        """
        return check_compatibility(
            capsule,
            current_control_abi=self.control_registry_abi,
            current_voice_abi=self.voice_registry_abi,
            current_voiceframe_abi=self.voiceframe_abi,
            current_sample_rate=self.sample_rate,
            current_block_size=self.block_size,
            strict=strict,
        )

    def apply_capsule(
        self,
        capsule: CapsuleV0,
        strict_compat: bool = False,
    ) -> ApplyResult:
        """
        Apply capsule to engine state per §6 algorithm.

        Steps:
        1. Validate engine_compat
        2. (External) Load program identity into engine
        3. Apply voice_param_defaults (clamped)
        4. Apply control_defaults (clamped)
        5. (External) Reset/allocate voices
        6. (External) Begin runtime

        Args:
            capsule: Capsule to apply
            strict_compat: If True, unknown IDs cause errors

        Returns:
            ApplyResult with applied defaults and any issues
        """
        result = ApplyResult(success=True)

        # Step 1: Validate compatibility
        compat = self.check_compat(capsule, strict=strict_compat)
        if not compat.compatible:
            result.success = False
            result.errors = compat.errors
            return result

        # Step 2: Program identity (caller handles via result.program_id)

        # Step 3: Apply voice_param_defaults
        voice_applied, voice_ignored, voice_errors = apply_voice_param_defaults(
            capsule.defaults.voice_param_defaults,
            self.voice_registry,
            strict=strict_compat,
        )
        result.applied_voice_params = voice_applied
        result.ignored_param_ids = voice_ignored

        if voice_errors:
            result.errors.extend(voice_errors)
            if strict_compat:
                result.success = False
                return result

        # Step 4: Apply control_defaults
        control_applied, control_ignored, control_errors = apply_control_defaults(
            capsule.defaults.control_defaults,
            self.control_registry,
            strict=strict_compat,
        )
        result.applied_control_defaults = control_applied
        result.ignored_control_ids = control_ignored

        if control_errors:
            result.errors.extend(control_errors)
            if strict_compat:
                result.success = False
                return result

        # Store current state
        self._current_capsule = capsule
        self._applied_control_defaults = control_applied
        self._applied_voice_params = voice_applied

        return result

    def get_program_id(self) -> Optional[int]:
        """Get program_id from current capsule (for engine setup, §6 step 2)."""
        if self._current_capsule is None:
            return None
        return self._current_capsule.program.program_id

    def get_instrument_ir_id(self) -> Optional[str]:
        """Get instrument_ir_id from current capsule (for engine setup, §6 step 2)."""
        if self._current_capsule is None:
            return None
        return self._current_capsule.program.instrument_ir_id

    def get_master_seed(self) -> int:
        """Get master_seed from current capsule (for deterministic voice allocation, §6 step 5)."""
        if self._current_capsule is None:
            return 0x12345678
        return self._current_capsule.seeds.master_seed

    def get_applied_control_default(self, control_id: int) -> Optional[int]:
        """Get an applied control default value."""
        return self._applied_control_defaults.get(control_id)

    def get_applied_voice_param(self, param_id: int) -> Optional[int]:
        """Get an applied voice param default value."""
        return self._applied_voice_params.get(param_id)

    def get_control_value(self, control_id: int) -> int:
        """
        Get effective control value (applied override or registry default).

        Args:
            control_id: Control ID

        Returns:
            Effective value
        """
        if control_id in self._applied_control_defaults:
            return self._applied_control_defaults[control_id]
        return self.control_registry.get_default(control_id)

    def get_voice_param_value(self, param_id: int) -> int:
        """
        Get effective voice param value (applied override or registry default).

        Args:
            param_id: Param ID

        Returns:
            Effective value
        """
        if param_id in self._applied_voice_params:
            return self._applied_voice_params[param_id]
        return self.voice_registry.get_default(param_id)

    # =========================================================================
    # REGISTRY RESOURCE ACCESS (via CentralRegistry)
    # =========================================================================

    def get_regime_graph(self, regime_graph_id: int):
        """
        Get regime graph from CentralRegistry.

        Args:
            regime_graph_id: Regime graph ID

        Returns:
            RegimeGraphSpec or None if not found
        """
        return self.central_registry.regimes.get(regime_graph_id)

    def get_tree_family(self, tree_family_id: int):
        """
        Get tree family from CentralRegistry.

        Args:
            tree_family_id: Tree family ID

        Returns:
            TreeFamily or None if not found
        """
        return self.central_registry.trees.get(tree_family_id)

    def get_pattern(self, pattern_id: int):
        """
        Get node mask pattern from CentralRegistry.

        Args:
            pattern_id: Pattern ID

        Returns:
            PatternSpec or None if not found
        """
        return self.central_registry.patterns.get(pattern_id)

    def validate_resource_references(
        self,
        regime_graph_id: Optional[int] = None,
        tree_a_id: Optional[int] = None,
        tree_b_id: Optional[int] = None,
        pattern_id: Optional[int] = None,
    ) -> List[str]:
        """
        Validate that resource references exist in CentralRegistry.

        Args:
            regime_graph_id: Optional regime graph ID to validate
            tree_a_id: Optional tree A ID to validate
            tree_b_id: Optional tree B ID to validate
            pattern_id: Optional pattern ID to validate

        Returns:
            List of error messages (empty if all valid)
        """
        return self.central_registry.validate_preset_resources(
            regime_graph_id=regime_graph_id,
            tree_a_id=tree_a_id,
            tree_b_id=tree_b_id,
            pattern_id=pattern_id,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_capsule_to_registries(
    capsule: CapsuleV0,
    control_registry: ControlRegistry = None,
    voice_registry: VoiceEngineRegistry = None,
    strict: bool = False,
) -> ApplyResult:
    """
    Apply capsule defaults to registries (simplified interface).

    Args:
        capsule: Capsule to apply
        control_registry: ControlRegistry (uses default if None)
        voice_registry: VoiceEngineRegistry (uses default if None)
        strict: If True, unknown IDs cause errors

    Returns:
        ApplyResult
    """
    loader = CapsuleLoader(
        control_registry=control_registry,
        voice_registry=voice_registry,
    )
    return loader.apply_capsule(capsule, strict_compat=strict)


__all__ = [
    'CompatResult',
    'ApplyResult',
    'check_compatibility',
    'apply_control_defaults',
    'apply_voice_param_defaults',
    'CapsuleLoader',
    'apply_capsule_to_registries',
]
