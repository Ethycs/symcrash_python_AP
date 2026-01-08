"""
Capsule Compiler (WSE-STD-IPLC v0 §4)
======================================

Converts authoring inputs (InstrumentIR, etc.) into COMPILED capsule bundles.

Compiler Contract (§4):
1. All overrides are clampable against registries
2. All IDs are resolved (program_id, graph_id)
3. All assets are referenced or embedded deterministically
4. All ABI requirements are declared in manifest

Versioning Rules (§7):
- capsule_id identifies "the instrument identity" across versions
- Same identity = same capsule_id across rebuilds
- Changed identity = new capsule_id
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .capsule import (
    CapsuleV0,
    CapsuleKind,
    CapsuleIdentity,
    EngineCompat,
    ProgramReference,
    CapsuleDefaults,
    SeedConfig,
    AssetRef,
    AssetType,
    create_instrument_capsule,
    create_synth_capsule,
)
from .lifecycle import LifecycleState, CapsuleRecord, create_compiled_record
from ..control_registry import ControlRegistry
from ..voice.voice_engine_registry import VoiceEngineRegistry, get_registry


# =============================================================================
# COMPILE RESULT
# =============================================================================

@dataclass
class CompileResult:
    """Result of capsule compilation."""
    success: bool
    capsule: Optional[CapsuleV0] = None
    record: Optional[CapsuleRecord] = None  # In COMPILED state
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    capsule_id: Optional[str] = None  # Generated stable ID


# =============================================================================
# CAPSULE COMPILER
# =============================================================================

class CapsuleCompiler:
    """
    Converts authoring inputs into COMPILED capsule bundles.

    Per WSE-STD-IPLC v0 §4.
    """

    def __init__(
        self,
        control_registry: ControlRegistry = None,
        voice_registry: VoiceEngineRegistry = None,
        sample_rate: int = 48000,
        block_size: int = 64,
    ):
        """
        Initialize compiler with registries.

        Args:
            control_registry: ControlRegistry for validation
            voice_registry: VoiceEngineRegistry for validation
            sample_rate: Target sample rate
            block_size: Target block size
        """
        self.control_registry = control_registry or ControlRegistry()
        self.voice_registry = voice_registry or get_registry()
        self.sample_rate = sample_rate
        self.block_size = block_size

        # ABI versions
        self.control_registry_abi = 0
        self.voice_registry_abi = 0
        self.voiceframe_abi = 0

    def compile_instrument(
        self,
        program_id: int,
        name: str,
        author: str = "",
        tags: List[str] = None,
        control_defaults: Dict[int, int] = None,
        voice_param_defaults: Dict[int, int] = None,
        regime_graph_id: int = None,
        tree_family_id: int = None,
        instrument_ir: dict = None,
        master_seed: int = 0x12345678,
        assets: List[AssetRef] = None,
        capsule_id: str = None,
    ) -> CompileResult:
        """
        Compile inputs to CapsuleInstrument.

        Args:
            program_id: Voice engine program ID (required)
            name: Human-readable name (required)
            author: Optional author name
            tags: Optional list of tags
            control_defaults: Optional control default overrides
            voice_param_defaults: Optional voice param default overrides
            regime_graph_id: Optional regime graph ID
            tree_family_id: Optional tree family ID
            instrument_ir: Optional InstrumentIR data (dict)
            master_seed: PRNG seed for determinism
            assets: Optional asset references
            capsule_id: Optional explicit capsule_id (generated if None)

        Returns:
            CompileResult with capsule in COMPILED state
        """
        result = CompileResult(success=True)
        tags = tags or []
        control_defaults = control_defaults or {}
        voice_param_defaults = voice_param_defaults or {}
        assets = assets or []

        # Generate capsule_id if not provided (§7)
        if capsule_id is None:
            capsule_id = self._generate_capsule_id(
                kind=CapsuleKind.INSTRUMENT,
                program_id=program_id,
                name=name,
                author=author,
            )
        result.capsule_id = capsule_id

        # Step 1: Validate and clamp control defaults (§4.1)
        clamped_controls, control_warnings = self._validate_control_defaults(control_defaults)
        result.warnings.extend(control_warnings)

        # Step 2: Validate and clamp voice param defaults (§4.1)
        clamped_voice_params, voice_warnings = self._validate_voice_param_defaults(voice_param_defaults)
        result.warnings.extend(voice_warnings)

        # Step 3: Validate program_id and other IDs (§4.2)
        id_errors = self._validate_ids(program_id, regime_graph_id, tree_family_id)
        if id_errors:
            result.errors.extend(id_errors)
            result.success = False
            return result

        # Step 4: Create capsule with ABI requirements (§4.4)
        capsule = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id=capsule_id,
                kind=CapsuleKind.INSTRUMENT,
                name=name,
                author=author,
                tags=tags,
                engine_compat=EngineCompat(
                    control_registry_abi=self.control_registry_abi,
                    voice_registry_abi=self.voice_registry_abi,
                    voiceframe_abi=self.voiceframe_abi,
                    sample_rate_hz=self.sample_rate,
                    block_size=self.block_size,
                ),
            ),
            program=ProgramReference(
                program_id=program_id,
                instrument_ir_id=instrument_ir.get('id') if instrument_ir else None,
                regime_graph_id=regime_graph_id,
                tree_family_id=tree_family_id,
            ),
            defaults=CapsuleDefaults(
                control_defaults=clamped_controls,
                voice_param_defaults=clamped_voice_params,
            ),
            assets=assets,
            seeds=SeedConfig(
                master_seed=master_seed,
                derivation_rule_id=0,  # deterministic
            ),
        )

        # Validate capsule structure
        validation_issues = capsule.validate()
        for issue in validation_issues:
            if issue.startswith("ERROR"):
                result.errors.append(issue)
                result.success = False
            else:
                result.warnings.append(issue)

        if result.success:
            result.capsule = capsule
            result.record = create_compiled_record(capsule)

        return result

    def compile_synth(
        self,
        program_id: int,
        name: str,
        author: str = "",
        tags: List[str] = None,
        control_defaults: Dict[int, int] = None,
        voice_param_defaults: Dict[int, int] = None,
        master_seed: int = 0x12345678,
        capsule_id: str = None,
    ) -> CompileResult:
        """
        Compile inputs to CapsuleSynth.

        Args:
            program_id: Voice engine program ID
            name: Human-readable name
            author: Optional author name
            tags: Optional list of tags
            control_defaults: Optional control default overrides
            voice_param_defaults: Optional voice param default overrides
            master_seed: PRNG seed for determinism
            capsule_id: Optional explicit capsule_id

        Returns:
            CompileResult with capsule in COMPILED state
        """
        result = CompileResult(success=True)
        tags = tags or []
        control_defaults = control_defaults or {}
        voice_param_defaults = voice_param_defaults or {}

        # Generate capsule_id if not provided
        if capsule_id is None:
            capsule_id = self._generate_capsule_id(
                kind=CapsuleKind.SYNTH,
                program_id=program_id,
                name=name,
                author=author,
            )
        result.capsule_id = capsule_id

        # Validate defaults
        clamped_controls, control_warnings = self._validate_control_defaults(control_defaults)
        result.warnings.extend(control_warnings)

        clamped_voice_params, voice_warnings = self._validate_voice_param_defaults(voice_param_defaults)
        result.warnings.extend(voice_warnings)

        # Validate IDs
        id_errors = self._validate_ids(program_id)
        if id_errors:
            result.errors.extend(id_errors)
            result.success = False
            return result

        # Create capsule
        capsule = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id=capsule_id,
                kind=CapsuleKind.SYNTH,
                name=name,
                author=author,
                tags=tags,
                engine_compat=EngineCompat(
                    control_registry_abi=self.control_registry_abi,
                    voice_registry_abi=self.voice_registry_abi,
                    voiceframe_abi=self.voiceframe_abi,
                    sample_rate_hz=self.sample_rate,
                    block_size=self.block_size,
                ),
            ),
            program=ProgramReference(program_id=program_id),
            defaults=CapsuleDefaults(
                control_defaults=clamped_controls,
                voice_param_defaults=clamped_voice_params,
            ),
            seeds=SeedConfig(master_seed=master_seed, derivation_rule_id=0),
        )

        validation_issues = capsule.validate()
        for issue in validation_issues:
            if issue.startswith("ERROR"):
                result.errors.append(issue)
                result.success = False
            else:
                result.warnings.append(issue)

        if result.success:
            result.capsule = capsule
            result.record = create_compiled_record(capsule)

        return result

    def _generate_capsule_id(
        self,
        kind: CapsuleKind,
        program_id: int,
        name: str,
        author: str,
    ) -> str:
        """
        Generate stable capsule_id from identity (§7).

        Same identity = same capsule_id across rebuilds.
        """
        # Create identity string
        identity_str = f"{kind.name}:{program_id}:{name}:{author}"

        # Hash to create stable ID
        hash_bytes = hashlib.sha256(identity_str.encode()).hexdigest()[:12]

        # Create human-readable ID
        safe_name = name.lower().replace(' ', '-')[:20]
        return f"{safe_name}-{hash_bytes}"

    def _validate_control_defaults(
        self,
        defaults: Dict[int, int],
    ) -> tuple[Dict[int, int], List[str]]:
        """
        Validate and clamp control defaults against registry (§4.1).

        Returns:
            (clamped_defaults, warnings)
        """
        clamped = {}
        warnings = []

        for control_id, value in defaults.items():
            if self.control_registry.has_control(control_id):
                clamped_value = self.control_registry.clamp(control_id, value)
                clamped[control_id] = clamped_value

                if clamped_value != value:
                    warnings.append(
                        f"Control {control_id} value {value} clamped to {clamped_value}"
                    )
            else:
                warnings.append(f"Unknown control_id {control_id} ignored")

        return clamped, warnings

    def _validate_voice_param_defaults(
        self,
        defaults: Dict[int, int],
    ) -> tuple[Dict[int, int], List[str]]:
        """
        Validate and clamp voice param defaults against registry (§4.1).

        Returns:
            (clamped_defaults, warnings)
        """
        clamped = {}
        warnings = []

        for param_id, value in defaults.items():
            entry = self.voice_registry.get(param_id)
            if entry is not None:
                clamped_value = self.voice_registry.clamp(param_id, value)
                clamped[param_id] = clamped_value

                if clamped_value != value:
                    warnings.append(
                        f"Voice param {param_id} value {value} clamped to {clamped_value}"
                    )
            else:
                warnings.append(f"Unknown param_id {param_id} ignored")

        return clamped, warnings

    def _validate_ids(
        self,
        program_id: int,
        regime_graph_id: int = None,
        tree_family_id: int = None,
    ) -> List[str]:
        """
        Validate IDs are resolved (§4.2).

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Program ID validation
        if program_id is None:
            errors.append("program_id is required")
        elif program_id < 0:
            errors.append(f"Invalid program_id: {program_id}")

        # Regime graph ID validation (if provided)
        if regime_graph_id is not None and regime_graph_id < 0:
            errors.append(f"Invalid regime_graph_id: {regime_graph_id}")

        # Tree family ID validation (if provided)
        if tree_family_id is not None and tree_family_id < 0:
            errors.append(f"Invalid tree_family_id: {tree_family_id}")

        return errors


__all__ = [
    'CompileResult',
    'CapsuleCompiler',
]
