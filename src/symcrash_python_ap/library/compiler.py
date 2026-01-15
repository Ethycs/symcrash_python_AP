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

Natural IR Compilation:
- InstrumentIR.to_capsule() compiles the IR to CapsuleV0
- CapsuleCompiler.compile_from_ir(ir) handles full compilation with validation
- L0 macros map to control_defaults
- L2 regime graph provides regime_graph_id
- L3 emit templates provide tree family defaults
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

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
    ControlProgram,
    create_instrument_capsule,
    create_synth_capsule,
)
from .lifecycle import LifecycleState, CapsuleRecord, create_compiled_record
from ..control_registry import ControlRegistry, ControlID
from ..voice.voice_engine_registry import VoiceEngineRegistry, VoiceParamID, get_registry

if TYPE_CHECKING:
    from ..instrument_ir import InstrumentIR
    from ..macro_ir import MacroIR


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
        control_program: 'ControlProgram' = None,
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
            control_program: Optional ControlProgram with system macro defaults

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
            control=control_program,  # System macro defaults for cockpit
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

    # =========================================================================
    # NATURAL IR COMPILATION
    # =========================================================================

    def compile_from_ir(
        self,
        ir: 'InstrumentIR',
        program_id: int = None,
        capsule_id: str = None,
    ) -> CompileResult:
        """
        Compile InstrumentIR directly to CapsuleV0.

        This is the natural compilation path - the IR contains all metadata,
        control mappings, regime graphs, and emission templates needed.

        Args:
            ir: Complete InstrumentIR v0
            program_id: Override program_id (uses hash of instrument_id if None)
            capsule_id: Override capsule_id (generated from IR if None)

        Returns:
            CompileResult with capsule in COMPILED state

        Example:
            compiler = CapsuleCompiler()
            ir = load_instrument("my_synth.yaml")
            result = compiler.compile_from_ir(ir)
            if result.success:
                capsule = result.capsule
        """
        result = CompileResult(success=True)

        # Validate IR structure first
        valid, error_msg = ir.validate()
        if not valid:
            result.errors.append(f"IR validation failed: {error_msg}")
            result.success = False
            return result

        # Extract metadata from IR
        meta = ir.meta
        name = meta.name or meta.instrument_id or "Unnamed"
        author = meta.author or ""
        master_seed = meta.seed_u32 or 0x12345678

        # Generate program_id from instrument_id if not provided
        if program_id is None:
            if meta.instrument_id:
                # Hash instrument_id to get stable program_id
                hash_val = int(hashlib.sha256(
                    meta.instrument_id.encode()
                ).hexdigest()[:8], 16)
                program_id = hash_val & 0x7FFFFFFF  # Keep positive
            else:
                result.errors.append("Either program_id or instrument_id is required")
                result.success = False
                return result

        # Extract control defaults from L0 macros
        control_defaults = self._extract_control_defaults_from_ir(ir)
        result.warnings.extend(control_defaults.get('_warnings', []))
        control_defaults.pop('_warnings', None)

        # Extract voice param defaults from compile config
        voice_param_defaults = self._extract_voice_param_defaults_from_ir(ir)

        # Extract regime graph ID (hash of regime structure)
        regime_graph_id = self._extract_regime_graph_id(ir)

        # Extract tree family ID from emission templates
        tree_family_id = self._extract_tree_family_id(ir)

        # Build tags from IR
        tags = self._extract_tags_from_ir(ir)

        # Build asset references from compile.map_library
        assets = self._extract_assets_from_ir(ir)

        # Extract control program from IR (system macro defaults)
        control_program = None
        if ir.control is not None:
            control_program = ir.control.to_control_program()

        # Use standard compile_instrument with extracted values
        return self.compile_instrument(
            program_id=program_id,
            name=name,
            author=author,
            tags=tags,
            control_defaults=control_defaults,
            voice_param_defaults=voice_param_defaults,
            regime_graph_id=regime_graph_id,
            tree_family_id=tree_family_id,
            instrument_ir={'id': meta.instrument_id},
            master_seed=master_seed,
            assets=assets,
            capsule_id=capsule_id,
            control_program=control_program,
        )

    def compile_from_macro(
        self,
        macro_ir: 'MacroIR',
        capsule_id: str = None,
    ) -> CompileResult:
        """
        Compile MacroIR to CapsuleV0 with CapsuleKind.MACRO.

        Args:
            macro_ir: Complete MacroIR v0
            capsule_id: Override capsule_id (generated from IR if None)

        Returns:
            CompileResult with macro capsule
        """
        result = CompileResult(success=True)

        # Validate macro structure
        valid, error_msg = macro_ir.validate()
        if not valid:
            result.errors.append(f"Macro validation failed: {error_msg}")
            result.success = False
            return result

        try:
            # Delegate to MacroIR.to_capsule()
            capsule = macro_ir.to_capsule(capsule_id)
            result.capsule = capsule
            result.capsule_id = capsule.identity.capsule_id
            result.record = create_compiled_record(capsule)
        except Exception as e:
            result.errors.append(f"Macro compilation failed: {e}")
            result.success = False

        return result

    def _extract_control_defaults_from_ir(
        self,
        ir: 'InstrumentIR',
    ) -> Dict[int, int]:
        """
        Extract control defaults from IR L0 macros and compile config.

        Maps L0 macro defaults to ControlRegistry control IDs.
        """
        defaults: Dict[int, int] = {}
        warnings: List[str] = []

        # Map L0 macros to control IDs
        macro_to_control = {
            'M_BRIGHT': ControlID.TIMBRE,
            'M_ATTACK': ControlID.ATK_BLOCKS,
            'M_DECAY': ControlID.DCY_BLOCKS,
            'M_SUSTAIN': ControlID.SUS_Q15,
            'M_RELEASE': ControlID.REL_BLOCKS,
            'M_GLITCH': ControlID.CHAOS,
            'M_MORPH': ControlID.MORPH,
            'M_TEXTURE': ControlID.TEXTURE,
            'M_DETUNE': ControlID.DETUNE,
        }

        # Extract from L0 macros
        for macro in ir.layers.L0_graph.macros:
            macro_id = macro.id.upper()
            if macro_id in macro_to_control:
                control_id = macro_to_control[macro_id]
                # Convert macro default (0.0-1.0) to Q15 (0-32767)
                default_q15 = int(macro.default * 32767)
                defaults[control_id] = default_q15
            else:
                warnings.append(f"Unknown macro {macro.id} - no control mapping")

        # Extract ADSR from L0 Amp node if present
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Amp" and node.params.adsr:
                adsr = node.params.adsr
                # Convert ms to blocks (64 samples @ 48kHz = 1.33ms/block)
                ms_per_block = 64 / 48.0  # ~1.33ms
                defaults[ControlID.ATK_BLOCKS] = max(1, int(adsr.a_ms / ms_per_block))
                defaults[ControlID.DCY_BLOCKS] = max(1, int(adsr.d_ms / ms_per_block))
                defaults[ControlID.SUS_Q15] = int(adsr.s * 32767)
                defaults[ControlID.REL_BLOCKS] = max(1, int(adsr.r_ms / ms_per_block))

        defaults['_warnings'] = warnings
        return defaults

    def _extract_voice_param_defaults_from_ir(
        self,
        ir: 'InstrumentIR',
    ) -> Dict[int, int]:
        """
        Extract voice param defaults from IR.

        Sources:
        - compile.frame_emitter_policy for frame emission params
        - L0 nodes for waveform/carrier params
        - L0 FM6 nodes for 6-operator FM synthesis params
        """
        defaults: Dict[int, int] = {}

        # Extract from L0 nodes
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Osc":
                # Could map waveform, detune, pulse width to voice params
                if node.params.waveform:
                    # Map waveform string to ID
                    wave_map = {'sine': 0, 'saw': 1, 'pulse': 2, 'tri': 3}
                    wave_id = wave_map.get(node.params.waveform.lower(), 0)
                    # Would need VoiceParamID.WAVEFORM constant
                    pass

            elif node.type == "FM6":
                # Extract FM6 synthesis parameters
                # Set OSC_MODE to 3 (FM6)
                defaults[VoiceParamID.OSC_MODE] = 3

                # Algorithm (1-32)
                if node.params.algorithm is not None:
                    defaults[VoiceParamID.FM6_ALGORITHM] = node.params.algorithm

                # Feedback (0-7)
                if node.params.feedback is not None:
                    defaults[VoiceParamID.FM6_FEEDBACK] = node.params.feedback

                # Operator ratios (Q16.16: 65536 = 1.0x)
                # Stored as U32 to support ratios > 1.0x
                if node.params.op_ratios:
                    ratio_params = [
                        VoiceParamID.FM6_OP0_RATIO,
                        VoiceParamID.FM6_OP1_RATIO,
                        VoiceParamID.FM6_OP2_RATIO,
                        VoiceParamID.FM6_OP3_RATIO,
                        VoiceParamID.FM6_OP4_RATIO,
                        VoiceParamID.FM6_OP5_RATIO,
                    ]
                    for i, ratio in enumerate(node.params.op_ratios[:6]):
                        defaults[ratio_params[i]] = ratio

                # Operator levels (Q15: 32767 = full)
                if node.params.op_levels:
                    level_params = [
                        VoiceParamID.FM6_OP0_LEVEL,
                        VoiceParamID.FM6_OP1_LEVEL,
                        VoiceParamID.FM6_OP2_LEVEL,
                        VoiceParamID.FM6_OP3_LEVEL,
                        VoiceParamID.FM6_OP4_LEVEL,
                        VoiceParamID.FM6_OP5_LEVEL,
                    ]
                    for i, level in enumerate(node.params.op_levels[:6]):
                        defaults[level_params[i]] = level

                # Topology morphing
                if node.params.fm6_topo_a is not None:
                    defaults[VoiceParamID.FM6_TOPO_A] = node.params.fm6_topo_a
                if node.params.fm6_topo_b is not None:
                    defaults[VoiceParamID.FM6_TOPO_B] = node.params.fm6_topo_b
                if node.params.fm6_topo_blend_q15 is not None:
                    defaults[VoiceParamID.FM6_TOPO_BLEND] = node.params.fm6_topo_blend_q15

        return defaults

    def _extract_regime_graph_id(self, ir: 'InstrumentIR') -> Optional[int]:
        """
        Generate regime graph ID from L2 regime structure.

        Creates a stable hash of the regime graph for identification.
        """
        regimes = ir.layers.L2_regimes
        if not regimes.regime_nodes:
            return None

        # Create identity string from regime structure
        regime_str = ""
        for node in sorted(regimes.regime_nodes, key=lambda n: n.id):
            regime_str += f"{node.id}:{node.kind}:{node.emit_template}|"
        for trans in sorted(regimes.transitions, key=lambda t: (t.from_regime, t.to)):
            regime_str += f"{trans.from_regime}->{trans.to}:{trans.when}|"

        # Hash to get stable ID
        hash_val = int(hashlib.sha256(regime_str.encode()).hexdigest()[:8], 16)
        return hash_val & 0x7FFFFFFF

    def _extract_tree_family_id(self, ir: 'InstrumentIR') -> Optional[int]:
        """
        Extract tree family ID from L3 emission templates.

        Looks at tree targets across templates to identify the family.
        """
        emit_templates = ir.layers.L3_cmmm.emit_templates
        if not emit_templates:
            return None

        # Collect unique tree references
        tree_refs = set()
        for template in emit_templates.values():
            tree_refs.add(template.trees.A)
            tree_refs.add(template.trees.B)

        if not tree_refs:
            return None

        # Create stable hash from tree references
        tree_str = "|".join(sorted(tree_refs))
        hash_val = int(hashlib.sha256(tree_str.encode()).hexdigest()[:8], 16)
        return hash_val & 0x7FFFFFFF

    def _extract_tags_from_ir(self, ir: 'InstrumentIR') -> List[str]:
        """
        Extract tags from IR structure.

        Infers tags from:
        - L0 node types (has filter, has noise, etc.)
        - L2 regime kinds present
        - Compile config features
        """
        tags = []

        # Check L0 nodes
        node_types = {node.type for node in ir.layers.L0_graph.nodes}
        if "Filter" in node_types:
            tags.append("filtered")
        if "Noise" in node_types:
            tags.append("noise")
        if "FXSend" in node_types:
            tags.append("fx")
        if "FM6" in node_types:
            tags.append("fm6")
            tags.append("fm")

        # Check waveforms
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Osc" and node.params.waveform:
                tags.append(node.params.waveform.lower())

        # Check regime kinds
        regime_kinds = {node.kind for node in ir.layers.L2_regimes.regime_nodes}
        if "GlitchChatter" in regime_kinds:
            tags.append("glitch")
        if "NoiseTexture" in regime_kinds:
            tags.append("texture")

        return list(set(tags))  # Remove duplicates

    def _extract_assets_from_ir(self, ir: 'InstrumentIR') -> List[AssetRef]:
        """
        Extract asset references from IR compile config.

        Maps compile.map_library entries to AssetRef objects.
        """
        assets = []

        for asset_id, asset_path in ir.compile.map_library.items():
            # Infer asset type from path/id
            if 'wavetable' in asset_id.lower() or asset_path.endswith('.wtbl'):
                asset_type = AssetType.WAVETABLE
            elif 'material' in asset_id.lower():
                asset_type = AssetType.MATERIAL
            elif 'atlas' in asset_id.lower():
                asset_type = AssetType.TABLE_ATLAS
            else:
                asset_type = AssetType.OTHER

            assets.append(AssetRef(
                asset_type=asset_type,
                asset_id=asset_id,
                asset_path=asset_path,
            ))

        return assets


__all__ = [
    'CompileResult',
    'CapsuleCompiler',
]
