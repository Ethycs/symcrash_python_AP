"""
Library Module
==============

Implements:
- WSE-STD-CAPSULE-COMPAT v0: Capsule compatibility contract
- WSE-STD-IPLC v0: Instrument Preset Lifecycle
- WSE-STD-PRESETLIB v0: Preset Library Storage + Index

Components:
- capsule.py: CapsuleV0 data model
- capsule_codec.py: Binary serialization (.wsecap files)
- capsule_loader.py: Capsule apply algorithm
- lifecycle.py: IPLC state machine
- compiler.py: Capsule compiler
- preset_library.py: PresetLibraryV0

Usage:
    from symcrash_python_ap.library import (
        CapsuleV0,
        create_instrument_capsule,
        CapsuleCompiler,
        PresetLibraryV0,
    )

    # Compile a capsule
    compiler = CapsuleCompiler()
    result = compiler.compile_instrument(
        program_id=1001,
        name="Glitch Violin",
    )

    # Import to library
    library = PresetLibraryV0("./my_library")
    library.import_capsule(result.capsule)

    # Resolve and apply
    record = library.resolve_capsule("glitch-violin-...")
"""

from .capsule import (
    CapsuleKind,
    AssetType,
    EngineCompat,
    CapsuleIdentity,
    ProgramReference,
    AssetRef,
    SeedConfig,
    CapsuleDefaults,
    CapsuleV0,
    create_instrument_capsule,
    create_synth_capsule,
    create_asset_capsule,
)

from .capsule_codec import (
    CAPSULE_MAGIC,
    CAPSULE_VERSION,
    pack_control_defaults,
    unpack_control_defaults,
    pack_voice_params,
    unpack_voice_params,
    capsule_to_bytes,
    capsule_from_bytes,
    save_capsule,
    load_capsule,
    save_capsule_bundle,
    load_capsule_bundle,
)

from .capsule_loader import (
    CompatResult,
    ApplyResult,
    check_compatibility,
    apply_control_defaults,
    apply_voice_param_defaults,
    CapsuleLoader,
    apply_capsule_to_registries,
)

from .lifecycle import (
    LifecycleState,
    CapsuleRecord,
    ALLOWED_TRANSITIONS,
    TRANSITION_TRIGGERS,
    validate_transition,
    get_transition_trigger,
    transition,
    TransitionError,
    create_draft_record,
    create_compiled_record,
    create_published_record,
    get_allowed_transitions,
    is_terminal_state,
    is_library_state,
    is_runtime_state,
)

from .compiler import (
    CompileResult,
    CapsuleCompiler,
)

from .preset_library import (
    ImportResult,
    AssetRecord,
    Query,
    ResolutionSelector,
    PresetLibraryV0,
)


__all__ = [
    # Capsule model
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

    # Codec
    'CAPSULE_MAGIC',
    'CAPSULE_VERSION',
    'pack_control_defaults',
    'unpack_control_defaults',
    'pack_voice_params',
    'unpack_voice_params',
    'capsule_to_bytes',
    'capsule_from_bytes',
    'save_capsule',
    'load_capsule',
    'save_capsule_bundle',
    'load_capsule_bundle',

    # Loader
    'CompatResult',
    'ApplyResult',
    'check_compatibility',
    'apply_control_defaults',
    'apply_voice_param_defaults',
    'CapsuleLoader',
    'apply_capsule_to_registries',

    # Lifecycle (IPLC)
    'LifecycleState',
    'CapsuleRecord',
    'ALLOWED_TRANSITIONS',
    'TRANSITION_TRIGGERS',
    'validate_transition',
    'get_transition_trigger',
    'transition',
    'TransitionError',
    'create_draft_record',
    'create_compiled_record',
    'create_published_record',
    'get_allowed_transitions',
    'is_terminal_state',
    'is_library_state',
    'is_runtime_state',

    # Compiler
    'CompileResult',
    'CapsuleCompiler',

    # Preset Library
    'ImportResult',
    'AssetRecord',
    'Query',
    'ResolutionSelector',
    'PresetLibraryV0',
]
