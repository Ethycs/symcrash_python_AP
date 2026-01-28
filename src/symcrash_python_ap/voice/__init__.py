"""Voice Engine - CMMM voice control, scheduling, and degradation.

WSE-STD-050 (degradation), WSE-STD-060 (time ownership) compliant.
"""

from .voice_engine import (
    VoiceEngine,
    VoiceBank,
    VoiceState,
    VoicePhase,
    VoiceStateWithProgram,
    VoiceFrameABI,
    InstrumentProgram,
    RegimeGraph,
    RegimeSpec,
    Regime,
    GestureBasis,
    IntentVector,
    EventPayload,
    TreeMorphState,
    ControlMap,
    TowerLevel,
    DegradationLevel,
    EngineMode,  # NEW: Simple vs Full CMMM mode
    HysteresisSpec,
    OverlaySpec,
    EdgeType,
    ConditionSpec,
    EdgeSpec,
    ArcSpec,
    ArcState,
    FamilySelectionRule,
    FamilyPolicySpec,
    TreeSelectionRule,
    TreeBlendRule,
    TreePolicySpec,
    LaneRuleType,
    LaneRuleSpec,
    EventPolicySpec,
)

# Scheduler components - canonical location is symbol.executor_scheduling
# but re-exported here for backward compatibility
from .scheduler import (
    BlockScheduler,
    VoiceSelector,
    TileScheduler,
    DegradeController,
    ScheduleResult,
    VoiceScheduleInfo,
    HealthTelemetry,
    EmissionGroup,
)

from .degradation import (
    VoiceClass,
    VoiceStatus,
    VoiceState as DegradationVoiceState,
    FreezeState,
    VoiceScheduler,
    EventDropPolicy,
    DeterminismValidator,
)

from .orchestrator import (
    BlockOrchestrator,
    VoiceSlot,
    VoiceState as OrchestratorVoiceState,
    OrchestratedPipeline,
)

from .time_ownership import (
    TimeDomain,
    EngineType,
    EngineCapabilities,
    BlockTime,
    ProhibitedPattern,
    PatternViolation,
    TimeOwnershipValidator,
    IntentFrame,
    VoiceFrame as TimeOwnershipVoiceFrame,
    DeterminismChecker,
    ComplianceReport,
)

from .emitter_bridge import (
    create_emitter_with_state,
    create_standalone_emitter,
    create_emitter_with_engine,
    VoiceEngineAdapter,
)

__all__ = [
    # Main engine
    'VoiceEngine',
    'VoiceBank',
    'VoiceState',
    'VoicePhase',
    'VoiceStateWithProgram',
    'VoiceFrameABI',
    'InstrumentProgram',
    # Regime system
    'RegimeGraph',
    'RegimeSpec',
    'Regime',
    'GestureBasis',
    'IntentVector',
    'EventPayload',
    'TreeMorphState',
    'ControlMap',
    # Tower
    'TowerLevel',
    'DegradationLevel',
    'EngineMode',  # Simple vs Full CMMM mode
    # Hysteresis & overlays
    'HysteresisSpec',
    'OverlaySpec',
    'EdgeType',
    'ConditionSpec',
    'EdgeSpec',
    'ArcSpec',
    'ArcState',
    # Policy specs
    'FamilySelectionRule',
    'FamilyPolicySpec',
    'TreeSelectionRule',
    'TreeBlendRule',
    'TreePolicySpec',
    'LaneRuleType',
    'LaneRuleSpec',
    'EventPolicySpec',
    # Scheduler
    'BlockScheduler',
    'VoiceSelector',
    'TileScheduler',
    'DegradeController',
    'ScheduleResult',
    'VoiceScheduleInfo',
    'HealthTelemetry',
    'EmissionGroup',
    # Degradation (WSE-STD-050)
    'VoiceClass',
    'VoiceStatus',
    'DegradationVoiceState',
    'FreezeState',
    'VoiceScheduler',
    'EventDropPolicy',
    'DeterminismValidator',
    # Orchestrator
    'BlockOrchestrator',
    'VoiceSlot',
    'OrchestratorVoiceState',
    'OrchestratedPipeline',
    # Time ownership (WSE-STD-060)
    'TimeDomain',
    'EngineType',
    'EngineCapabilities',
    'BlockTime',
    'ProhibitedPattern',
    'PatternViolation',
    'TimeOwnershipValidator',
    'IntentFrame',
    'TimeOwnershipVoiceFrame',
    'DeterminismChecker',
    'ComplianceReport',
    # Emitter bridge
    'create_emitter_with_state',
    'create_standalone_emitter',
    'create_emitter_with_engine',
    'VoiceEngineAdapter',
]
