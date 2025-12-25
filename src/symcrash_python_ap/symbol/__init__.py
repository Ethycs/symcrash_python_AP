"""Symbol Engine - Coefficient executor and operator scheduling.

WSE-STD-030 compliant executor with budget accounting.
"""

from .symbol_engine import (
    SymbolEngine,
    SymbolExecutor,
    OperatorEngine,
    VoiceCoeffState,
    EventPayload,
    FamilySpec,
    TailSpec,
    RegimeOverride,
    DegradePolicy,
    InstrumentProgram,
    ExecutorInput,
)

from .executor_scheduling import (
    Executor030,
    ExecutorStats,
    Budget,
    SymbolState,
    TailSafeResult,
    EventContext,
)

__all__ = [
    # Main engine
    'SymbolEngine',
    'SymbolExecutor',
    'OperatorEngine',
    # Data structures
    'VoiceCoeffState',
    'EventPayload',
    'FamilySpec',
    'TailSpec',
    'RegimeOverride',
    'DegradePolicy',
    'InstrumentProgram',
    'ExecutorInput',
    # Executor scheduling (WSE-STD-030)
    'Executor030',
    'ExecutorStats',
    'Budget',
    'SymbolState',
    'TailSafeResult',
    'EventContext',
]
