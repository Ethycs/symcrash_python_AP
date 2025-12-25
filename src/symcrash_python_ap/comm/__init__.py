"""Communication - UART protocols and dual-channel link.

WSE-STD-020 (UART protocol), WSE-STD-100 (Dual UART) compliant.
"""

from .uart_protocol import (
    VoiceFrameV0,
    VoiceFrameDelta,
    UARTPacket,
    UARTListener,
    VoiceLatch,
    MockUART,
    BackboneEntry,
    Payload64,
    PatternEntry,
    EventLane,
    MsgType,
    PayloadMode,
    TargetKind,
    ErrorCode,
    AckStatus,
    FieldMask,
)

from .dual_uart_link import (
    DualUARTLink,
    DualUARTPacket,
    FrameHeader,
    MockDualUART,
    # Control messages
    NoteEvent,
    SetTreeTarget,
    Status,
    HelloAck,
    SetMacro,
    StreamCtrl,
    # Data messages
    VoiceDelta,
    VoiceBurst,
    DataCredit,
    DataStreamStart,
    # Entry types
    LeafGainEntry,
    LeafCoeffEntry,
    EventEntry,
    # Enums
    ControlMsgType,
    DataMsgType,
    ErrorCode as DualErrorCode,
    VoiceDeltaSubtype,
    VoiceBurstMode,
    NoteEventFlags,
    TreeMorphCurve,
    VoiceState,
)

from .ctrl_delta import (
    CtrlDeltaPacket,
    ScopeType,
)

from .uart_listener import (
    InteractiveListener,
    Colors,
)

__all__ = [
    # UART Protocol (WSE-STD-020)
    'VoiceFrameV0',
    'VoiceFrameDelta',
    'UARTPacket',
    'UARTListener',
    'VoiceLatch',
    'MockUART',
    'BackboneEntry',
    'Payload64',
    'PatternEntry',
    'EventLane',
    'MsgType',
    'PayloadMode',
    'TargetKind',
    'ErrorCode',
    'AckStatus',
    'FieldMask',
    # Dual UART Link (WSE-STD-100)
    'DualUARTLink',
    'DualUARTPacket',
    'FrameHeader',
    'MockDualUART',
    'NoteEvent',
    'SetTreeTarget',
    'Status',
    'HelloAck',
    'SetMacro',
    'StreamCtrl',
    'VoiceDelta',
    'VoiceBurst',
    'DataCredit',
    'DataStreamStart',
    'LeafGainEntry',
    'LeafCoeffEntry',
    'EventEntry',
    'ControlMsgType',
    'DataMsgType',
    'DualErrorCode',
    'VoiceDeltaSubtype',
    'VoiceBurstMode',
    'NoteEventFlags',
    'TreeMorphCurve',
    'VoiceState',
    # Ctrl Delta
    'CtrlDeltaPacket',
    'ScopeType',
    # Listener
    'InteractiveListener',
    'Colors',
]
