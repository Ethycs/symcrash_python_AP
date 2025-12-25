"""
WSE-STD-020 UART Protocol Tests
================================

Tests for VoiceFrame ABI, Payload64, BackboneEntry, and UART packet protocol.
"""

import pytest
import struct
from typing import List

# Import from uart_protocol
from symcrash_python_ap.comm import (
    # Constants - import directly from module for constants
    VoiceFrameV0, VoiceFrameDelta, VoiceLatch,
    UARTPacket, MockUART,
    BackboneEntry, Payload64,
    # Enums
    EventLane, MsgType, PayloadMode, TargetKind, ErrorCode, AckStatus, FieldMask,
    PatternEntry,
)
from symcrash_python_ap.comm.uart_protocol import (
    SYNC_WORD, PROTOCOL_VERSION, NUM_EVENT_LANES, MAX_BACKBONE_ENTRIES,
    crc32c, resolve_pattern, DEFAULT_PATTERN_TABLE,
)
from symcrash_python_ap.fixed import Q15_ONE, Q15_HALF


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def q15_one():
    return Q15_ONE


@pytest.fixture
def q15_half():
    return Q15_HALF


@pytest.fixture
def sample_backbone():
    """Create sample backbone entries."""
    return [
        BackboneEntry(op_id=1, strength_q15=0x4000, param0_q15=0x1000, param1_q15=0),
        BackboneEntry(op_id=5, strength_q15=0x6000, param0_q15=0x2000, param1_q15=0x0800),
    ]


@pytest.fixture
def sample_frame(sample_backbone):
    """Create sample VoiceFrameV0."""
    return VoiceFrameV0(
        voice_id=5,
        block_index=100,
        regime_id=2,
        gear_q15=0x5000,
        tree_id_a=1,
        tree_id_b=3,
        tree_blend_q15=0x4000,
        gain_q15=0x6000,
        traction_q15=0x7000,
        backbone=sample_backbone,
        event_mask=0x05,
        event_strength_q15=[0x3000, 0, 0x5000, 0, 0, 0, 0, 0],
        payload_mode=PayloadMode.COMMON,
        payload_common=Payload64.for_leaf(15, variant0=1),
    )


# =============================================================================
# CRC-32C TESTS
# =============================================================================

class TestCRC32C:
    """CRC-32C implementation tests."""

    def test_known_value(self):
        """Test against known CRC-32C value."""
        # "123456789" has known CRC-32C = 0xE3069283
        data = b"123456789"
        expected = 0xE3069283
        computed = crc32c(data)
        assert computed == expected, f"CRC mismatch: {computed:08X} vs {expected:08X}"

    def test_empty_input(self):
        """CRC of empty input."""
        computed = crc32c(b"")
        assert computed == 0x00000000

    def test_single_byte(self):
        """CRC of single byte."""
        computed = crc32c(b"\x00")
        assert isinstance(computed, int)
        assert 0 <= computed <= 0xFFFFFFFF

    def test_deterministic(self):
        """CRC is deterministic."""
        data = b"test data for crc"
        crc1 = crc32c(data)
        crc2 = crc32c(data)
        assert crc1 == crc2


# =============================================================================
# PAYLOAD64 TESTS
# =============================================================================

class TestPayload64:
    """Payload64 format tests per WSE-STD-020 Section 3."""

    def test_pack_unpack_identity(self):
        """Pack and unpack should be inverse operations."""
        p = Payload64(
            payload_type=1,
            lane_hint=2,
            pattern_id=3,
            target_kind=TargetKind.LEAF,
            target_id=15,
            coeff_mask=0xAA,
            variant0=0x55,
            variant1=0xFF,
        )
        packed = p.pack()
        unpacked = Payload64.unpack(packed)

        assert unpacked.payload_type == p.payload_type
        assert unpacked.lane_hint == p.lane_hint
        assert unpacked.pattern_id == p.pattern_id
        assert unpacked.target_kind == p.target_kind
        assert unpacked.target_id == p.target_id
        assert unpacked.coeff_mask == p.coeff_mask
        assert unpacked.variant0 == p.variant0
        assert unpacked.variant1 == p.variant1

    def test_to_bytes_from_bytes(self):
        """Serialization round-trip."""
        p = Payload64.for_leaf(20, coeff_mask=0x0F, variant0=7)
        data = p.to_bytes()
        restored = Payload64.from_bytes(data)

        assert restored.target_kind == TargetKind.LEAF
        assert restored.target_id == 20
        assert restored.coeff_mask == 0x0F
        assert restored.variant0 == 7

    def test_for_pattern(self):
        """Factory method for pattern payloads."""
        p = Payload64.for_pattern(5, coeff_mask=0xFF)
        assert p.target_kind == TargetKind.PATTERN
        assert p.pattern_id == 5
        assert p.coeff_mask == 0xFF

    def test_for_leaf(self):
        """Factory method for leaf payloads."""
        p = Payload64.for_leaf(31, variant0=2)
        assert p.target_kind == TargetKind.LEAF
        assert p.target_id == 31
        assert p.variant0 == 2

    def test_bit_layout(self):
        """Verify correct bit positions per spec."""
        p = Payload64(
            payload_type=0x01,
            lane_hint=0x02,
            pattern_id=0x03,
            target_kind=TargetKind.LEAF,  # 0x01
            target_id=0x04,
            coeff_mask=0x05,
            variant0=0x06,
            variant1=0x07,
        )
        packed = p.pack()

        # Extract individual bytes
        assert (packed >> 0) & 0xFF == 0x01  # payload_type
        assert (packed >> 8) & 0xFF == 0x02  # lane_hint
        assert (packed >> 16) & 0xFF == 0x03  # pattern_id
        assert (packed >> 24) & 0xFF == 0x01  # target_kind (LEAF)
        assert (packed >> 32) & 0xFF == 0x04  # target_id
        assert (packed >> 40) & 0xFF == 0x05  # coeff_mask
        assert (packed >> 48) & 0xFF == 0x06  # variant0
        assert (packed >> 56) & 0xFF == 0x07  # variant1


# =============================================================================
# BACKBONE ENTRY TESTS
# =============================================================================

class TestBackboneEntry:
    """BackboneEntry tests per WSE-STD-020 Section 2.2."""

    def test_serialization(self):
        """Serialize and deserialize backbone entry."""
        entry = BackboneEntry(op_id=5, strength_q15=0x6000, param0_q15=0x1000, param1_q15=0x2000)
        data = entry.to_bytes()
        restored = BackboneEntry.from_bytes(data)

        assert restored.op_id == entry.op_id
        assert restored.strength_q15 == entry.strength_q15
        assert restored.param0_q15 == entry.param0_q15
        assert restored.param1_q15 == entry.param1_q15

    def test_byte_size(self):
        """Backbone entry has correct byte size."""
        entry = BackboneEntry()
        data = entry.to_bytes()
        assert len(data) == BackboneEntry.byte_size()
        assert len(data) == 7


# =============================================================================
# VOICEFRAME V0 TESTS
# =============================================================================

class TestVoiceFrameV0:
    """VoiceFrame v0 ABI tests per WSE-STD-020 Section 2.1."""

    def test_serialization(self, sample_frame):
        """Serialize and deserialize VoiceFrame."""
        data = sample_frame.to_bytes()
        restored = VoiceFrameV0.from_bytes(data)

        assert restored.voice_id == sample_frame.voice_id
        assert restored.block_index == sample_frame.block_index
        assert restored.regime_id == sample_frame.regime_id
        assert restored.gear_q15 == sample_frame.gear_q15
        assert restored.tree_id_a == sample_frame.tree_id_a
        assert restored.tree_id_b == sample_frame.tree_id_b
        assert restored.tree_blend_q15 == sample_frame.tree_blend_q15
        assert restored.gain_q15 == sample_frame.gain_q15
        assert restored.traction_q15 == sample_frame.traction_q15
        assert len(restored.backbone) == len(sample_frame.backbone)
        assert restored.event_mask == sample_frame.event_mask

    def test_backbone_order_preserved(self, sample_backbone):
        """Backbone operator order is semantic and must be preserved."""
        frame = VoiceFrameV0(backbone=sample_backbone)
        data = frame.to_bytes()
        restored = VoiceFrameV0.from_bytes(data)

        # Order must match
        for i, (orig, rest) in enumerate(zip(sample_backbone, restored.backbone)):
            assert rest.op_id == orig.op_id, f"Backbone order wrong at {i}"
            assert rest.strength_q15 == orig.strength_q15

    def test_event_strength_all_lanes(self):
        """All 8 event lanes have strengths."""
        strengths = [i * 1000 for i in range(NUM_EVENT_LANES)]
        frame = VoiceFrameV0(event_strength_q15=strengths, event_mask=0xFF)
        data = frame.to_bytes()
        restored = VoiceFrameV0.from_bytes(data)

        for i in range(NUM_EVENT_LANES):
            assert restored.event_strength_q15[i] == strengths[i]

    def test_payload_mode_common(self):
        """COMMON payload mode."""
        frame = VoiceFrameV0(
            payload_mode=PayloadMode.COMMON,
            payload_common=Payload64.for_leaf(10),
        )
        data = frame.to_bytes()
        restored = VoiceFrameV0.from_bytes(data)

        assert restored.payload_mode == PayloadMode.COMMON
        assert restored.payload_common.target_id == 10


# =============================================================================
# UART PACKET TESTS
# =============================================================================

class TestUARTPacket:
    """UART packet framing tests per WSE-STD-020 Section 6."""

    def test_sync_word(self, sample_frame):
        """Packet starts with sync word 0xA55A."""
        packet = UARTPacket(
            msg_type=MsgType.VF_ABSOLUTE,
            payload=sample_frame.to_bytes()
        )
        data = packet.to_bytes()

        sync = struct.unpack('<H', data[:2])[0]
        assert sync == SYNC_WORD

    def test_crc_verification(self, sample_frame):
        """Packet CRC is verified on parse."""
        packet = UARTPacket(
            msg_type=MsgType.VF_ABSOLUTE,
            seq=42,
            payload=sample_frame.to_bytes()
        )
        data = packet.to_bytes()

        # Parse should succeed
        parsed, consumed = UARTPacket.from_bytes(data)
        assert parsed is not None
        assert parsed.seq == 42
        assert parsed.msg_type == MsgType.VF_ABSOLUTE

    def test_crc_mismatch_rejected(self, sample_frame):
        """Corrupt packet is rejected."""
        packet = UARTPacket(
            msg_type=MsgType.VF_ABSOLUTE,
            payload=sample_frame.to_bytes()
        )
        data = bytearray(packet.to_bytes())

        # Corrupt the payload
        data[10] ^= 0xFF

        parsed, consumed = UARTPacket.from_bytes(bytes(data))
        assert parsed is None  # Should fail CRC

    def test_make_ack(self):
        """ACK packet creation."""
        ack = UARTPacket.make_ack(100, AckStatus.OK_QUEUED)
        assert ack.msg_type == MsgType.ACK

        seq, status, detail = struct.unpack('<HBB', ack.payload)
        assert seq == 100
        assert status == AckStatus.OK_QUEUED

    def test_make_nack(self):
        """NACK packet creation."""
        nack = UARTPacket.make_nack(200, ErrorCode.INVALID_TREE_ID, detail=3)
        assert nack.msg_type == MsgType.NACK

        seq, error, detail = struct.unpack('<HBB', nack.payload)
        assert seq == 200
        assert error == ErrorCode.INVALID_TREE_ID
        assert detail == 3


# =============================================================================
# DETERMINISTIC COMPLETION TESTS
# =============================================================================

class TestVoiceLatch:
    """Deterministic completion tests per WSE-STD-020 Section 5."""

    def test_keep_previous(self):
        """Omitted fields keep previous values."""
        latch = VoiceLatch()
        latch.regime_id = 2
        latch.tree_id_a = 1
        latch.gain_q15 = 0x6000
        latch.traction_q15 = 0x7000

        # Delta only updates event_mask
        delta = VoiceFrameDelta(
            voice_id=5,
            block_index=101,
            field_mask=(1 << FieldMask.EVENT_MASK),
            event_mask=0x03
        )

        completed = latch.apply_delta(delta)

        # Updated field
        assert completed.event_mask == 0x03

        # Kept fields
        assert completed.regime_id == 2
        assert completed.tree_id_a == 1
        assert completed.gain_q15 == 0x6000
        assert completed.traction_q15 == 0x7000

    def test_tree_b_defaults_to_a(self):
        """tree_id_b defaults to tree_id_a if 0."""
        latch = VoiceLatch()
        latch.tree_id_a = 3
        latch.tree_id_b = 0  # Not set

        delta = VoiceFrameDelta(
            voice_id=0,
            block_index=1,
            field_mask=0,  # No fields updated
        )

        completed = latch.apply_delta(delta)
        assert completed.tree_id_b == 3  # Defaults to A

    def test_reset_on_allocation(self):
        """Reset on voice allocation per Section 5.2."""
        latch = VoiceLatch()
        latch.event_mask = 0xFF  # Set
        latch.gain_q15 = 0x7FFF

        # Reset with preset defaults
        latch.reset(preset_defaults={'gear_q15': 0x5000, 'gain_q15': 0x4000})

        assert latch.event_mask == 0  # Reset
        assert latch.gear_q15 == 0x5000  # From preset
        assert latch.payload_mode == PayloadMode.COMMON


# =============================================================================
# PATTERN TABLE TESTS
# =============================================================================

class TestPatternTable:
    """Pattern table tests per WSE-STD-020 Section 3.3."""

    def test_pattern_0_all_leaves(self):
        """Pattern 0 selects all leaves."""
        p = Payload64.for_pattern(0)
        leaf_mask, coeff_mask = resolve_pattern(p)
        assert leaf_mask == 0xFFFFFFFF

    def test_pattern_1_low_band(self):
        """Pattern 1 selects low band (leaves 0-7)."""
        p = Payload64.for_pattern(1)
        leaf_mask, coeff_mask = resolve_pattern(p)
        assert leaf_mask == 0x000000FF

    def test_leaf_single_selection(self):
        """Direct leaf selection."""
        for leaf_id in [0, 15, 31]:
            p = Payload64.for_leaf(leaf_id)
            leaf_mask, coeff_mask = resolve_pattern(p)
            assert leaf_mask == (1 << leaf_id)

    def test_coeff_mask_override(self):
        """coeff_mask in payload overrides pattern default."""
        p = Payload64.for_pattern(0, coeff_mask=0x03)
        _, coeff_mask = resolve_pattern(p)
        assert coeff_mask == 0x03


# =============================================================================
# EVENT LANES TESTS
# =============================================================================

class TestEventLanes:
    """Event lanes tests per WSE-STD-020 Section 2.3."""

    def test_lane_ids(self):
        """Lane IDs match spec."""
        assert EventLane.EVT_OP_FIRE == 0
        assert EventLane.EVT_OP_TOGGLE == 1
        assert EventLane.EVT_INJECT == 2
        assert EventLane.EVT_MATERIAL == 3
        assert EventLane.EVT_TREE == 4
        assert EventLane.EVT_CLOUD == 5
        assert EventLane.EVT_VOICE == 6
        assert EventLane.EVT_DEBUG == 7

    def test_eight_lanes(self):
        """Exactly 8 event lanes."""
        assert len(EventLane) == 8
        assert NUM_EVENT_LANES == 8


# =============================================================================
# MOCK UART TESTS
# =============================================================================

class TestMockUART:
    """Mock UART tests for testing without hardware."""

    def test_host_device_roundtrip(self, sample_frame):
        """Host -> Device -> ACK round trip."""
        mock = MockUART()

        # Host sends
        packet = UARTPacket(
            msg_type=MsgType.VF_ABSOLUTE,
            seq=1,
            payload=sample_frame.to_bytes()
        )
        mock.host_send(packet)

        # Device processes
        mock.device_process()

        # Host receives ACK
        ack = mock.host_receive(timeout=1.0)
        assert ack is not None
        assert ack.msg_type == MsgType.ACK

    def test_block_scheduling(self, sample_frame):
        """Frames are queued by block_index."""
        mock = MockUART()
        mock.current_block = 0

        # Send frame for future block
        frame = VoiceFrameV0(voice_id=0, block_index=5, gain_q15=0x5000)
        packet = UARTPacket(
            msg_type=MsgType.VF_ABSOLUTE,
            payload=frame.to_bytes()
        )
        mock.host_send(packet)
        mock.device_process()

        # Advance blocks
        for _ in range(5):
            frames = mock.advance_block()

        # Frame should be due at block 5
        assert mock.current_block == 5
        assert len(frames) == 1
        assert frames[0].voice_id == 0

    def test_latch_persistence(self):
        """Device latches persist across frames."""
        mock = MockUART()

        # Send frame to establish latch
        frame1 = VoiceFrameV0(voice_id=0, block_index=0, tree_id_a=2, gain_q15=0x6000)
        mock.host_send(UARTPacket(msg_type=MsgType.VF_ABSOLUTE, payload=frame1.to_bytes()))
        mock.device_process()

        # Check latch
        latch = mock.device_latches.get(0)
        assert latch is not None
        assert latch.tree_id_a == 2
        assert latch.gain_q15 == 0x6000


# =============================================================================
# MESSAGE TYPE TESTS
# =============================================================================

class TestMessageTypes:
    """Message type enum tests per WSE-STD-020 Section 6.3."""

    def test_message_type_values(self):
        """Message types have correct values."""
        assert MsgType.VF_ABSOLUTE == 0x01
        assert MsgType.VF_DELTA == 0x02
        assert MsgType.PARAM_ABSOLUTE == 0x03
        assert MsgType.PARAM_DELTA == 0x04
        assert MsgType.PRESET_LOAD == 0x05
        assert MsgType.ACK == 0x06
        assert MsgType.NACK == 0x07
        assert MsgType.TELEMETRY == 0x08
        assert MsgType.CAPTURE == 0x09
