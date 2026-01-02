"""
Pico2 Hardware Controller Receiver
==================================

Receives SetMacro messages from the Pico2 hardware controller over USB CDC
and routes them to the parameter registry for synthesis pipeline integration.

Usage:
    receiver = Pico2Receiver(port="COM3")
    receiver.start()

    # Get current macro values
    value = receiver.get_macro(0)  # Returns Q15 value (0-32767)

    receiver.stop()
"""

import serial
import struct
import threading
import queue
import time
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field

from .dual_uart_link import (
    cobs_decode,
    cobs_encode,
    crc32c,
    FRAME_MAGIC,
    FRAME_HEADER_SIZE,
    ControlMsgType,
    SetMacro,
    FrameHeader,
    HelloAck,
    DualUARTPacket,
)
from ..parameter_registry import (
    ParameterBank,
    ParameterRegistry,
    get_registry,
    FieldMask,
    MaskedUpdate,
    UpdateMode,
    Q15_MAX,
)


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_MACROS = 10  # Maximum macro channels (6 ADSR+Filter + 4 Oscillator)


# =============================================================================
# MACRO STATE
# =============================================================================

@dataclass
class MacroState:
    """Current state of a macro parameter."""
    value_q15: int = 0      # Raw Q15 value (0-32767)
    value_float: float = 0.0  # Normalized float (0.0-1.0)
    last_update: float = 0.0  # Timestamp of last update
    changed: bool = False     # Changed since last read


@dataclass
class ReceiverStats:
    """Statistics for the receiver."""
    frames_received: int = 0
    frames_valid: int = 0
    crc_errors: int = 0
    decode_errors: int = 0
    macro_updates: int = 0
    bytes_received: int = 0


# =============================================================================
# PICO2 RECEIVER
# =============================================================================

class Pico2Receiver:
    """
    Receives SetMacro messages from Pico2 hardware controller.

    Connects to the USB CDC control port and parses WSE-STD-100 frames
    to extract macro values from the hardware potentiometers.
    """

    def __init__(
        self,
        port: str = None,
        baud: int = 115200,
        registry: Optional[ParameterRegistry] = None,
    ):
        """
        Initialize the Pico2 receiver.

        Args:
            port: Serial port (e.g., "COM3" on Windows, "/dev/ttyACM0" on Linux)
            baud: Baud rate (default 115200 for USB CDC)
            registry: Optional ParameterRegistry to route updates to
        """
        self.port = port
        self.baud = baud
        self.registry = registry or get_registry()

        # Serial connection
        self._serial: Optional[serial.Serial] = None

        # State
        self._running = False
        self._macros: Dict[int, MacroState] = {
            i: MacroState() for i in range(NUM_MACROS)
        }
        self._lock = threading.Lock()

        # Receive thread
        self._rx_thread: Optional[threading.Thread] = None
        self._rx_buffer = bytearray()

        # Message queue for async processing
        self._message_queue: queue.Queue = queue.Queue()

        # Callbacks
        self._macro_callbacks: List[Callable[[int, int], None]] = []

        # Stats
        self.stats = ReceiverStats()

    # =========================================================================
    # CONNECTION
    # =========================================================================

    def connect(self) -> bool:
        """
        Connect to the Pico2 serial port.

        Returns:
            True if connected successfully
        """
        if self._serial is not None:
            return True

        if self.port is None:
            return False

        try:
            self._serial = serial.Serial(
                self.port,
                self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.001  # Reduced from 0.1 for lower latency
            )
            # Explicitly set DTR to signal connection to Pico2
            # This triggers tud_cdc_n_connected() to return true
            self._serial.dtr = True
            return True
        except serial.SerialException as e:
            print(f"[Pico2Receiver] Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from the serial port."""
        if self._serial:
            self._serial.close()
            self._serial = None

    def is_connected(self) -> bool:
        """Check if connected to serial port."""
        return self._serial is not None and self._serial.is_open

    # =========================================================================
    # START/STOP
    # =========================================================================

    def start(self) -> bool:
        """
        Start the receiver thread.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        if not self.connect():
            return False

        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        return True

    def stop(self):
        """Stop the receiver thread."""
        self._running = False

        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None

        self.disconnect()

    # =========================================================================
    # MACRO ACCESS
    # =========================================================================

    def get_macro(self, macro_id: int) -> int:
        """
        Get current Q15 value for a macro.

        Args:
            macro_id: Macro channel (0-7)

        Returns:
            Q15 value (0-32767)
        """
        with self._lock:
            if 0 <= macro_id < NUM_MACROS:
                return self._macros[macro_id].value_q15
            return 0

    def get_macro_float(self, macro_id: int) -> float:
        """
        Get current normalized float value for a macro.

        Args:
            macro_id: Macro channel (0-7)

        Returns:
            Float value (0.0-1.0)
        """
        with self._lock:
            if 0 <= macro_id < NUM_MACROS:
                return self._macros[macro_id].value_float
            return 0.0

    def get_all_macros(self) -> Dict[int, int]:
        """
        Get all current macro values.

        Returns:
            Dict mapping macro_id -> Q15 value
        """
        with self._lock:
            return {i: m.value_q15 for i, m in self._macros.items()}

    def has_changed(self, macro_id: int) -> bool:
        """
        Check if a macro has changed since last read.

        Args:
            macro_id: Macro channel (0-7)

        Returns:
            True if value changed
        """
        with self._lock:
            if 0 <= macro_id < NUM_MACROS:
                changed = self._macros[macro_id].changed
                self._macros[macro_id].changed = False
                return changed
            return False

    def get_changed_macros(self) -> Dict[int, int]:
        """
        Get all macros that have changed since last call.

        Returns:
            Dict mapping macro_id -> Q15 value for changed macros
        """
        result = {}
        with self._lock:
            for i, m in self._macros.items():
                if m.changed:
                    result[i] = m.value_q15
                    m.changed = False
        return result

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def add_macro_callback(self, callback: Callable[[int, int], None]):
        """
        Add callback for macro value changes.

        Callback signature: callback(macro_id: int, value_q15: int)
        """
        self._macro_callbacks.append(callback)

    def inject_macro(self, macro_id: int, value_q15: int):
        """
        Inject a macro value change programmatically.

        This allows software controls (like ImGui) to send macro changes
        through the same callback chain as hardware messages.

        Args:
            macro_id: Macro channel (0-9)
            value_q15: Q15 value (0-32767)
        """
        # Update state
        with self._lock:
            if 0 <= macro_id < NUM_MACROS:
                state = self._macros[macro_id]
                state.value_q15 = value_q15
                state.value_float = value_q15 / Q15_MAX
                state.last_update = time.time()
                state.changed = True

        # Fire callbacks (same as _handle_set_macro)
        for callback in self._macro_callbacks:
            try:
                callback(macro_id, value_q15)
            except Exception as e:
                print(f"[Pico2Receiver] Callback error: {e}")

    def remove_macro_callback(self, callback: Callable[[int, int], None]):
        """Remove a macro callback."""
        if callback in self._macro_callbacks:
            self._macro_callbacks.remove(callback)

    # =========================================================================
    # PARAMETER REGISTRY INTEGRATION
    # =========================================================================

    def create_masked_update(self) -> MaskedUpdate:
        """
        Create a MaskedUpdate from current macro values.

        Maps macros to parameter registry fields for pipeline integration.

        Returns:
            MaskedUpdate with macro values
        """
        update = MaskedUpdate(mode=UpdateMode.ABS)

        # Map macro 0-5 to common synth parameters
        # This is a basic mapping - can be customized per instrument
        macro_to_field = {
            0: FieldMask.DRIVE_Q15,      # Macro 0 -> Drive
            1: FieldMask.TONE_Q15,       # Macro 1 -> Tone
            2: FieldMask.TREE_BLEND_Q15, # Macro 2 -> Tree blend
            3: FieldMask.TRACTION_Q15,   # Macro 3 -> Traction
            4: FieldMask.GAIN_Q15,       # Macro 4 -> Gain
            5: FieldMask.PRESSURE_Q15,   # Macro 5 -> Pressure
        }

        with self._lock:
            for macro_id, field in macro_to_field.items():
                if macro_id < NUM_MACROS:
                    update.set(field, self._macros[macro_id].value_q15)

        return update

    # =========================================================================
    # RECEIVE LOOP
    # =========================================================================

    def _rx_loop(self):
        """Main receive loop (runs in background thread)."""
        while self._running:
            try:
                if self._serial and self._serial.in_waiting:
                    data = self._serial.read(self._serial.in_waiting)
                    self.stats.bytes_received += len(data)
                    self._rx_buffer.extend(data)
                    self._process_buffer()
                else:
                    time.sleep(0.0001)  # Reduced from 0.001 for faster polling
            except serial.SerialException as e:
                print(f"[Pico2Receiver] Serial error: {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"[Pico2Receiver] RX error: {e}")
                time.sleep(0.01)

    def _process_buffer(self):
        """Process buffered data, extracting complete frames."""
        # Look for COBS delimiter (0x00)
        while b'\x00' in self._rx_buffer:
            idx = self._rx_buffer.index(b'\x00')
            frame_data = bytes(self._rx_buffer[:idx])
            self._rx_buffer = self._rx_buffer[idx + 1:]

            if frame_data:
                self._process_frame(frame_data)

    def _process_frame(self, cobs_data: bytes):
        """Process a single COBS-encoded frame."""
        self.stats.frames_received += 1

        # COBS decode
        decoded = cobs_decode(cobs_data)
        if decoded is None or len(decoded) < FRAME_HEADER_SIZE:
            self.stats.decode_errors += 1
            return

        # Parse header
        header = FrameHeader.from_bytes(decoded)
        if header is None or header.magic != FRAME_MAGIC:
            self.stats.decode_errors += 1
            return

        # Extract payload
        payload = decoded[FRAME_HEADER_SIZE:FRAME_HEADER_SIZE + header.length]

        # Verify CRC - matches firmware algorithm
        header_for_crc = decoded[:10] + b'\x00\x00\x00\x00'
        computed_crc = crc32c(header_for_crc + payload)

        if computed_crc != header.crc32:
            self.stats.crc_errors += 1
            return

        self.stats.frames_valid += 1

        # Handle message by type
        if header.msg_type == ControlMsgType.HELLO:
            self._handle_hello(header.seq)
        elif header.msg_type == ControlMsgType.SET_MACRO:
            self._handle_set_macro(payload)

    def _handle_hello(self, seq: int):
        """Handle HELLO message from Pico2 - send HELLO_ACK to establish connection."""
        print("[Pico2Receiver] Received HELLO, sending HELLO_ACK")

        # Create HELLO_ACK payload
        hello_ack = HelloAck(
            block_id=0,
            credit_bytes=4096,
            max_payload=512,
            protocol_version=0x01,
        )

        # Build frame header
        header = FrameHeader(
            magic=FRAME_MAGIC,
            version=0x01,
            channel=0,  # Control channel
            msg_type=ControlMsgType.HELLO_ACK,
            flags=0,
            seq=seq,
            length=len(hello_ack.to_bytes()),
            crc32=0,  # Will be computed
        )

        # Serialize header (first 10 bytes) + placeholder CRC
        header_bytes = header.to_bytes()[:10] + b'\x00\x00\x00\x00'
        payload_bytes = hello_ack.to_bytes()

        # Compute CRC over header (with zero CRC field) + payload
        computed_crc = crc32c(header_bytes + payload_bytes)

        # Rebuild header with correct CRC
        header.crc32 = computed_crc
        frame_data = header.to_bytes() + payload_bytes

        # COBS encode and send
        encoded = cobs_encode(frame_data)
        if self._serial and self._serial.is_open:
            self._serial.write(encoded + b'\x00')

    def _handle_set_macro(self, payload: bytes):
        """Handle a SetMacro message."""
        if len(payload) < 8:
            return

        macro = SetMacro.from_bytes(payload)
        self.stats.macro_updates += 1

        # Update state
        with self._lock:
            if 0 <= macro.macro_id < NUM_MACROS:
                state = self._macros[macro.macro_id]
                state.value_q15 = macro.value_q15
                state.value_float = macro.value_q15 / Q15_MAX
                state.last_update = time.time()
                state.changed = True

        # Fire callbacks
        for callback in self._macro_callbacks:
            try:
                callback(macro.macro_id, macro.value_q15)
            except Exception as e:
                print(f"[Pico2Receiver] Callback error: {e}")

    # =========================================================================
    # UTILITY
    # =========================================================================

    def print_stats(self):
        """Print receiver statistics."""
        print(f"[Pico2Receiver] Stats:")
        print(f"  Bytes received: {self.stats.bytes_received}")
        print(f"  Frames received: {self.stats.frames_received}")
        print(f"  Frames valid: {self.stats.frames_valid}")
        print(f"  CRC errors: {self.stats.crc_errors}")
        print(f"  Decode errors: {self.stats.decode_errors}")
        print(f"  Macro updates: {self.stats.macro_updates}")

    def print_macros(self):
        """Print current macro values."""
        print(f"[Pico2Receiver] Macros:")
        with self._lock:
            for i, m in self._macros.items():
                print(f"  Macro {i}: {m.value_float:.3f} (Q15={m.value_q15})")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def find_pico2_ports() -> tuple[Optional[str], Optional[str]]:
    """
    Auto-detect Pico2 USB CDC ports (dual CDC device).

    The Pico2 presents two virtual COM ports:
    - Control channel (CDC 0): Used for commands and SetMacro messages
    - Data channel (CDC 1): Used for streaming voice data

    Returns:
        Tuple of (control_port, data_port), either may be None if not found
    """
    try:
        from serial.tools import list_ports

        # Find all Pico2 ports
        pico_ports = []
        for port in list_ports.comports():
            # Look for Raspberry Pi Pico (VID 0x2E8A)
            if "2E8A" in (port.hwid or ""):
                pico_ports.append(port.device)
            elif "Pico" in port.description:
                pico_ports.append(port.device)
            elif "Symcrash" in port.description:
                pico_ports.append(port.device)

        if not pico_ports:
            return None, None

        # Sort ports - typically lower COM number is CDC 0 (control)
        pico_ports.sort()

        if len(pico_ports) >= 2:
            # Dual CDC: first is control, second is data
            return pico_ports[0], pico_ports[1]
        else:
            # Single port found (maybe only one interface or old firmware)
            return pico_ports[0], None

    except ImportError:
        return None, None


def find_pico2_port() -> Optional[str]:
    """
    Auto-detect Pico2 USB CDC control port.

    For dual CDC devices, returns the control channel port (CDC 0).

    Returns:
        Port name if found, None otherwise
    """
    control_port, _ = find_pico2_ports()
    return control_port


def create_receiver(port: str = None) -> Pico2Receiver:
    """
    Create and optionally auto-configure a Pico2Receiver.

    Args:
        port: Serial port, or None to auto-detect

    Returns:
        Configured Pico2Receiver
    """
    if port is None:
        control_port, data_port = find_pico2_ports()
        if control_port:
            print(f"[Pico2Receiver] Auto-detected control port: {control_port}")
            if data_port:
                print(f"[Pico2Receiver] Auto-detected data port: {data_port}")
            port = control_port
        else:
            print("[Pico2Receiver] No Pico2 detected, port must be specified manually")

    return Pico2Receiver(port=port)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    port = sys.argv[1] if len(sys.argv) > 1 else None

    if port is None:
        port = find_pico2_port()
        if port is None:
            print("Usage: python pico2_receiver.py <PORT>")
            print("       or auto-detect if Pico2 is connected")
            sys.exit(1)

    print(f"Connecting to {port}...")

    receiver = Pico2Receiver(port=port)

    # Add callback to print macro changes
    def on_macro_change(macro_id: int, value_q15: int):
        value_float = value_q15 / Q15_MAX
        print(f"  Macro {macro_id}: {value_float:.3f} (Q15={value_q15})")

    receiver.add_macro_callback(on_macro_change)

    if not receiver.start():
        print("Failed to start receiver")
        sys.exit(1)

    print("Listening for macro changes (Ctrl+C to exit)...")
    print("-" * 50)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n" + "-" * 50)
        receiver.print_stats()
        receiver.print_macros()
    finally:
        receiver.stop()
