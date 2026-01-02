"""
UART Sender - WSE-STD-100 Protocol for ImGui Controls.

Sends control messages via UART, same protocol as pico2_controls.
"""

from typing import Optional
import serial
import serial.tools.list_ports

from symcrash_python_ap.comm.dual_uart_link import (
    DualUARTLink,
    SetMacro,
    NoteEvent,
    SetTreeTarget,
)


def find_available_ports() -> list:
    """List available serial ports."""
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


class UARTSender:
    """
    Sends control messages via UART using WSE-STD-100 protocol.

    This mirrors the Pico2 hardware controller's communication,
    allowing the ImGui interface to control the synth engine
    identically to the hardware.
    """

    def __init__(self, port: Optional[str] = None):
        """
        Initialize UART sender.

        Args:
            port: Serial port (e.g., "COM3" or "/dev/ttyUSB0")
        """
        self.port = port
        self.link: Optional[DualUARTLink] = None
        self._connected = False

    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to the synth engine via UART.

        Args:
            port: Serial port (uses stored port if None)

        Returns:
            True if connected successfully
        """
        if port:
            self.port = port

        if not self.port:
            print("[UARTSender] No port specified")
            return False

        try:
            self.link = DualUARTLink(c_uart_port=self.port)
            success = self.link.connect()  # connect() not open()
            if success:
                self.link.start()  # Start RX threads
                self._connected = True
                print(f"[UARTSender] Connected to {self.port}")
            else:
                self._connected = False
                print(f"[UARTSender] Failed to connect to {self.port}")
            return success
        except Exception as e:
            print(f"[UARTSender] Connection error: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from UART."""
        if self.link:
            self.link.close()
            self.link = None
        self._connected = False
        print("[UARTSender] Disconnected")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self.link is not None

    def send_macro(self, macro_id: int, value_q15: int) -> bool:
        """
        Send SetMacro message.

        Args:
            macro_id: Macro index (0-9)
            value_q15: Q15 value (0-32767)

        Returns:
            True if sent successfully
        """
        if not self.is_connected():
            return False

        try:
            msg = SetMacro(
                apply_block_id=0,  # Immediate
                macro_id=macro_id,
                value_q15=value_q15 & 0x7FFF,  # Clamp to Q15 range
            )
            self.link.send_set_macro(msg)
            return True
        except Exception as e:
            print(f"[UARTSender] send_macro error: {e}")
            return False

    def send_note(self, note: int, velocity: int, on: bool) -> bool:
        """
        Send NoteEvent message.

        Args:
            note: MIDI note number (0-127)
            velocity: Velocity (0-127)
            on: True for note-on, False for note-off

        Returns:
            True if sent successfully
        """
        if not self.is_connected():
            return False

        try:
            msg = NoteEvent(
                apply_block_id=0,  # Immediate
                note=note & 0x7F,
                vel=velocity & 0x7F,
                on=on,
                chan=0,
                flags=0,
            )
            self.link.send_note_event(msg)
            return True
        except Exception as e:
            print(f"[UARTSender] send_note error: {e}")
            return False

    def send_tree_target(self, tree_index: int, tau_ms: int = 1000) -> bool:
        """
        Send SetTreeTarget message.

        Args:
            tree_index: Tree index (0-4)
            tau_ms: Morph time in milliseconds

        Returns:
            True if sent successfully
        """
        if not self.is_connected():
            return False

        try:
            msg = SetTreeTarget(
                apply_block_id=0,
                tree_index=tree_index & 0x07,
                tau_ms=tau_ms,
                curve=0,  # Default curve
            )
            self.link.send_set_tree_target(msg)
            return True
        except Exception as e:
            print(f"[UARTSender] send_tree_target error: {e}")
            return False


# Convenience function for testing
def create_sender(port: Optional[str] = None) -> UARTSender:
    """Create and optionally connect a UARTSender."""
    sender = UARTSender(port)
    if port:
        sender.connect()
    return sender
