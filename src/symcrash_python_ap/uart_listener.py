#!/usr/bin/env python3
"""
WSE-STD-020 UART Listener CLI
=============================

Interactive command-line tool for testing controllers over UART.

Usage:
    python uart_listener.py [OPTIONS]

Examples:
    # List available serial ports
    python uart_listener.py --list

    # Connect to a port and listen
    python uart_listener.py --port COM3 --baud 921600

    # Run in mock mode (no hardware)
    python uart_listener.py --mock

    # Send test frames
    python uart_listener.py --port COM3 --test-send
"""

import argparse
import sys
import time
import threading
import struct
from typing import Optional, List, Dict
from dataclasses import dataclass

from uart_protocol import (
    UARTListener, UARTPacket, MockUART,
    VoiceFrameV0, VoiceFrameDelta, BackboneEntry, Payload64,
    MsgType, EventLane, PayloadMode, TargetKind,
    AckStatus, ErrorCode, FieldMask, VoiceLatch,
    crc32c, SYNC_WORD, NUM_EVENT_LANES,
    run_conformance_test
)
from fixed import Q15_ONE, Q15_HALF


# =============================================================================
# CONSOLE COLORS
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def color(text: str, c: str) -> str:
    """Apply color to text."""
    return f"{c}{text}{Colors.RESET}"


# =============================================================================
# PACKET FORMATTER
# =============================================================================

def format_packet(packet: UARTPacket, direction: str = "RX") -> str:
    """Format packet for display."""
    lines = []

    # Header
    dir_color = Colors.GREEN if direction == "RX" else Colors.CYAN
    lines.append(color(f"[{direction}] ", dir_color) +
                 color(f"Seq={packet.seq:04X} ", Colors.YELLOW) +
                 color(f"Type={packet.msg_type.name}", Colors.MAGENTA))

    # Payload details
    if packet.msg_type == MsgType.VF_ABSOLUTE:
        try:
            frame = VoiceFrameV0.from_bytes(packet.payload)
            lines.append(f"  Voice: {frame.voice_id}, Block: {frame.block_index}")
            lines.append(f"  Regime: {frame.regime_id}, Gear: {frame.gear_q15}")
            lines.append(f"  Trees: A={frame.tree_id_a}, B={frame.tree_id_b}, Blend={frame.tree_blend_q15}")
            lines.append(f"  Gain: {frame.gain_q15}, Traction: {frame.traction_q15}")
            lines.append(f"  Backbone: {len(frame.backbone)} entries")
            lines.append(f"  Events: mask=0x{frame.event_mask:02X}")
            if frame.event_mask:
                for i in range(NUM_EVENT_LANES):
                    if frame.event_mask & (1 << i):
                        lines.append(f"    Lane {i} ({EventLane(i).name}): strength={frame.event_strength_q15[i]}")
        except Exception as e:
            lines.append(f"  [Parse error: {e}]")

    elif packet.msg_type == MsgType.VF_DELTA:
        try:
            voice_id, block_idx, field_mask, _ = struct.unpack('<HHII', packet.payload[:12])
            lines.append(f"  Voice: {voice_id}, Block: {block_idx}")
            lines.append(f"  Field mask: 0x{field_mask:08X}")
            for fm in FieldMask:
                if field_mask & (1 << fm):
                    lines.append(f"    {fm.name}")
        except Exception as e:
            lines.append(f"  [Parse error: {e}]")

    elif packet.msg_type == MsgType.PARAM_ABSOLUTE:
        try:
            reg_id, value = struct.unpack('<Hi', packet.payload[:6])
            lines.append(f"  Register: {reg_id}, Value: {value}")
        except Exception as e:
            lines.append(f"  [Parse error: {e}]")

    elif packet.msg_type == MsgType.ACK:
        try:
            seq_acked, status, detail = struct.unpack('<HBB', packet.payload)
            lines.append(f"  Acked seq: {seq_acked:04X}, Status: {AckStatus(status).name}")
        except Exception as e:
            lines.append(f"  [Parse error: {e}]")

    elif packet.msg_type == MsgType.NACK:
        try:
            seq_nacked, error, detail = struct.unpack('<HBB', packet.payload)
            lines.append(f"  Nacked seq: {seq_nacked:04X}, Error: {ErrorCode(error).name}")
        except Exception as e:
            lines.append(f"  [Parse error: {e}]")

    else:
        lines.append(f"  Payload: {packet.payload.hex()[:64]}...")

    return "\n".join(lines)


# =============================================================================
# INTERACTIVE LISTENER
# =============================================================================

class InteractiveListener:
    """Interactive UART listener with command prompt."""

    def __init__(self, listener: Optional[UARTListener] = None, mock: Optional[MockUART] = None):
        self.listener = listener
        self.mock = mock
        self.running = False
        self.verbose = True
        self.block_index = 0

        # Voice state tracking
        self.voice_latches: Dict[int, VoiceLatch] = {}

        # History
        self.packet_history: List[UARTPacket] = []

    def start(self):
        """Start the interactive listener."""
        self.running = True

        if self.listener:
            self.listener.start()
            print(color(f"Connected to {self.listener.port} at {self.listener.baudrate} baud", Colors.GREEN))

        print(color("\nWSE-STD-020 UART Listener", Colors.BOLD))
        print("Type 'help' for commands, 'quit' to exit\n")

        # Start packet display thread
        display_thread = threading.Thread(target=self._display_loop, daemon=True)
        display_thread.start()

        # Command loop
        try:
            while self.running:
                try:
                    cmd = input(color("> ", Colors.CYAN)).strip()
                    if cmd:
                        self._handle_command(cmd)
                except EOFError:
                    break
        except KeyboardInterrupt:
            pass

        self.stop()

    def stop(self):
        """Stop the listener."""
        self.running = False
        if self.listener:
            self.listener.stop()
            self.listener.disconnect()
        print(color("\nDisconnected", Colors.YELLOW))

    def _display_loop(self):
        """Background thread to display incoming packets."""
        while self.running:
            if self.listener:
                packets = self.listener.process_messages(timeout=0.1)
                for packet in packets:
                    self.packet_history.append(packet)
                    if self.verbose:
                        print("\n" + format_packet(packet, "RX"))
                        print(color("> ", Colors.CYAN), end="", flush=True)

            elif self.mock:
                # Process mock device
                frame = self.mock.device_process()
                if frame:
                    print(f"\n[MOCK] Processed frame for voice {frame.voice_id}")

                # Check for ACKs
                ack = self.mock.host_receive(timeout=0.01)
                if ack:
                    self.packet_history.append(ack)
                    if self.verbose:
                        print("\n" + format_packet(ack, "RX"))
                        print(color("> ", Colors.CYAN), end="", flush=True)

            time.sleep(0.05)

    def _handle_command(self, cmd: str):
        """Handle a command."""
        parts = cmd.split()
        if not parts:
            return

        verb = parts[0].lower()

        if verb in ("quit", "exit", "q"):
            self.running = False

        elif verb == "help":
            self._cmd_help()

        elif verb == "status":
            self._cmd_status()

        elif verb == "verbose":
            self.verbose = not self.verbose
            print(f"Verbose: {self.verbose}")

        elif verb == "send":
            self._cmd_send(parts[1:])

        elif verb == "frame":
            self._cmd_frame(parts[1:])

        elif verb == "event":
            self._cmd_event(parts[1:])

        elif verb == "param":
            self._cmd_param(parts[1:])

        elif verb == "history":
            self._cmd_history(parts[1:])

        elif verb == "test":
            self._cmd_test(parts[1:])

        elif verb == "mock":
            self._cmd_mock(parts[1:])

        else:
            print(color(f"Unknown command: {verb}", Colors.RED))

    def _cmd_help(self):
        """Print help."""
        print("""
Commands:
  help          - Show this help
  quit          - Exit the listener
  status        - Show connection status
  verbose       - Toggle verbose output

  send <hex>    - Send raw bytes (hex string)
  frame <voice> [options] - Send VoiceFrame
  event <voice> <lane> <strength> - Send event
  param <reg> <value> - Send parameter

  history [n]   - Show last n packets
  test <name>   - Run test (conformance, roundtrip)
  mock <cmd>    - Mock commands (advance, status)

Frame options:
  --tree-a N    - Primary tree ID
  --tree-b N    - Secondary tree ID
  --blend N     - Tree blend (0-32767)
  --gain N      - Gain (0-32767)
  --regime N    - Regime ID
  --gear N      - Gear (0-32767)

Event lanes:
  0=OP_FIRE, 1=OP_TOGGLE, 2=INJECT, 3=MATERIAL,
  4=TREE, 5=CLOUD, 6=VOICE, 7=DEBUG
""")

    def _cmd_status(self):
        """Show status."""
        if self.listener:
            print(f"Port: {self.listener.port}")
            print(f"Baud: {self.listener.baudrate}")
            print(f"Stats: {self.listener.stats}")
        elif self.mock:
            print("Mode: Mock (no hardware)")
            print(f"Current block: {self.mock.current_block}")
            print(f"Device latches: {len(self.mock.device_latches)}")
        else:
            print("Not connected")

    def _cmd_send(self, args: List[str]):
        """Send raw bytes."""
        if not args:
            print("Usage: send <hex bytes>")
            return

        try:
            data = bytes.fromhex("".join(args))
            if self.listener and self.listener.serial:
                self.listener.serial.write(data)
                print(f"Sent {len(data)} bytes")
            else:
                print("Not connected")
        except ValueError as e:
            print(f"Invalid hex: {e}")

    def _cmd_frame(self, args: List[str]):
        """Send VoiceFrame."""
        if not args:
            print("Usage: frame <voice_id> [options]")
            return

        voice_id = int(args[0])
        self.block_index += 1

        # Parse options
        frame = VoiceFrameV0(
            voice_id=voice_id,
            block_index=self.block_index,
            gain_q15=0x4000,
            traction_q15=Q15_ONE,
        )

        i = 1
        while i < len(args):
            opt = args[i]
            if opt == "--tree-a" and i + 1 < len(args):
                frame.tree_id_a = int(args[i + 1])
                i += 2
            elif opt == "--tree-b" and i + 1 < len(args):
                frame.tree_id_b = int(args[i + 1])
                i += 2
            elif opt == "--blend" and i + 1 < len(args):
                frame.tree_blend_q15 = int(args[i + 1])
                i += 2
            elif opt == "--gain" and i + 1 < len(args):
                frame.gain_q15 = int(args[i + 1])
                i += 2
            elif opt == "--regime" and i + 1 < len(args):
                frame.regime_id = int(args[i + 1])
                i += 2
            elif opt == "--gear" and i + 1 < len(args):
                frame.gear_q15 = int(args[i + 1])
                i += 2
            else:
                print(f"Unknown option: {opt}")
                i += 1

        # Send
        self._send_frame(frame)

    def _cmd_event(self, args: List[str]):
        """Send event."""
        if len(args) < 3:
            print("Usage: event <voice_id> <lane> <strength>")
            return

        voice_id = int(args[0])
        lane = int(args[1])
        strength = int(args[2])

        self.block_index += 1

        frame = VoiceFrameV0(
            voice_id=voice_id,
            block_index=self.block_index,
            event_mask=(1 << lane),
            event_strength_q15=[0] * NUM_EVENT_LANES,
            gain_q15=0x4000,
        )
        frame.event_strength_q15[lane] = strength

        self._send_frame(frame)
        print(f"Sent event: lane={EventLane(lane).name}, strength={strength}")

    def _cmd_param(self, args: List[str]):
        """Send parameter."""
        if len(args) < 2:
            print("Usage: param <reg_id> <value>")
            return

        reg_id = int(args[0])
        value = int(args[1])

        if self.listener:
            self.listener.send_param(reg_id, value)
            print(f"Sent param: reg={reg_id}, value={value}")
        elif self.mock:
            payload = struct.pack('<Hi', reg_id, value)
            packet = UARTPacket(msg_type=MsgType.PARAM_ABSOLUTE, payload=payload)
            self.mock.host_send(packet)
            print(f"[MOCK] Sent param: reg={reg_id}, value={value}")

    def _cmd_history(self, args: List[str]):
        """Show packet history."""
        n = int(args[0]) if args else 10
        packets = self.packet_history[-n:]

        for i, packet in enumerate(packets):
            print(f"\n[{i}] " + format_packet(packet, "RX"))

    def _cmd_test(self, args: List[str]):
        """Run tests."""
        if not args:
            print("Usage: test <conformance|roundtrip|stress>")
            return

        test_name = args[0].lower()

        if test_name == "conformance":
            run_conformance_test()

        elif test_name == "roundtrip":
            self._test_roundtrip()

        elif test_name == "stress":
            self._test_stress(int(args[1]) if len(args) > 1 else 100)

        else:
            print(f"Unknown test: {test_name}")

    def _cmd_mock(self, args: List[str]):
        """Mock commands."""
        if not self.mock:
            print("Not in mock mode")
            return

        if not args:
            print("Usage: mock <advance|status>")
            return

        cmd = args[0].lower()

        if cmd == "advance":
            n = int(args[1]) if len(args) > 1 else 1
            for _ in range(n):
                frames = self.mock.advance_block()
                if frames:
                    print(f"Block {self.mock.current_block}: {len(frames)} frames due")
            print(f"Current block: {self.mock.current_block}")

        elif cmd == "status":
            print(f"Current block: {self.mock.current_block}")
            print(f"Device latches: {len(self.mock.device_latches)}")
            for vid, latch in self.mock.device_latches.items():
                print(f"  Voice {vid}: tree_a={latch.tree_id_a}, gain={latch.gain_q15}")

    def _send_frame(self, frame: VoiceFrameV0):
        """Send a VoiceFrame."""
        if self.listener:
            self.listener.send_voice_frame(frame)
            print(f"Sent frame: voice={frame.voice_id}, block={frame.block_index}")

        elif self.mock:
            packet = UARTPacket(
                msg_type=MsgType.VF_ABSOLUTE,
                payload=frame.to_bytes()
            )
            self.mock.host_send(packet)
            print(f"[MOCK] Sent frame: voice={frame.voice_id}, block={frame.block_index}")

    def _test_roundtrip(self):
        """Test frame round-trip."""
        print("\nRunning round-trip test...")

        frame = VoiceFrameV0(
            voice_id=0,
            block_index=1,
            regime_id=2,
            gear_q15=0x4000,
            tree_id_a=1,
            tree_id_b=2,
            tree_blend_q15=0x2000,
            gain_q15=0x5000,
            traction_q15=0x7000,
            backbone=[BackboneEntry(op_id=1, strength_q15=0x3000)],
            event_mask=0x05,
            event_strength_q15=[0x2000, 0, 0x3000, 0, 0, 0, 0, 0],
        )

        self._send_frame(frame)

        # Wait for ACK
        time.sleep(0.5)

        if self.mock:
            self.mock.device_process()
            ack = self.mock.host_receive(timeout=0.5)
            if ack and ack.msg_type == MsgType.ACK:
                print(color("Round-trip test PASSED", Colors.GREEN))
            else:
                print(color("Round-trip test FAILED (no ACK)", Colors.RED))
        else:
            print("Sent frame, check for ACK...")

    def _test_stress(self, count: int):
        """Stress test with many frames."""
        print(f"\nRunning stress test ({count} frames)...")

        start = time.time()

        for i in range(count):
            frame = VoiceFrameV0(
                voice_id=i % 8,
                block_index=self.block_index + i,
                gain_q15=0x4000 + (i * 10) % 0x3000,
                event_mask=i % 256,
            )
            self._send_frame(frame)

        elapsed = time.time() - start
        rate = count / elapsed

        print(f"Sent {count} frames in {elapsed:.2f}s ({rate:.1f} frames/sec)")


# =============================================================================
# PORT LISTING
# =============================================================================

def list_serial_ports():
    """List available serial ports."""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        if ports:
            print("Available serial ports:")
            for port in ports:
                print(f"  {port.device}: {port.description}")
        else:
            print("No serial ports found")
    except ImportError:
        print("Install pyserial to list ports: pip install pyserial")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WSE-STD-020 UART Listener for Controller Testing"
    )

    parser.add_argument("--port", "-p", help="Serial port (e.g., COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", "-b", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--list", "-l", action="store_true", help="List available serial ports")
    parser.add_argument("--mock", "-m", action="store_true", help="Run in mock mode (no hardware)")
    parser.add_argument("--test", "-t", action="store_true", help="Run conformance test and exit")
    parser.add_argument("--test-send", action="store_true", help="Send test frames")

    args = parser.parse_args()

    if args.list:
        list_serial_ports()
        return 0

    if args.test:
        run_conformance_test()
        return 0

    if args.mock:
        print(color("Starting in mock mode (no hardware)", Colors.YELLOW))
        mock = MockUART()
        interactive = InteractiveListener(mock=mock)
        interactive.start()
        return 0

    if args.port:
        listener = UARTListener(port=args.port, baudrate=args.baud)
        if listener.connect():
            if args.test_send:
                # Send test frames
                print("Sending test frames...")
                for i in range(5):
                    frame = VoiceFrameV0(
                        voice_id=i,
                        block_index=i,
                        gain_q15=0x4000 + i * 0x1000,
                    )
                    listener.send_voice_frame(frame)
                    time.sleep(0.1)
                print("Done sending test frames")
                listener.disconnect()
            else:
                interactive = InteractiveListener(listener=listener)
                interactive.start()
        else:
            print(color(f"Failed to connect to {args.port}", Colors.RED))
            return 1
    else:
        print("Usage: uart_listener.py --port <PORT> or --mock")
        print("Use --list to see available ports")
        print("Use --test to run conformance test")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
