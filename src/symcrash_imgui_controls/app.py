"""
Symcrash ImGui Controls Application.

Main application window with knobs, buttons, and communication.

Supports two connection modes:
1. UART mode: Connect to hardware via COM port (sends WSE-STD-100 messages)
2. Bridge mode: Connect to pico2_live_synth instance (injects into receiver)
"""

from typing import Optional, Union
from imgui_bundle import imgui, hello_imgui

from .state import (
    ControlState,
    MACRO_LABELS,
    NOTE_LABELS,
    MACRO_ATTACK,
    MACRO_CUTOFF,
    MACRO_HARMONICS,
)
from .controls import (
    render_knob_row,
    render_note_row,
    render_preset_combo,
    render_tree_combo,
    render_connection_status,
)
from .uart_sender import UARTSender, find_available_ports
from .synth_bridge import ImGuiSynthBridge

# Global app instance for external access
_app_instance: Optional['SymcrashImGuiApp'] = None


def get_app() -> Optional['SymcrashImGuiApp']:
    """Get the global app instance."""
    return _app_instance


class SymcrashImGuiApp:
    """
    Main ImGui application for synth control.

    Provides:
    - 10 knobs (ADSR, Filter, Oscillator)
    - 4 note buttons
    - Preset/tree selection
    - UART or bridge connection

    Connection modes:
    - UART: Send WSE-STD-100 messages to hardware (like pico2_controls)
    - Bridge: Inject macros into pico2_live_synth via Pico2Receiver
    """

    def __init__(self):
        global _app_instance
        _app_instance = self

        self.state = ControlState()

        # Connection mode (mutually exclusive)
        self.sender: Optional[UARTSender] = None      # UART mode
        self.bridge: Optional[ImGuiSynthBridge] = None  # Bridge mode

        # Port selection state
        self._port_list = []
        self._selected_port_idx = 0

    # =========================================================================
    # UART MODE - Connect to hardware via serial port
    # =========================================================================

    def connect_uart(self, port: str) -> bool:
        """
        Connect to hardware via UART.

        This sends WSE-STD-100 messages over serial, just like the
        Pico2 hardware controller does.

        Args:
            port: Serial port (e.g., "COM3")

        Returns:
            True if connected successfully
        """
        # Disconnect any existing connection
        self.disconnect()

        self.sender = UARTSender(port)
        if self.sender.connect():
            self.state.connected = True
            self.state.port = port
            print(f"[ImGuiApp] Connected to hardware on {port}")
            return True
        else:
            self.sender = None
            print(f"[ImGuiApp] Failed to connect to {port}")
            return False

    # =========================================================================
    # BRIDGE MODE - Connect to pico2_live_synth instance
    # =========================================================================

    def connect_bridge(self, pico2_bridge) -> bool:
        """
        Connect to a running pico2_live_synth via its Pico2SynthBridge.

        This injects macros into the Pico2Receiver's callback chain,
        following the same control path as hardware messages.

        Args:
            pico2_bridge: Pico2SynthBridge instance from pico2_live_synth

        Returns:
            True if connected successfully
        """
        # Disconnect any existing connection
        self.disconnect()

        self.bridge = ImGuiSynthBridge(pico2_bridge)
        if self.bridge.is_connected():
            self.state.connected = True
            self.state.port = "Bridge"

            # Sync state from bridge
            self.state.macro_values = self.bridge.sync_state_from_bridge()
            print("[ImGuiApp] Connected to pico2_live_synth bridge")
            return True
        else:
            self.bridge = None
            print("[ImGuiApp] Failed to connect to bridge")
            return False

    def set_bridge(self, bridge: ImGuiSynthBridge) -> None:
        """Set an existing ImGuiSynthBridge."""
        self.disconnect()
        self.bridge = bridge
        self.state.connected = bridge.is_connected()
        self.state.port = "Bridge" if self.state.connected else ""

        if self.state.connected:
            self.state.macro_values = bridge.sync_state_from_bridge()

    # =========================================================================
    # COMMON
    # =========================================================================

    def disconnect(self) -> None:
        """Disconnect from current connection."""
        if self.sender:
            self.sender.disconnect()
            self.sender = None
        if self.bridge:
            self.bridge = None

        self.state.connected = False
        self.state.port = ""

    def _send_macro(self, macro_id: int, value_q15: int) -> bool:
        """Send macro to current connection (UART or bridge)."""
        if self.sender:
            return self.sender.send_macro(macro_id, value_q15)
        elif self.bridge:
            return self.bridge.send_macro(macro_id, value_q15)
        return False

    def _send_note(self, note: int, velocity: int, on: bool) -> bool:
        """Send note to current connection (UART or bridge)."""
        if self.sender:
            return self.sender.send_note(note, velocity, on)
        elif self.bridge:
            return self.bridge.send_note(note, velocity, on)
        return False

    def _send_tree_target(self, tree_index: int) -> bool:
        """Send tree target to current connection."""
        if self.sender:
            return self.sender.send_tree_target(tree_index)
        # Bridge doesn't support tree target yet
        return False

    # =========================================================================
    # RENDER
    # =========================================================================

    def render(self) -> None:
        """Main render callback - called each frame."""
        imgui.begin("Symcrash Controls")

        self._render_connection_section()
        imgui.separator()

        self._render_adsr_section()
        imgui.separator()

        self._render_filter_section()
        imgui.separator()

        self._render_osc_section()
        imgui.separator()

        self._render_notes_section()
        imgui.separator()

        self._render_status_section()

        imgui.end()

    def _render_connection_section(self) -> None:
        """Render connection controls."""
        imgui.text("Connection")

        # Refresh ports button
        if imgui.button("Refresh"):
            self._port_list = find_available_ports()

        imgui.same_line()

        # Port combo
        if not self._port_list:
            self._port_list = find_available_ports()

        if self._port_list:
            imgui.set_next_item_width(150)
            changed, self._selected_port_idx = imgui.combo(
                "##port",
                self._selected_port_idx,
                self._port_list,
            )

        imgui.same_line()

        # Connect/Disconnect button
        if self.state.connected:
            if imgui.button("Disconnect"):
                self.disconnect()
        else:
            if imgui.button("Connect") and self._port_list:
                port = self._port_list[self._selected_port_idx]
                self.connect_uart(port)

        imgui.same_line()
        render_connection_status(self.state.connected, self.state.port)

    def _render_adsr_section(self) -> None:
        """Render ADSR knobs (macros 0-3)."""
        imgui.text("ADSR")

        labels = MACRO_LABELS[MACRO_ATTACK:MACRO_ATTACK + 4]
        values = self.state.macro_values[MACRO_ATTACK:MACRO_ATTACK + 4]

        changed, new_values, changed_idx = render_knob_row(labels, values)

        if changed:
            # Update state
            for i, val in enumerate(new_values):
                self.state.macro_values[MACRO_ATTACK + i] = val

            # Send to connection
            if changed_idx >= 0:
                macro_id = MACRO_ATTACK + changed_idx
                self._send_macro(macro_id, new_values[changed_idx])

    def _render_filter_section(self) -> None:
        """Render Filter knobs (macros 4-5)."""
        imgui.text("Filter")

        labels = MACRO_LABELS[MACRO_CUTOFF:MACRO_CUTOFF + 2]
        values = self.state.macro_values[MACRO_CUTOFF:MACRO_CUTOFF + 2]

        changed, new_values, changed_idx = render_knob_row(labels, values)

        if changed:
            for i, val in enumerate(new_values):
                self.state.macro_values[MACRO_CUTOFF + i] = val

            if changed_idx >= 0:
                macro_id = MACRO_CUTOFF + changed_idx
                self._send_macro(macro_id, new_values[changed_idx])

    def _render_osc_section(self) -> None:
        """Render Oscillator knobs (macros 6-9)."""
        imgui.text("Oscillator")

        labels = MACRO_LABELS[MACRO_HARMONICS:MACRO_HARMONICS + 4]
        values = self.state.macro_values[MACRO_HARMONICS:MACRO_HARMONICS + 4]

        changed, new_values, changed_idx = render_knob_row(labels, values)

        if changed:
            for i, val in enumerate(new_values):
                self.state.macro_values[MACRO_HARMONICS + i] = val

            if changed_idx >= 0:
                macro_id = MACRO_HARMONICS + changed_idx
                self._send_macro(macro_id, new_values[changed_idx])

    def _render_notes_section(self) -> None:
        """Render note buttons and preset/tree selectors."""
        imgui.text("Notes")

        # Note buttons
        any_clicked, clicked_idx = render_note_row(
            NOTE_LABELS,
            self.state.note_states,
        )

        if any_clicked:
            # Toggle note state
            new_state = self.state.toggle_note(clicked_idx)
            note = self.state.get_note_number(clicked_idx)

            # Send to connection
            velocity = 100 if new_state else 0
            self._send_note(note, velocity, new_state)

        imgui.same_line()
        imgui.spacing()
        imgui.same_line()

        # Preset selector
        changed, new_preset = render_preset_combo(self.state.current_preset)
        if changed:
            self.state.current_preset = new_preset

        imgui.same_line()

        # Tree selector
        changed, new_tree = render_tree_combo(self.state.current_tree)
        if changed:
            self.state.current_tree = new_tree
            self._send_tree_target(new_tree)

    def _render_status_section(self) -> None:
        """Render status bar."""
        imgui.text("Values:")
        imgui.same_line()

        for i, (label, value) in enumerate(zip(MACRO_LABELS, self.state.macro_values)):
            if i > 0:
                imgui.same_line()
            pct = int((value / 32767) * 100)
            imgui.text(f"{label}:{pct}%")


def main():
    """Run the ImGui application standalone (UART mode)."""
    app = SymcrashImGuiApp()

    # Configure runner
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Symcrash Controls"
    runner_params.app_window_params.window_geometry.size = (800, 500)
    runner_params.callbacks.show_gui = app.render

    # Run
    hello_imgui.run(runner_params)


if __name__ == "__main__":
    main()
