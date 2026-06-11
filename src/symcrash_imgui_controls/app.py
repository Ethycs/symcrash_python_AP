"""
Symcrash Sound Design Workbench.

ImGui application for preset browsing, character control tweaking,
and render-and-play iteration.

Also supports hardware connection modes:
1. UART mode: Connect to hardware via COM port (sends WSE-STD-100 messages)
2. Bridge mode: Connect to pico2_live_synth instance (injects into receiver)
"""

import threading
from pathlib import Path
from typing import Optional, Union

from imgui_bundle import imgui, hello_imgui

from .state import (
    ControlState,
    MACRO_LABELS,
    NOTE_LABELS,
    MACRO_ATTACK,
    MACRO_CUTOFF,
    MACRO_HARMONICS,
    CHARACTER_LABELS,
    CHARACTER_FIELDS,
    PLAYBACK_NOTES,
)
from .controls import (
    render_knob_row,
    render_note_row,
    render_preset_combo,
    render_tree_combo,
    render_connection_status,
    render_character_knob_row,
    render_preset_file_combo,
)
from .uart_sender import UARTSender, find_available_ports
from .synth_bridge import ImGuiSynthBridge

# Default presets directory (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESETS_DIR = _PROJECT_ROOT / "presets" / "instruments"

# Global app instance for external access
_app_instance: Optional['SymcrashImGuiApp'] = None


def get_app() -> Optional['SymcrashImGuiApp']:
    """Get the global app instance."""
    return _app_instance


class SymcrashImGuiApp:
    """
    Sound Design Workbench with ImGui controls.

    Provides:
    - Preset file browser with Load button
    - 6 character knobs (brightness, density, energy, risk, dream, slip)
    - Render & Play button for offline rendering with character overrides
    - 10 hardware knobs (ADSR, Filter, Oscillator) for live synth
    - 4 note buttons
    - UART or bridge connection for live synth mode

    Connection modes:
    - UART: Send WSE-STD-100 messages to hardware
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

        # Playback note combo index
        self._note_combo_idx = 2  # default C4

        # Scan presets on creation
        self._scan_presets()

    # =========================================================================
    # PRESET SCANNING
    # =========================================================================

    def _scan_presets(self) -> None:
        """Scan presets/instruments/ for YAML files."""
        self.state.available_presets = []
        if PRESETS_DIR.exists():
            yaml_files = sorted(PRESETS_DIR.glob("*.yaml"))
            self.state.available_presets = [f.stem for f in yaml_files]
        if not self.state.available_presets:
            self.state.available_presets = ["(no presets found)"]

    def _load_selected_preset(self) -> None:
        """Load the currently selected preset and populate character knobs."""
        idx = self.state.selected_preset_idx
        if idx < 0 or idx >= len(self.state.available_presets):
            return

        name = self.state.available_presets[idx]
        preset_path = PRESETS_DIR / f"{name}.yaml"
        if not preset_path.exists():
            self.state.render_status = f"File not found: {name}.yaml"
            return

        try:
            from symcrash_python_ap.instrument_ir import load_instrument
            ir = load_instrument(str(preset_path))

            self.state.loaded_preset_path = str(preset_path)
            self.state.loaded_preset_name = ir.meta.name or name

            # Populate character knobs from preset's ControlConfig
            if ir.control:
                self.state.character_brightness = ir.control.brightness
                self.state.character_density = ir.control.density
                self.state.character_energy = ir.control.energy
                self.state.character_risk = ir.control.risk
                self.state.character_dream = ir.control.dream
                self.state.character_slip = ir.control.slip

            self.state.render_status = f"Loaded: {self.state.loaded_preset_name}"
        except Exception as e:
            self.state.render_status = f"Load error: {e}"

    # =========================================================================
    # RENDER & PLAY
    # =========================================================================

    def _on_render_and_play(self) -> None:
        """Render preset with current character values and play audio."""
        if self.state.is_rendering:
            return
        if not self.state.loaded_preset_path:
            self.state.render_status = "No preset loaded"
            return

        def _do_render():
            self.state.is_rendering = True
            self.state.render_status = "Rendering..."
            try:
                import numpy as np
                import sounddevice as sd
                from .workbench_render import render_with_character, normalize_to_rms

                note = PLAYBACK_NOTES[self._note_combo_idx][0]
                samples, name = render_with_character(
                    self.state.loaded_preset_path,
                    note=note,
                    duration_sec=self.state.playback_duration,
                    brightness=self.state.character_brightness,
                    density=self.state.character_density,
                    energy=self.state.character_energy,
                    risk=self.state.character_risk,
                    dream=self.state.character_dream,
                    slip=self.state.character_slip,
                )
                samples = normalize_to_rms(samples)

                # Play via sounddevice
                audio_f32 = samples.astype(np.float32) / 32767.0
                sd.play(audio_f32, samplerate=48000)

                self.state.render_status = f"Playing: {name} ({PLAYBACK_NOTES[self._note_combo_idx][1]})"
            except Exception as e:
                self.state.render_status = f"Render error: {e}"
            finally:
                self.state.is_rendering = False

        threading.Thread(target=_do_render, daemon=True).start()

    # =========================================================================
    # UART MODE
    # =========================================================================

    def connect_uart(self, port: str) -> bool:
        """Connect to hardware via UART."""
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
    # BRIDGE MODE
    # =========================================================================

    def connect_bridge(self, pico2_bridge) -> bool:
        """Connect to a running pico2_live_synth via its Pico2SynthBridge."""
        self.disconnect()

        self.bridge = ImGuiSynthBridge(pico2_bridge)
        if self.bridge.is_connected():
            self.state.connected = True
            self.state.port = "Bridge"
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
        return False

    # =========================================================================
    # RENDER (ImGui frame callback)
    # =========================================================================

    def render(self) -> None:
        """Main render callback - called each frame."""
        imgui.begin("Symcrash Sound Design Workbench")

        # --- Workbench sections ---
        self._render_preset_section()
        imgui.separator()

        self._render_character_section()
        imgui.separator()

        self._render_playback_section()
        imgui.separator()

        # --- Live synth sections ---
        if imgui.collapsing_header("Live Synth Controls"):
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

    # =========================================================================
    # WORKBENCH UI SECTIONS
    # =========================================================================

    def _render_preset_section(self) -> None:
        """Render preset file browser."""
        imgui.text("Preset")
        imgui.same_line()

        changed, new_idx = render_preset_file_combo(
            self.state.available_presets,
            self.state.selected_preset_idx,
        )
        if changed:
            self.state.selected_preset_idx = new_idx

        imgui.same_line()
        if imgui.button("Load"):
            self._load_selected_preset()

        imgui.same_line()
        if imgui.button("Refresh Presets"):
            self._scan_presets()

        # Show loaded preset name
        if self.state.loaded_preset_name:
            imgui.same_line()
            imgui.text_colored(
                imgui.ImVec4(0.4, 0.8, 0.4, 1.0),
                self.state.loaded_preset_name,
            )

    def _render_character_section(self) -> None:
        """Render character control knobs."""
        imgui.text("Character")

        # Get current values from state
        values = [getattr(self.state, f) for f in CHARACTER_FIELDS]

        changed, new_values, changed_idx = render_character_knob_row(
            CHARACTER_LABELS, values
        )

        if changed:
            for i, val in enumerate(new_values):
                setattr(self.state, CHARACTER_FIELDS[i], val)

    def _render_playback_section(self) -> None:
        """Render playback controls (note, duration, render & play)."""
        # Note selector
        note_labels = [n[1] for n in PLAYBACK_NOTES]
        imgui.text("Note")
        imgui.same_line()
        imgui.set_next_item_width(80)
        changed, new_idx = imgui.combo("##play_note", self._note_combo_idx, note_labels)
        if changed:
            self._note_combo_idx = new_idx

        # Duration slider
        imgui.same_line()
        imgui.text("Duration")
        imgui.same_line()
        imgui.set_next_item_width(120)
        changed, new_dur = imgui.slider_float(
            "##duration", self.state.playback_duration, 0.5, 4.0, "%.1fs"
        )
        if changed:
            self.state.playback_duration = new_dur

        # Render & Play button
        imgui.same_line()
        if self.state.is_rendering:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.5, 0.5, 0.2, 1.0))
            imgui.button("Rendering...")
            imgui.pop_style_color(1)
        else:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.8, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.6, 0.9, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.4, 0.7, 1.0))
            if imgui.button("Render & Play"):
                self._on_render_and_play()
            imgui.pop_style_color(3)

        # Status line
        if self.state.render_status:
            imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), self.state.render_status)

    # =========================================================================
    # LIVE SYNTH UI SECTIONS
    # =========================================================================

    def _render_connection_section(self) -> None:
        """Render connection controls."""
        imgui.text("Connection")

        if imgui.button("Refresh"):
            self._port_list = find_available_ports()

        imgui.same_line()

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
            for i, val in enumerate(new_values):
                self.state.macro_values[MACRO_ATTACK + i] = val

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

        any_clicked, clicked_idx = render_note_row(
            NOTE_LABELS,
            self.state.note_states,
        )

        if any_clicked:
            new_state = self.state.toggle_note(clicked_idx)
            note = self.state.get_note_number(clicked_idx)

            velocity = 100 if new_state else 0
            self._send_note(note, velocity, new_state)

        imgui.same_line()
        imgui.spacing()
        imgui.same_line()

        changed, new_preset = render_preset_combo(self.state.current_preset)
        if changed:
            self.state.current_preset = new_preset

        imgui.same_line()

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
    """Run the Sound Design Workbench standalone."""
    app = SymcrashImGuiApp()

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Symcrash Sound Design Workbench"
    runner_params.app_window_params.window_geometry.size = (1050, 650)
    runner_params.callbacks.show_gui = app.render

    hello_imgui.run(runner_params)


if __name__ == "__main__":
    main()
