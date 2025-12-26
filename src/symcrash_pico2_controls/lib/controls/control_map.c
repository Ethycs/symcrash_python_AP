/**
 * @file control_map.c
 * @brief Physical-to-parameter control mapping implementation
 */

#include "control_map.h"
#include "dual_uart_link.h"
#include "mcp3008.h"
#include "gpio_buttons.h"
#include "rotary_encoder.h"
#include "config.h"

// =============================================================================
// Mapping Tables
// =============================================================================

// Knob ID -> Macro ID mapping
static macro_id_t g_knob_to_macro[NUM_POTS] = {
    0, 1, 2, 3, 4, 5  // Default: knob N -> macro N
};

// Button ID -> MIDI note mapping
static midi_note_t g_button_to_note[NUM_BUTTONS] = {
    60, 62, 64, 65   // C4, D4, E4, F4
};

// Button note states (for toggle mode)
static bool g_note_states[NUM_BUTTONS] = {false};

// Current preset
static uint8_t g_current_preset = 0;
#define NUM_PRESETS 8

// =============================================================================
// Callbacks
// =============================================================================

static void on_adc_change(adc_channel_t channel, adc_value_t value) {
    control_map_process_knob(channel, value);
}

static void on_button_event(uint8_t button_id, button_event_t event) {
    if (event == BTN_EVENT_PRESS) {
        control_map_process_button(button_id, true);
    } else if (event == BTN_EVENT_RELEASE) {
        control_map_process_button(button_id, false);
    }
}

static void on_encoder_rotate(uint8_t encoder_id, int8_t delta) {
    control_map_process_encoder(encoder_id, delta);
}

static void on_encoder_switch(uint8_t encoder_id, bool pressed) {
    control_map_process_encoder_switch(encoder_id, pressed);
}

// =============================================================================
// Initialization
// =============================================================================

void control_map_init(void) {
    // Register callbacks
    mcp3008_set_callback(on_adc_change);
    buttons_set_callback(on_button_event);
    encoder_set_callback(on_encoder_rotate);
    encoder_set_switch_callback(on_encoder_switch);

    // Reset state
    g_current_preset = 0;
    for (int i = 0; i < NUM_BUTTONS; i++) {
        g_note_states[i] = false;
    }
}

// =============================================================================
// Processing
// =============================================================================

void control_map_process_knob(uint8_t knob_id, uint16_t value) {
    if (knob_id >= NUM_POTS) return;

    macro_id_t macro = g_knob_to_macro[knob_id];

    // Convert 10-bit ADC to Q15
    q15_t q15_value = adc10_to_q15(value);

    // Send SetMacro message
    dual_uart_send_macro(macro, q15_value);
}

void control_map_process_button(uint8_t button_id, bool pressed) {
    if (button_id >= NUM_BUTTONS) return;

    midi_note_t note = g_button_to_note[button_id];

    if (pressed) {
        // Note on
        dual_uart_send_note(note, 100, true);
        g_note_states[button_id] = true;
    } else {
        // Note off
        dual_uart_send_note(note, 0, false);
        g_note_states[button_id] = false;
    }
}

void control_map_process_encoder(uint8_t encoder_id, int8_t delta) {
    (void)encoder_id;

    // Use encoder for preset selection
    if (delta > 0) {
        // Next preset
        if (g_current_preset < NUM_PRESETS - 1) {
            g_current_preset++;
            // TODO: Send preset change command
        }
    } else if (delta < 0) {
        // Previous preset
        if (g_current_preset > 0) {
            g_current_preset--;
            // TODO: Send preset change command
        }
    }
}

void control_map_process_encoder_switch(uint8_t encoder_id, bool pressed) {
    (void)encoder_id;

    if (pressed) {
        // Encoder switch pressed - could enter menu, confirm selection, etc.
        // For now, send a tree target change as an example
        dual_uart_send_tree_target(g_current_preset % 5, 2000);
    }
}

// =============================================================================
// Configuration
// =============================================================================

void control_map_set_knob_macro(uint8_t knob_id, macro_id_t macro_id) {
    if (knob_id < NUM_POTS && macro_id < MAX_MACROS) {
        g_knob_to_macro[knob_id] = macro_id;
    }
}

void control_map_set_button_note(uint8_t button_id, midi_note_t note) {
    if (button_id < NUM_BUTTONS && note < 128) {
        g_button_to_note[button_id] = note;
    }
}

// =============================================================================
// State
// =============================================================================

uint8_t control_map_get_preset(void) {
    return g_current_preset;
}

void control_map_set_preset(uint8_t preset) {
    if (preset < NUM_PRESETS) {
        g_current_preset = preset;
    }
}

uint8_t control_map_get_preset_count(void) {
    return NUM_PRESETS;
}
