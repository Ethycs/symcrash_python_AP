/**
 * @file control_map.h
 * @brief Physical-to-parameter control mapping
 */

#ifndef SYMCRASH_CONTROL_MAP_H
#define SYMCRASH_CONTROL_MAP_H

#include <stdint.h>
#include <stdbool.h>
#include "types.h"

// =============================================================================
// Types
// =============================================================================

/** Action types for controls */
typedef enum {
    ACTION_NONE,
    ACTION_SET_MACRO,       // Send SetMacro message
    ACTION_NOTE_ON,         // Send NoteEvent (on)
    ACTION_NOTE_OFF,        // Send NoteEvent (off)
    ACTION_NOTE_TOGGLE,     // Toggle note on/off
    ACTION_TREE_SELECT,     // Send SetTreeTarget
    ACTION_PRESET_NEXT,     // Next preset (local)
    ACTION_PRESET_PREV,     // Previous preset (local)
    ACTION_PARAM_INC,       // Increment parameter
    ACTION_PARAM_DEC        // Decrement parameter
} action_type_t;

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize control mapping system
 *
 * Sets up default mappings and registers callbacks with HAL.
 */
void control_map_init(void);

// =============================================================================
// Processing
// =============================================================================

/**
 * @brief Process knob value change
 *
 * Called by MCP3008 callback when a pot value changes.
 *
 * @param knob_id   Knob/pot ID (0-5)
 * @param value     New ADC value (0-1023)
 */
void control_map_process_knob(uint8_t knob_id, uint16_t value);

/**
 * @brief Process button event
 *
 * Called by buttons callback on press/release.
 *
 * @param button_id     Button ID (0-3)
 * @param pressed       true if pressed, false if released
 */
void control_map_process_button(uint8_t button_id, bool pressed);

/**
 * @brief Process encoder rotation
 *
 * Called by encoder callback on rotation.
 *
 * @param encoder_id    Encoder ID (0)
 * @param delta         Rotation delta (+1 = CW, -1 = CCW)
 */
void control_map_process_encoder(uint8_t encoder_id, int8_t delta);

/**
 * @brief Process encoder switch
 *
 * @param encoder_id    Encoder ID (0)
 * @param pressed       true if pressed
 */
void control_map_process_encoder_switch(uint8_t encoder_id, bool pressed);

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief Set knob-to-macro mapping
 *
 * @param knob_id   Knob ID (0-5)
 * @param macro_id  Macro ID to control (0-7)
 */
void control_map_set_knob_macro(uint8_t knob_id, macro_id_t macro_id);

/**
 * @brief Set button-to-note mapping
 *
 * @param button_id     Button ID (0-3)
 * @param note          MIDI note number
 */
void control_map_set_button_note(uint8_t button_id, midi_note_t note);

// =============================================================================
// State
// =============================================================================

/**
 * @brief Get current preset index
 */
uint8_t control_map_get_preset(void);

/**
 * @brief Set current preset index
 */
void control_map_set_preset(uint8_t preset);

/**
 * @brief Get number of available presets
 */
uint8_t control_map_get_preset_count(void);

#endif // SYMCRASH_CONTROL_MAP_H
