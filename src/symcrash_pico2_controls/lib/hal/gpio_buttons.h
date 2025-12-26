/**
 * @file gpio_buttons.h
 * @brief GPIO button handler with debouncing
 */

#ifndef SYMCRASH_GPIO_BUTTONS_H
#define SYMCRASH_GPIO_BUTTONS_H

#include <stdint.h>
#include <stdbool.h>
#include "config.h"

// =============================================================================
// Types
// =============================================================================

/** Button state */
typedef enum {
    BTN_STATE_RELEASED,
    BTN_STATE_PRESSED,
    BTN_STATE_HELD,
    BTN_STATE_LONG_PRESS
} button_state_t;

/** Button event */
typedef enum {
    BTN_EVENT_NONE,
    BTN_EVENT_PRESS,
    BTN_EVENT_RELEASE,
    BTN_EVENT_LONG_PRESS,
    BTN_EVENT_REPEAT
} button_event_t;

/** Callback for button events */
typedef void (*button_callback_t)(uint8_t button_id, button_event_t event);

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize button GPIO pins
 */
void buttons_init(void);

/**
 * @brief Set callback for button events
 *
 * @param callback  Function to call on button events
 */
void buttons_set_callback(button_callback_t callback);

// =============================================================================
// Reading
// =============================================================================

/**
 * @brief Get current button state
 *
 * @param button_id     Button ID (0 to NUM_BUTTONS-1)
 * @return Current state
 */
button_state_t buttons_get_state(uint8_t button_id);

/**
 * @brief Poll for button event
 *
 * @param button_id     Button ID
 * @return Event that occurred, or BTN_EVENT_NONE
 */
button_event_t buttons_poll(uint8_t button_id);

/**
 * @brief Check if button is currently pressed
 *
 * @param button_id     Button ID
 * @return true if pressed
 */
bool buttons_is_pressed(uint8_t button_id);

// =============================================================================
// Processing
// =============================================================================

/**
 * @brief Update button states (call from timer at ~200 Hz)
 *
 * Handles debouncing, long-press detection, and auto-repeat.
 */
void buttons_update(void);

#endif // SYMCRASH_GPIO_BUTTONS_H
