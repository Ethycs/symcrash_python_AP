/**
 * @file gpio_buttons.c
 * @brief GPIO button handler implementation
 */

#include "gpio_buttons.h"
#include "pico/stdlib.h"
#include "hardware/gpio.h"

// =============================================================================
// Pin mapping
// =============================================================================

static const uint8_t g_button_pins[NUM_BUTTONS] = {
    BTN_0,
    BTN_1,
    BTN_2,
    BTN_3
};

// =============================================================================
// State
// =============================================================================

typedef struct {
    button_state_t state;
    uint32_t press_time_ms;         // Time when pressed
    uint32_t last_event_time_ms;    // Time of last event
    uint8_t debounce_count;         // Debounce counter
    button_event_t pending_event;   // Event to be polled
    bool raw_state;                 // Raw GPIO state
} button_internal_t;

static button_internal_t g_buttons[NUM_BUTTONS];
static button_callback_t g_callback = NULL;

// =============================================================================
// Initialization
// =============================================================================

void buttons_init(void) {
    for (int i = 0; i < NUM_BUTTONS; i++) {
        uint8_t pin = g_button_pins[i];

        // Configure as input with pull-up
        gpio_init(pin);
        gpio_set_dir(pin, GPIO_IN);
        gpio_pull_up(pin);

        // Initialize state
        g_buttons[i].state = BTN_STATE_RELEASED;
        g_buttons[i].press_time_ms = 0;
        g_buttons[i].last_event_time_ms = 0;
        g_buttons[i].debounce_count = 0;
        g_buttons[i].pending_event = BTN_EVENT_NONE;
        g_buttons[i].raw_state = false;
    }
}

void buttons_set_callback(button_callback_t callback) {
    g_callback = callback;
}

// =============================================================================
// Reading
// =============================================================================

button_state_t buttons_get_state(uint8_t button_id) {
    if (button_id >= NUM_BUTTONS) {
        return BTN_STATE_RELEASED;
    }
    return g_buttons[button_id].state;
}

button_event_t buttons_poll(uint8_t button_id) {
    if (button_id >= NUM_BUTTONS) {
        return BTN_EVENT_NONE;
    }

    button_event_t event = g_buttons[button_id].pending_event;
    g_buttons[button_id].pending_event = BTN_EVENT_NONE;
    return event;
}

bool buttons_is_pressed(uint8_t button_id) {
    if (button_id >= NUM_BUTTONS) {
        return false;
    }
    return g_buttons[button_id].state != BTN_STATE_RELEASED;
}

// =============================================================================
// Processing
// =============================================================================

void buttons_update(void) {
    uint32_t now_ms = to_ms_since_boot(get_absolute_time());

    for (int i = 0; i < NUM_BUTTONS; i++) {
        uint8_t pin = g_button_pins[i];
        button_internal_t *btn = &g_buttons[i];

        // Read raw state (active low)
        bool pressed = !gpio_get(pin);

        // Debounce: require stable state for debounce period
        if (pressed == btn->raw_state) {
            btn->debounce_count++;
        } else {
            btn->debounce_count = 0;
            btn->raw_state = pressed;
        }

        // State machine
        switch (btn->state) {
            case BTN_STATE_RELEASED:
                if (btn->debounce_count >= (BUTTON_DEBOUNCE_MS / 5) && pressed) {
                    btn->state = BTN_STATE_PRESSED;
                    btn->press_time_ms = now_ms;
                    btn->pending_event = BTN_EVENT_PRESS;
                    if (g_callback) {
                        g_callback(i, BTN_EVENT_PRESS);
                    }
                }
                break;

            case BTN_STATE_PRESSED:
                if (!pressed && btn->debounce_count >= (BUTTON_DEBOUNCE_MS / 5)) {
                    btn->state = BTN_STATE_RELEASED;
                    btn->pending_event = BTN_EVENT_RELEASE;
                    if (g_callback) {
                        g_callback(i, BTN_EVENT_RELEASE);
                    }
                } else if (pressed && (now_ms - btn->press_time_ms) >= BUTTON_LONG_PRESS_MS) {
                    btn->state = BTN_STATE_LONG_PRESS;
                    btn->last_event_time_ms = now_ms;
                    btn->pending_event = BTN_EVENT_LONG_PRESS;
                    if (g_callback) {
                        g_callback(i, BTN_EVENT_LONG_PRESS);
                    }
                }
                break;

            case BTN_STATE_LONG_PRESS:
            case BTN_STATE_HELD:
                if (!pressed && btn->debounce_count >= (BUTTON_DEBOUNCE_MS / 5)) {
                    btn->state = BTN_STATE_RELEASED;
                    btn->pending_event = BTN_EVENT_RELEASE;
                    if (g_callback) {
                        g_callback(i, BTN_EVENT_RELEASE);
                    }
                } else if (pressed && BUTTON_REPEAT_MS > 0) {
                    // Auto-repeat
                    if ((now_ms - btn->last_event_time_ms) >= BUTTON_REPEAT_MS) {
                        btn->last_event_time_ms = now_ms;
                        btn->pending_event = BTN_EVENT_REPEAT;
                        if (g_callback) {
                            g_callback(i, BTN_EVENT_REPEAT);
                        }
                    }
                }
                break;
        }
    }
}
