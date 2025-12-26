/**
 * @file rotary_encoder.c
 * @brief Rotary encoder driver implementation
 *
 * Uses GPIO polling with state machine for reliable decoding.
 * PIO version available for higher performance if needed.
 */

#include "rotary_encoder.h"
#include "config.h"

#include "pico/stdlib.h"
#include "hardware/gpio.h"

// =============================================================================
// State
// =============================================================================

typedef struct {
    int32_t position;           // Cumulative position
    int8_t delta;               // Accumulated delta since last read
    uint8_t last_state;         // Previous A/B state
    uint8_t switch_debounce;    // Switch debounce counter
    bool switch_state;          // Debounced switch state
    bool switch_raw;            // Raw switch state
} encoder_state_t;

static encoder_state_t g_encoders[NUM_ENCODERS];
static encoder_callback_t g_rotate_callback = NULL;
static encoder_switch_callback_t g_switch_callback = NULL;

// Quadrature state transition table
// Index: (old_state << 2) | new_state
// Values: 0 = no movement, 1 = CW, -1 = CCW, 2 = error (skipped state)
static const int8_t g_quad_table[16] = {
     0,  1, -1,  2,   // 00 -> 00, 01, 10, 11
    -1,  0,  2,  1,   // 01 -> 00, 01, 10, 11
     1,  2,  0, -1,   // 10 -> 00, 01, 10, 11
     2, -1,  1,  0    // 11 -> 00, 01, 10, 11
};

// =============================================================================
// Initialization
// =============================================================================

void encoder_init(void) {
    // Initialize encoder 0
    gpio_init(ENC0_A_PIN);
    gpio_init(ENC0_B_PIN);
    gpio_init(ENC0_SW_PIN);

    gpio_set_dir(ENC0_A_PIN, GPIO_IN);
    gpio_set_dir(ENC0_B_PIN, GPIO_IN);
    gpio_set_dir(ENC0_SW_PIN, GPIO_IN);

    gpio_pull_up(ENC0_A_PIN);
    gpio_pull_up(ENC0_B_PIN);
    gpio_pull_up(ENC0_SW_PIN);

    // Initialize state
    for (int i = 0; i < NUM_ENCODERS; i++) {
        g_encoders[i].position = 0;
        g_encoders[i].delta = 0;
        g_encoders[i].switch_debounce = 0;
        g_encoders[i].switch_state = false;
        g_encoders[i].switch_raw = false;

        // Read initial state
        bool a = !gpio_get(ENC0_A_PIN);
        bool b = !gpio_get(ENC0_B_PIN);
        g_encoders[i].last_state = (a << 1) | b;
    }
}

void encoder_set_callback(encoder_callback_t callback) {
    g_rotate_callback = callback;
}

void encoder_set_switch_callback(encoder_switch_callback_t callback) {
    g_switch_callback = callback;
}

// =============================================================================
// Reading
// =============================================================================

int8_t encoder_get_delta(uint8_t encoder_id) {
    if (encoder_id >= NUM_ENCODERS) {
        return 0;
    }

    int8_t delta = g_encoders[encoder_id].delta;
    g_encoders[encoder_id].delta = 0;
    return delta;
}

int32_t encoder_get_position(uint8_t encoder_id) {
    if (encoder_id >= NUM_ENCODERS) {
        return 0;
    }
    return g_encoders[encoder_id].position;
}

void encoder_reset_position(uint8_t encoder_id) {
    if (encoder_id < NUM_ENCODERS) {
        g_encoders[encoder_id].position = 0;
    }
}

bool encoder_switch_pressed(uint8_t encoder_id) {
    if (encoder_id >= NUM_ENCODERS) {
        return false;
    }
    return g_encoders[encoder_id].switch_state;
}

// =============================================================================
// Processing
// =============================================================================

void encoder_update(void) {
    // Update encoder 0
    if (NUM_ENCODERS >= 1) {
        encoder_state_t *enc = &g_encoders[0];

        // Read current state
        bool a = !gpio_get(ENC0_A_PIN);
        bool b = !gpio_get(ENC0_B_PIN);
        uint8_t new_state = (a << 1) | b;

        // Detect rotation
        if (new_state != enc->last_state) {
            uint8_t index = (enc->last_state << 2) | new_state;
            int8_t movement = g_quad_table[index];

            if (movement == 1 || movement == -1) {
                enc->position += movement;
                enc->delta += movement;

                // Check for detent (every ENCODER_PULSES_PER_DETENT)
                if ((enc->position % ENCODER_PULSES_PER_DETENT) == 0) {
                    if (g_rotate_callback) {
                        g_rotate_callback(0, movement > 0 ? 1 : -1);
                    }
                }
            }

            enc->last_state = new_state;
        }

        // Update switch with debouncing
        bool sw_pressed = !gpio_get(ENC0_SW_PIN);

        if (sw_pressed == enc->switch_raw) {
            enc->switch_debounce++;
            if (enc->switch_debounce >= 4) {  // ~20ms at 200Hz
                if (sw_pressed != enc->switch_state) {
                    enc->switch_state = sw_pressed;
                    if (g_switch_callback) {
                        g_switch_callback(0, sw_pressed);
                    }
                }
            }
        } else {
            enc->switch_debounce = 0;
            enc->switch_raw = sw_pressed;
        }
    }
}
