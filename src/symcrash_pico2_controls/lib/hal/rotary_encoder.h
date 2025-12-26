/**
 * @file rotary_encoder.h
 * @brief Rotary encoder driver using PIO
 */

#ifndef SYMCRASH_ROTARY_ENCODER_H
#define SYMCRASH_ROTARY_ENCODER_H

#include <stdint.h>
#include <stdbool.h>
#include "config.h"

// =============================================================================
// Types
// =============================================================================

/** Callback for encoder rotation */
typedef void (*encoder_callback_t)(uint8_t encoder_id, int8_t delta);

/** Callback for encoder switch */
typedef void (*encoder_switch_callback_t)(uint8_t encoder_id, bool pressed);

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize rotary encoder(s)
 */
void encoder_init(void);

/**
 * @brief Set callback for encoder rotation
 *
 * @param callback  Function to call when encoder rotates
 */
void encoder_set_callback(encoder_callback_t callback);

/**
 * @brief Set callback for encoder switch
 *
 * @param callback  Function to call when switch is pressed/released
 */
void encoder_set_switch_callback(encoder_switch_callback_t callback);

// =============================================================================
// Reading
// =============================================================================

/**
 * @brief Get and clear accumulated delta for an encoder
 *
 * Returns the number of detent steps since last call.
 * Positive = clockwise, negative = counter-clockwise.
 *
 * @param encoder_id    Encoder ID (0 to NUM_ENCODERS-1)
 * @return Accumulated delta (cleared after call)
 */
int8_t encoder_get_delta(uint8_t encoder_id);

/**
 * @brief Get current position count
 *
 * Unbounded counter that increments/decrements with rotation.
 *
 * @param encoder_id    Encoder ID
 * @return Position count
 */
int32_t encoder_get_position(uint8_t encoder_id);

/**
 * @brief Reset position counter to zero
 *
 * @param encoder_id    Encoder ID
 */
void encoder_reset_position(uint8_t encoder_id);

/**
 * @brief Check if encoder switch is pressed
 *
 * @param encoder_id    Encoder ID
 * @return true if pressed
 */
bool encoder_switch_pressed(uint8_t encoder_id);

// =============================================================================
// Processing
// =============================================================================

/**
 * @brief Update encoder state (call from main loop or timer)
 */
void encoder_update(void);

#endif // SYMCRASH_ROTARY_ENCODER_H
