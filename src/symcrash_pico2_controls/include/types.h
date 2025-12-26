/**
 * @file types.h
 * @brief Core type definitions for Symcrash Pico2 Controller
 *
 * Fixed-point arithmetic types matching the Python implementation (WSE-STD-001).
 * All Q15 values use the same format as the synthesis engine.
 */

#ifndef SYMCRASH_TYPES_H
#define SYMCRASH_TYPES_H

#include <stdint.h>
#include <stdbool.h>

// =============================================================================
// Fixed-Point Types (Q1.15 format)
// =============================================================================

/** Q1.15 fixed-point: 1 sign bit, 15 fractional bits. Range: [-1.0, +0.99997] */
typedef int16_t q15_t;

/** Q1.7 fixed-point for compact storage */
typedef int8_t q7_t;

/** Unsigned Q0.15 for positive-only values [0, 0.99997] */
typedef uint16_t uq15_t;

// Q15 constants
#define Q15_ONE         32767       // Maximum positive value (~0.99997)
#define Q15_HALF        16384       // 0.5
#define Q15_ZERO        0           // 0.0
#define Q15_NEG_ONE     (-32768)    // -1.0
#define Q15_SCALE       32768       // Scaling factor (2^15)

// =============================================================================
// Fixed-Point Arithmetic
// =============================================================================

/**
 * @brief Multiply two Q15 values with rounding
 * @param a First operand
 * @param b Second operand
 * @return Product in Q15 format
 */
static inline q15_t q15_mul(q15_t a, q15_t b) {
    int32_t prod = (int32_t)a * (int32_t)b;
    return (q15_t)((prod + 0x4000) >> 15);  // Round to nearest
}

/**
 * @brief Saturating addition of two Q15 values
 * @param a First operand
 * @param b Second operand
 * @return Sum clamped to Q15 range
 */
static inline q15_t q15_add_sat(q15_t a, q15_t b) {
    int32_t sum = (int32_t)a + (int32_t)b;
    if (sum > Q15_ONE) return Q15_ONE;
    if (sum < Q15_NEG_ONE) return Q15_NEG_ONE;
    return (q15_t)sum;
}

/**
 * @brief Convert unsigned 10-bit ADC value to Q15 [0, Q15_ONE]
 * @param adc_val 10-bit ADC value (0-1023)
 * @return Q15 value (0-32767)
 */
static inline q15_t adc10_to_q15(uint16_t adc_val) {
    // Scale: (adc_val * 32767) / 1023
    // Use 32-bit math: (adc_val * 32767 + 511) / 1023
    return (q15_t)(((uint32_t)adc_val * Q15_ONE + 511) / 1023);
}

/**
 * @brief Convert unsigned 12-bit ADC value to Q15 [0, Q15_ONE]
 * @param adc_val 12-bit ADC value (0-4095)
 * @return Q15 value (0-32767)
 */
static inline q15_t adc12_to_q15(uint16_t adc_val) {
    return (q15_t)(((uint32_t)adc_val * Q15_ONE + 2047) / 4095);
}

// =============================================================================
// MIDI Types
// =============================================================================

/** MIDI note number (0-127) */
typedef uint8_t midi_note_t;

/** MIDI velocity (0-127) */
typedef uint8_t velocity_t;

/** MIDI channel (0-15) */
typedef uint8_t midi_channel_t;

// =============================================================================
// Control Types
// =============================================================================

/** Macro knob ID (0-7 typical) */
typedef uint8_t macro_id_t;

/** Tree index (0-4 for wavelet trees) */
typedef uint8_t tree_index_t;

/** Block ID for timing synchronization (monotonic counter) */
typedef uint32_t block_id_t;

/** Voice ID (0-127) */
typedef uint8_t voice_id_t;

/** Preset ID */
typedef uint8_t preset_id_t;

// =============================================================================
// ADC Types
// =============================================================================

/** Raw 10-bit ADC value from MCP3008 (0-1023) */
typedef uint16_t adc_value_t;

/** ADC channel index (0-7 for MCP3008) */
typedef uint8_t adc_channel_t;

// =============================================================================
// Limits
// =============================================================================

#define MAX_VOICES          128
#define MAX_MACROS          8
#define MAX_TREES           5
#define MAX_ADC_CHANNELS    8
#define MAX_BUTTONS         16
#define MAX_ENCODERS        4

#endif // SYMCRASH_TYPES_H
