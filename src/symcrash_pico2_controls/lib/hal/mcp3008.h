/**
 * @file mcp3008.h
 * @brief MCP3008 SPI ADC driver (supports dual ADCs)
 *
 * 10-bit ADC with 8 channels over SPI.
 * Used for reading potentiometers.
 * Supports two MCP3008 chips on the same SPI bus with different CS pins.
 *
 * Channel mapping:
 *   - Channels 0-5: ADC0 (MCP3008_0_CS_PIN)
 *   - Channels 6-9: ADC1 (MCP3008_1_CS_PIN), mapped to ADC1 CH0-3
 */

#ifndef SYMCRASH_MCP3008_H
#define SYMCRASH_MCP3008_H

#include <stdint.h>
#include <stdbool.h>
#include "types.h"
#include "config.h"

// =============================================================================
// Types
// =============================================================================

/** ADC chip ID */
typedef enum {
    MCP3008_ADC0 = 0,   // First ADC (CS at MCP3008_0_CS_PIN)
    MCP3008_ADC1 = 1,   // Second ADC (CS at MCP3008_1_CS_PIN)
    MCP3008_NUM_ADCS = 2
} mcp3008_id_t;

/** Callback for ADC value change (channel is global: 0-5 for ADC0, 6-9 for ADC1) */
typedef void (*adc_change_callback_t)(adc_channel_t channel, adc_value_t value);

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize both MCP3008 SPI interfaces
 */
void mcp3008_init(void);

/**
 * @brief Set callback for ADC value changes
 *
 * Called when a channel value changes beyond deadband.
 *
 * @param callback  Function to call on value change
 */
void mcp3008_set_callback(adc_change_callback_t callback);

// =============================================================================
// Reading
// =============================================================================

/**
 * @brief Read raw value from a channel
 *
 * @param channel   Global channel (0-9)
 * @return 10-bit ADC value (0-1023)
 */
adc_value_t mcp3008_read_raw(adc_channel_t channel);

/**
 * @brief Read smoothed value from a channel
 *
 * @param channel   Global channel (0-9)
 * @return Smoothed 10-bit value
 */
adc_value_t mcp3008_read_smoothed(adc_channel_t channel);

/**
 * @brief Get current (cached) value for a channel
 *
 * Does not trigger a new SPI read.
 *
 * @param channel   Global channel (0-9)
 * @return Cached value
 */
adc_value_t mcp3008_get_value(adc_channel_t channel);

/**
 * @brief Convert ADC value to Q15
 *
 * @param value     10-bit ADC value
 * @return Q15 value (0-32767)
 */
static inline q15_t mcp3008_to_q15(adc_value_t value) {
    return adc10_to_q15(value);
}

// =============================================================================
// Scanning
// =============================================================================

/**
 * @brief Scan all configured channels on both ADCs
 *
 * Reads all channels, applies smoothing, and triggers callbacks
 * for any channels that changed beyond deadband.
 * Call this from a timer at ADC_SCAN_RATE_HZ.
 */
void mcp3008_scan_all(void);

/**
 * @brief Check if a channel value changed since last scan
 *
 * @param channel   Global channel (0-9)
 * @return true if value changed beyond deadband
 */
bool mcp3008_changed(adc_channel_t channel);

/**
 * @brief Clear change flag for a channel
 *
 * @param channel   Global channel (0-9)
 */
void mcp3008_clear_changed(adc_channel_t channel);

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief Set smoothing factor for a channel
 *
 * Higher values = more smoothing (slower response).
 *
 * @param channel   Global channel (0-9)
 * @param factor    Smoothing factor (1-16, default from config)
 */
void mcp3008_set_smoothing(adc_channel_t channel, uint8_t factor);

/**
 * @brief Set deadband for a channel
 *
 * Changes smaller than deadband are ignored.
 *
 * @param channel   Global channel (0-9)
 * @param deadband  Deadband value (0-1023, default from config)
 */
void mcp3008_set_deadband(adc_channel_t channel, uint16_t deadband);

#endif // SYMCRASH_MCP3008_H
