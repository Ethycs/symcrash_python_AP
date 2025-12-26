/**
 * @file mcp3008.h
 * @brief MCP3008 SPI ADC driver
 *
 * 10-bit ADC with 8 channels over SPI.
 * Used for reading potentiometers.
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

/** Callback for ADC value change */
typedef void (*adc_change_callback_t)(adc_channel_t channel, adc_value_t value);

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize MCP3008 SPI interface
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
 * @param channel   Channel to read (0-7)
 * @return 10-bit ADC value (0-1023)
 */
adc_value_t mcp3008_read_raw(adc_channel_t channel);

/**
 * @brief Read smoothed value from a channel
 *
 * @param channel   Channel to read (0-7)
 * @return Smoothed 10-bit value
 */
adc_value_t mcp3008_read_smoothed(adc_channel_t channel);

/**
 * @brief Get current (cached) value for a channel
 *
 * Does not trigger a new SPI read.
 *
 * @param channel   Channel (0-7)
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
 * @brief Scan all configured channels
 *
 * Reads all channels, applies smoothing, and triggers callbacks
 * for any channels that changed beyond deadband.
 * Call this from a timer at ADC_SCAN_RATE_HZ.
 */
void mcp3008_scan_all(void);

/**
 * @brief Check if a channel value changed since last scan
 *
 * @param channel   Channel to check (0-7)
 * @return true if value changed beyond deadband
 */
bool mcp3008_changed(adc_channel_t channel);

/**
 * @brief Clear change flag for a channel
 *
 * @param channel   Channel to clear (0-7)
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
 * @param channel   Channel (0-7)
 * @param factor    Smoothing factor (1-16, default from config)
 */
void mcp3008_set_smoothing(adc_channel_t channel, uint8_t factor);

/**
 * @brief Set deadband for a channel
 *
 * Changes smaller than deadband are ignored.
 *
 * @param channel   Channel (0-7)
 * @param deadband  Deadband value (0-1023, default from config)
 */
void mcp3008_set_deadband(adc_channel_t channel, uint16_t deadband);

#endif // SYMCRASH_MCP3008_H
