/**
 * @file mcp3008.c
 * @brief MCP3008 SPI ADC driver implementation
 */

#include "mcp3008.h"
#include "config.h"

#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"

// =============================================================================
// State
// =============================================================================

typedef struct {
    adc_value_t raw;            // Last raw reading
    adc_value_t smoothed;       // Smoothed value
    adc_value_t last_sent;      // Last value sent to callback
    uint8_t smoothing;          // Smoothing factor (1-16)
    uint16_t deadband;          // Change threshold
    bool changed;               // Value changed flag
} channel_state_t;

static channel_state_t g_channels[8];
static adc_change_callback_t g_callback = NULL;

// =============================================================================
// Initialization
// =============================================================================

void mcp3008_init(void) {
    // Initialize SPI
    spi_init(MCP3008_SPI_ID, SPI0_BAUD);

    // Set SPI format: 8 bits, CPOL=0, CPHA=0
    spi_set_format(MCP3008_SPI_ID, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST);

    // Set pin functions
    gpio_set_function(SPI0_SCK_PIN, GPIO_FUNC_SPI);
    gpio_set_function(SPI0_MOSI_PIN, GPIO_FUNC_SPI);
    gpio_set_function(SPI0_MISO_PIN, GPIO_FUNC_SPI);

    // CS pin as GPIO output (directly controlled)
    gpio_init(MCP3008_CS_PIN);
    gpio_set_dir(MCP3008_CS_PIN, GPIO_OUT);
    gpio_put(MCP3008_CS_PIN, 1);  // CS high (inactive)

    // Initialize channel states
    for (int i = 0; i < 8; i++) {
        g_channels[i].raw = 0;
        g_channels[i].smoothed = 0;
        g_channels[i].last_sent = 0;
        g_channels[i].smoothing = ADC_SMOOTHING;
        g_channels[i].deadband = ADC_DEADBAND;
        g_channels[i].changed = false;
    }
}

void mcp3008_set_callback(adc_change_callback_t callback) {
    g_callback = callback;
}

// =============================================================================
// Reading
// =============================================================================

adc_value_t mcp3008_read_raw(adc_channel_t channel) {
    if (channel > 7) {
        return 0;
    }

    // MCP3008 SPI protocol:
    // Send: Start bit (1), Single-ended mode (1), Channel (D2-D0)
    // Receive: 10-bit result

    uint8_t tx[3] = {
        0x01,                           // Start bit
        (0x80 | (channel << 4)),        // Single-ended, channel select
        0x00                            // Dummy byte
    };
    uint8_t rx[3] = {0};

    gpio_put(MCP3008_CS_PIN, 0);        // CS low
    spi_write_read_blocking(MCP3008_SPI_ID, tx, rx, 3);
    gpio_put(MCP3008_CS_PIN, 1);        // CS high

    // Extract 10-bit result from rx[1] and rx[2]
    // rx[1]: lower 2 bits of result (bits 9-8)
    // rx[2]: lower 8 bits of result (bits 7-0)
    uint16_t result = ((rx[1] & 0x03) << 8) | rx[2];

    return (adc_value_t)result;
}

adc_value_t mcp3008_read_smoothed(adc_channel_t channel) {
    if (channel > 7) {
        return 0;
    }

    // Read raw value
    adc_value_t raw = mcp3008_read_raw(channel);
    g_channels[channel].raw = raw;

    // Apply exponential smoothing
    // smoothed = smoothed + (raw - smoothed) / factor
    uint8_t factor = g_channels[channel].smoothing;
    if (factor < 1) factor = 1;
    if (factor > 16) factor = 16;

    int32_t smoothed = g_channels[channel].smoothed;
    int32_t diff = (int32_t)raw - smoothed;
    smoothed += diff / factor;

    g_channels[channel].smoothed = (adc_value_t)smoothed;

    return g_channels[channel].smoothed;
}

adc_value_t mcp3008_get_value(adc_channel_t channel) {
    if (channel > 7) {
        return 0;
    }
    return g_channels[channel].smoothed;
}

// =============================================================================
// Scanning
// =============================================================================

void mcp3008_scan_all(void) {
    for (adc_channel_t ch = 0; ch < NUM_POTS; ch++) {
        adc_value_t value = mcp3008_read_smoothed(ch);

        // Check if value changed beyond deadband
        int32_t diff = (int32_t)value - (int32_t)g_channels[ch].last_sent;
        if (diff < 0) diff = -diff;

        if ((uint16_t)diff > g_channels[ch].deadband) {
            g_channels[ch].changed = true;
            g_channels[ch].last_sent = value;

            // Call callback if set
            if (g_callback != NULL) {
                g_callback(ch, value);
            }
        }
    }
}

bool mcp3008_changed(adc_channel_t channel) {
    if (channel > 7) {
        return false;
    }
    return g_channels[channel].changed;
}

void mcp3008_clear_changed(adc_channel_t channel) {
    if (channel <= 7) {
        g_channels[channel].changed = false;
    }
}

// =============================================================================
// Configuration
// =============================================================================

void mcp3008_set_smoothing(adc_channel_t channel, uint8_t factor) {
    if (channel <= 7) {
        g_channels[channel].smoothing = factor;
    }
}

void mcp3008_set_deadband(adc_channel_t channel, uint16_t deadband) {
    if (channel <= 7) {
        g_channels[channel].deadband = deadband;
    }
}
