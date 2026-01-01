/**
 * @file mcp3008.c
 * @brief MCP3008 SPI ADC driver implementation (dual ADC support)
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

// Support up to 16 channels (2 ADCs x 8 channels each)
static channel_state_t g_channels[16];
static adc_change_callback_t g_callback = NULL;

// CS pins for each ADC
static const uint8_t g_cs_pins[MCP3008_NUM_ADCS] = {
    MCP3008_0_CS_PIN,
    MCP3008_1_CS_PIN
};

// =============================================================================
// Helpers
// =============================================================================

/**
 * @brief Get ADC ID and local channel from global channel number
 */
static void channel_to_adc(adc_channel_t global_channel, mcp3008_id_t *adc_id, uint8_t *local_channel) {
    if (global_channel < NUM_POTS_ADC0) {
        *adc_id = MCP3008_ADC0;
        *local_channel = global_channel;
    } else {
        *adc_id = MCP3008_ADC1;
        *local_channel = global_channel - NUM_POTS_ADC0;
    }
}

/**
 * @brief Read raw value from specific ADC and channel
 */
static adc_value_t read_adc_channel(mcp3008_id_t adc_id, uint8_t local_channel) {
    if (adc_id >= MCP3008_NUM_ADCS || local_channel > 7) {
        return 0;
    }

    uint8_t cs_pin = g_cs_pins[adc_id];

    // MCP3008 SPI protocol:
    // Send: Start bit (1), Single-ended mode (1), Channel (D2-D0)
    // Receive: 10-bit result

    uint8_t tx[3] = {
        0x01,                               // Start bit
        (0x80 | (local_channel << 4)),      // Single-ended, channel select
        0x00                                // Dummy byte
    };
    uint8_t rx[3] = {0};

    gpio_put(cs_pin, 0);                    // CS low
    spi_write_read_blocking(MCP3008_SPI_ID, tx, rx, 3);
    gpio_put(cs_pin, 1);                    // CS high

    // Extract 10-bit result from rx[1] and rx[2]
    // rx[1]: lower 2 bits of result (bits 9-8)
    // rx[2]: lower 8 bits of result (bits 7-0)
    uint16_t result = ((rx[1] & 0x03) << 8) | rx[2];

    return (adc_value_t)result;
}

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

    // Initialize CS pins for both ADCs
    for (int adc = 0; adc < MCP3008_NUM_ADCS; adc++) {
        gpio_init(g_cs_pins[adc]);
        gpio_set_dir(g_cs_pins[adc], GPIO_OUT);
        gpio_put(g_cs_pins[adc], 1);  // CS high (inactive)
    }

    // Initialize channel states for all channels
    for (int i = 0; i < NUM_POTS; i++) {
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
    if (channel >= NUM_POTS) {
        return 0;
    }

    mcp3008_id_t adc_id;
    uint8_t local_channel;
    channel_to_adc(channel, &adc_id, &local_channel);

    return read_adc_channel(adc_id, local_channel);
}

adc_value_t mcp3008_read_smoothed(adc_channel_t channel) {
    if (channel >= NUM_POTS) {
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
    if (channel >= NUM_POTS) {
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
    if (channel >= NUM_POTS) {
        return false;
    }
    return g_channels[channel].changed;
}

void mcp3008_clear_changed(adc_channel_t channel) {
    if (channel < NUM_POTS) {
        g_channels[channel].changed = false;
    }
}

// =============================================================================
// Configuration
// =============================================================================

void mcp3008_set_smoothing(adc_channel_t channel, uint8_t factor) {
    if (channel < NUM_POTS) {
        g_channels[channel].smoothing = factor;
    }
}

void mcp3008_set_deadband(adc_channel_t channel, uint16_t deadband) {
    if (channel < NUM_POTS) {
        g_channels[channel].deadband = deadband;
    }
}
