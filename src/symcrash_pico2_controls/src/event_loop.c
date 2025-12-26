/**
 * @file event_loop.c
 * @brief Main event loop implementation
 */

#include "event_loop.h"
#include "config.h"

// Protocol
#include "dual_uart_link.h"

// HAL
#include "mcp3008.h"
#include "gpio_buttons.h"
#include "rotary_encoder.h"
#include "gc9a01.h"

// Controls
#include "control_map.h"

// UI
#include "menu.h"
#include "display_pages.h"

#include "pico/stdlib.h"
#include "hardware/timer.h"
#include "hardware/sync.h"

// =============================================================================
// Timer Periods (microseconds)
// =============================================================================

#define ADC_SCAN_PERIOD_US      1000    // 1 kHz ADC scanning
#define BUTTON_SCAN_PERIOD_US   5000    // 200 Hz button/encoder scanning
#define DISPLAY_UPDATE_PERIOD_US 33333  // 30 Hz display refresh
#define UART_PROCESS_PERIOD_US  100     // 10 kHz UART processing

// =============================================================================
// Event Flags
// =============================================================================

static volatile uint32_t g_event_flags = 0;

#define EVT_ADC_SCAN        (1 << 0)
#define EVT_BUTTON_SCAN     (1 << 1)
#define EVT_DISPLAY_UPDATE  (1 << 2)

// =============================================================================
// Timer Callbacks
// =============================================================================

static bool adc_timer_callback(repeating_timer_t *rt) {
    (void)rt;
    g_event_flags |= EVT_ADC_SCAN;
    return true;
}

static bool button_timer_callback(repeating_timer_t *rt) {
    (void)rt;
    g_event_flags |= EVT_BUTTON_SCAN;
    return true;
}

static bool display_timer_callback(repeating_timer_t *rt) {
    (void)rt;
    g_event_flags |= EVT_DISPLAY_UPDATE;
    return true;
}

// =============================================================================
// Initialization
// =============================================================================

static repeating_timer_t g_adc_timer;
static repeating_timer_t g_button_timer;
static repeating_timer_t g_display_timer;

void event_loop_init(void) {
    // Start timers (negative period = delay between calls)
    add_repeating_timer_us(-ADC_SCAN_PERIOD_US, adc_timer_callback, NULL, &g_adc_timer);
    add_repeating_timer_us(-BUTTON_SCAN_PERIOD_US, button_timer_callback, NULL, &g_button_timer);
    add_repeating_timer_us(-DISPLAY_UPDATE_PERIOD_US, display_timer_callback, NULL, &g_display_timer);
}

// =============================================================================
// Main Event Loop
// =============================================================================

void event_loop_run(void) {
    while (true) {
        // Process UART (high priority, run every iteration)
        dual_uart_process();

        // Process ADC scanning
        if (g_event_flags & EVT_ADC_SCAN) {
            g_event_flags &= ~EVT_ADC_SCAN;
            mcp3008_scan_all();

            // DEBUG: Send a test message every ~1 second (every 1000 ADC scans)
            static uint32_t test_counter = 0;
            if (++test_counter >= 1000) {
                test_counter = 0;
                // Send macro 0 with incrementing value
                static uint16_t test_value = 0;
                test_value = (test_value + 1000) % 32768;
                dual_uart_send_macro(0, test_value);
            }
        }

        // Process buttons and encoders
        if (g_event_flags & EVT_BUTTON_SCAN) {
            g_event_flags &= ~EVT_BUTTON_SCAN;
            buttons_update();
            encoder_update();
        }

        // Process display update
        if (g_event_flags & EVT_DISPLAY_UPDATE) {
            g_event_flags &= ~EVT_DISPLAY_UPDATE;

            // Update connection status
            display_pages_set_connected(dual_uart_is_connected());

            // Update preset from control map
            display_pages_set_preset(control_map_get_preset(), NULL);

            // Render displays
            display_pages_render();
        }

        // Low-power wait for next event
        __wfi();
    }
}

void event_loop_request_display_update(void) {
    display_pages_invalidate();
    g_event_flags |= EVT_DISPLAY_UPDATE;
}
