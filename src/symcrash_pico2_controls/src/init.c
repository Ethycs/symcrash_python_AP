/**
 * @file init.c
 * @brief System initialization
 */

#include "config.h"
#include "error_codes.h"

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

// Event loop
#include "event_loop.h"

#include "pico/stdlib.h"
#include "hardware/gpio.h"

/**
 * @brief Initialize status LED
 */
static void init_led(void) {
    gpio_init(LED_STATUS);
    gpio_set_dir(LED_STATUS, GPIO_OUT);
    gpio_put(LED_STATUS, 0);
}

/**
 * @brief Blink LED pattern to indicate startup
 */
static void startup_blink(void) {
    for (int i = 0; i < 3; i++) {
        gpio_put(LED_STATUS, 1);
        sleep_ms(100);
        gpio_put(LED_STATUS, 0);
        sleep_ms(100);
    }
}

/**
 * @brief Initialize all hardware subsystems
 */
void system_init(void) {
    // Initialize stdio for debug output
    stdio_init_all();

    // Status LED
    init_led();
    startup_blink();

    // Communication
    dual_uart_init();

    // Hardware abstraction
    mcp3008_init();
    buttons_init();
    encoder_init();
    gc9a01_init();

    // Control mapping (registers HAL callbacks)
    control_map_init();

    // UI
    menu_init();
    display_pages_init();

    // Event loop timers
    event_loop_init();

    // Show startup screen
    display_pages_invalidate();
    display_pages_render();

    // Try to connect to synthesis engine
    error_t err = dual_uart_connect();
    if (err == ERR_OK) {
        gpio_put(LED_STATUS, 1);  // LED on = connected
    }

    // Update display with connection status
    display_pages_set_connected(dual_uart_is_connected());
    display_pages_invalidate();
    display_pages_render();
}
