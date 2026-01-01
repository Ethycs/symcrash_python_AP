/**
 * @file init.c
 * @brief System initialization for Pico 2 W
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
#include "pico/cyw43_arch.h"

#if TRANSPORT_USB_MODE
#include "tusb.h"
#endif

// =============================================================================
// LED Control (Pico 2 W uses CYW43)
// =============================================================================

static bool g_cyw43_initialized = false;

/**
 * @brief Initialize CYW43 for LED control (Pico 2 W)
 */
static bool init_cyw43(void) {
    if (cyw43_arch_init()) {
        return false;
    }
    g_cyw43_initialized = true;
    return true;
}

/**
 * @brief Set LED state (Pico 2 W uses CYW43)
 */
void led_set(bool on) {
    if (g_cyw43_initialized) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, on ? 1 : 0);
    }
}

/**
 * @brief Toggle LED state
 */
void led_toggle(void) {
    static bool led_state = false;
    led_state = !led_state;
    led_set(led_state);
}

/**
 * @brief Blink LED pattern to indicate startup
 */
static void startup_blink(void) {
    for (int i = 0; i < 3; i++) {
        led_set(true);
        sleep_ms(100);
        led_set(false);
        sleep_ms(100);
    }
}

// =============================================================================
// System Initialization
// =============================================================================

/**
 * @brief Initialize all hardware subsystems
 */
void system_init(void) {
    // FIRST: Initialize CYW43 for LED control
    if (!init_cyw43()) {
        // Can't even blink LED to show error - just halt
        while (1) tight_loop_contents();
    }

    // Blink to show we're alive
    startup_blink();

    // Initialize TinyUSB for dual CDC
    tusb_init();

    // Brief wait for USB enumeration
    for (int i = 0; i < 100; i++) {
        tud_task();
        sleep_ms(10);
    }

    // Initialize MCP3008 ADC for potentiometers
    mcp3008_init();

    // Initialize GC9A01 LCD displays (uses full Adafruit init sequence)
    gc9a01_init();

    // Initialize display pages
    display_pages_init();

    // Initialize encoder (moved to GP20/21/22 to avoid LCD conflicts)
    // buttons_init();  // Disabled - BTN_0/GP5 conflicts with LCD DC pin
    // encoder_init();  // TEMP DISABLED to test crosstalk

    // Initialize control mapping (connects pots to macros and display)
    control_map_init();

    // Force display update with initial pot values
    // This ensures pots are shown before entering main loop
    display_pages_render();

    // Initialize menu
    menu_init();

    // Initialize communication
    dual_uart_init();

    // LED on to show init complete
    led_set(true);
}
