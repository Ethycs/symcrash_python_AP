/**
 * @file main.c
 * @brief Symcrash Pico2 W Hardware UI Controller
 *
 * Main entry point for the Raspberry Pi Pico 2 W firmware.
 * Implements a hardware control surface for the symcrash audio synthesis engine.
 *
 * Hardware:
 * - 6x potentiometers via MCP3008 on SPI0
 * - 4x buttons on GPIO
 * - 1x rotary encoder with switch
 * - 2x GC9A01 round LCD displays on SPI1
 * - Dual USB CDC: Control channel + Data channel
 *
 * Protocol: WSE-STD-100 Dual-UART Link over USB CDC
 */

#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "config.h"
#include "dual_uart_link.h"
#include "mcp3008.h"
#include "gpio_buttons.h"
#include "rotary_encoder.h"
#include "display_pages.h"

#if TRANSPORT_USB_MODE
#include "tusb.h"
#endif

// Forward declarations from init.c
extern void system_init(void);
extern void led_toggle(void);

int main(void) {
    // Initialize all hardware and software subsystems
    system_init();

    // Timing intervals
    uint32_t last_adc_scan = 0;
    uint32_t last_button_scan = 0;
    uint32_t last_display_update = 0;
    uint32_t last_led_toggle = 0;

    const uint32_t ADC_SCAN_INTERVAL_MS = 1;       // 1 kHz ADC scanning
    const uint32_t BUTTON_SCAN_INTERVAL_MS = 5;    // 200 Hz button/encoder scan
    const uint32_t DISPLAY_UPDATE_INTERVAL_MS = 33; // 30 Hz display refresh
    const uint32_t LED_TOGGLE_INTERVAL_MS = 1000;  // 1 Hz LED heartbeat

    while (true) {
        uint32_t now = to_ms_since_boot(get_absolute_time());

#if TRANSPORT_USB_MODE
        tud_task();  // Keep USB alive
#endif
        dual_uart_process();

        // Scan ADC at 1 kHz
        if (now - last_adc_scan >= ADC_SCAN_INTERVAL_MS) {
            last_adc_scan = now;
            mcp3008_scan_all();  // Triggers callbacks for changed values
        }

        // Scan buttons and encoder at 200 Hz
        if (now - last_button_scan >= BUTTON_SCAN_INTERVAL_MS) {
            last_button_scan = now;
            buttons_update();
            encoder_update();
        }

        // Update display at 30 Hz
        if (now - last_display_update >= DISPLAY_UPDATE_INTERVAL_MS) {
            last_display_update = now;
            display_pages_set_connected(dual_uart_is_connected());
            display_pages_render();
        }

        // LED heartbeat every second
        if (now - last_led_toggle >= LED_TOGGLE_INTERVAL_MS) {
            last_led_toggle = now;
            led_toggle();
        }

        // Small sleep to prevent tight looping
        sleep_us(100);
    }

    return 0;
}
