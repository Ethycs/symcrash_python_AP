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

#if TRANSPORT_USB_MODE
#include "tusb.h"
#endif

// Forward declaration from init.c
extern void system_init(void);

int main(void) {
    // Initialize all hardware and software subsystems
    system_init();

    // Main loop - send test messages over USB CDC
    uint32_t counter = 0;
    uint16_t test_value = 0;

    while (true) {
#if TRANSPORT_USB_MODE
        tud_task();  // Keep USB alive
#endif
        dual_uart_process();

        sleep_ms(100);
        counter++;

        // Toggle LED every second
        if (counter % 10 == 0) {
            static bool led_state = true;
            led_state = !led_state;
            cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, led_state);
        }

        // Send test message every second
        if (counter % 10 == 0) {
            test_value = (test_value + 1000) % 32768;
            dual_uart_send_macro(0, test_value);
        }
    }

    return 0;
}
