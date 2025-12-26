/**
 * @file main.c
 * @brief Symcrash Pico2 Hardware UI Controller
 *
 * Main entry point for the Raspberry Pi Pico2 firmware.
 * Implements a hardware control surface for the symcrash audio synthesis engine.
 *
 * Hardware:
 * - 6x potentiometers via MCP3008 on SPI0
 * - 4x buttons on GPIO
 * - 1x rotary encoder with switch
 * - 2x GC9A01 round LCD displays on SPI1
 * - Dual UART: C-UART (control) + D-UART (data)
 *
 * Protocol: WSE-STD-100 Dual-UART Link
 */

#include "pico/stdlib.h"
#include "event_loop.h"

// Forward declaration from init.c
extern void system_init(void);

int main(void) {
    // Initialize all hardware and software subsystems
    system_init();

    // Run the main event loop (never returns)
    event_loop_run();

    // Should never reach here
    return 0;
}
