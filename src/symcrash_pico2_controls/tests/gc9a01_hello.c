/**
 * @file gc9a01_hello.c
 * @brief Hello World for GC9A01 round LCD on SPI1
 *
 * Simple test program that initializes the GC9A01 display and shows
 * "HELLO WORLD" with some basic graphics.
 *
 * Hardware:
 *   - GC9A01 240x240 round LCD on SPI1
 *   - SCK:  GP10
 *   - MOSI: GP11
 *   - CS:   GP12 (Display 0)
 *   - DC:   GP13
 *   - RST:  GP14
 */

#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "gc9a01.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>

// Animation frame counter
static uint32_t frame = 0;

// Simple integer square root (no math.h needed)
static int16_t isqrt(int32_t n) {
    if (n < 0) return 0;
    if (n < 2) return (int16_t)n;
    int32_t x = n;
    int32_t y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return (int16_t)x;
}

// Simple pseudo-random number generator (no stdlib rand needed)
static uint32_t g_rand_seed = 12345;
static int simple_rand(void) {
    g_rand_seed = g_rand_seed * 1103515245 + 12345;
    return (int)((g_rand_seed >> 16) & 0x7FFF);
}

/**
 * @brief Blink onboard LED
 */
static void led_blink(int count, int delay_ms) {
    for (int i = 0; i < count; i++) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(delay_ms);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(delay_ms);
    }
}

/**
 * @brief Draw a simple rainbow gradient
 */
static void draw_rainbow(display_id_t display) {
    // Draw concentric colored rings
    gc9a01_fill_circle(display, 120, 120, 110, COLOR_RED);
    gc9a01_fill_circle(display, 120, 120, 95, COLOR_ORANGE);
    gc9a01_fill_circle(display, 120, 120, 80, COLOR_YELLOW);
    gc9a01_fill_circle(display, 120, 120, 65, COLOR_GREEN);
    gc9a01_fill_circle(display, 120, 120, 50, COLOR_CYAN);
    gc9a01_fill_circle(display, 120, 120, 35, COLOR_BLUE);
    gc9a01_fill_circle(display, 120, 120, 20, COLOR_MAGENTA);
}

/**
 * @brief Draw the main hello world screen
 */
static void draw_hello_screen(display_id_t display) {
    // Clear to dark blue
    gc9a01_fill(display, color_rgb(0, 0, 40));

    // Draw border circle
    gc9a01_circle(display, 120, 120, 115, COLOR_WHITE);
    gc9a01_circle(display, 120, 120, 114, COLOR_WHITE);

    // Draw "HELLO" at top
    gc9a01_draw_string_centered(display, 60, "HELLO", COLOR_CYAN, color_rgb(0, 0, 40), 3);

    // Draw "WORLD" at bottom
    gc9a01_draw_string_centered(display, 140, "WORLD", COLOR_YELLOW, color_rgb(0, 0, 40), 3);

    // Draw small decoration
    gc9a01_fill_circle(display, 120, 110, 8, COLOR_RED);
    gc9a01_circle(display, 120, 110, 12, COLOR_WHITE);
}

/**
 * @brief Animated bouncing ball demo
 */
static void animate_ball(display_id_t display) {
    static int16_t ball_x = 120;
    static int16_t ball_y = 120;
    static int16_t vel_x = 3;
    static int16_t vel_y = 2;
    const int16_t radius = 15;

    // Clear old ball position (draw background color)
    gc9a01_fill_circle(display, ball_x, ball_y, radius + 2, COLOR_BLACK);

    // Update position
    ball_x += vel_x;
    ball_y += vel_y;

    // Bounce off edges (keeping within circular display)
    int32_t dx = ball_x - 120;
    int32_t dy = ball_y - 120;
    int16_t dist_from_center = isqrt(dx * dx + dy * dy);

    if (dist_from_center > 120 - radius) {
        // Reflect velocity with slight randomness
        vel_x = -vel_x + (simple_rand() % 3 - 1);
        vel_y = -vel_y + (simple_rand() % 3 - 1);

        // Clamp velocity
        if (vel_x > 5) vel_x = 5;
        if (vel_x < -5) vel_x = -5;
        if (vel_y > 5) vel_y = 5;
        if (vel_y < -5) vel_y = -5;

        // Push ball back inside
        if (dist_from_center > 0) {
            ball_x = 120 + (int16_t)(dx * (120 - radius) / dist_from_center);
            ball_y = 120 + (int16_t)(dy * (120 - radius) / dist_from_center);
        }
    }

    // Draw new ball
    gc9a01_fill_circle(display, ball_x, ball_y, radius, COLOR_RED);
    gc9a01_circle(display, ball_x, ball_y, radius, COLOR_WHITE);

    // Draw frame counter
    char buf[16];
    snprintf(buf, sizeof(buf), "F:%lu", (unsigned long)frame);
    gc9a01_draw_string(display, 5, 5, buf, COLOR_GREEN, COLOR_BLACK, 1);

    frame++;
}

/**
 * @brief Color test pattern
 */
static void draw_color_test(display_id_t display) {
    // Draw color bars
    gc9a01_fill_rect(display, 20, 20, 30, 200, COLOR_RED);
    gc9a01_fill_rect(display, 55, 20, 30, 200, COLOR_GREEN);
    gc9a01_fill_rect(display, 90, 20, 30, 200, COLOR_BLUE);
    gc9a01_fill_rect(display, 125, 20, 30, 200, COLOR_YELLOW);
    gc9a01_fill_rect(display, 160, 20, 30, 200, COLOR_CYAN);
    gc9a01_fill_rect(display, 195, 20, 30, 200, COLOR_MAGENTA);

    // Label
    gc9a01_draw_string_centered(display, 225, "COLOR TEST", COLOR_WHITE, COLOR_BLACK, 1);
}

int main(void) {
    // Initialize stdio (for debug if needed)
    stdio_init_all();

    // Initialize CYW43 for LED control (Pico 2 W)
    if (cyw43_arch_init()) {
        // Failed - can't do much without LED
        while (1) tight_loop_contents();
    }

    // Startup blink
    led_blink(3, 100);

    // Initialize GC9A01 display on SPI1
    // Note: gc9a01_init() initializes both displays,
    // use gc9a01_init_display() for single display
    gc9a01_init_display(DISPLAY_0);

    // LED on to indicate init complete
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);

    // Demo sequence
    uint32_t demo_phase = 0;
    uint32_t phase_start = 0;

    while (true) {
        uint32_t now = to_ms_since_boot(get_absolute_time());

        switch (demo_phase) {
            case 0:
                // Phase 0: Color test (3 seconds)
                if (phase_start == 0) {
                    phase_start = now;
                    gc9a01_fill(DISPLAY_0, COLOR_BLACK);
                    draw_color_test(DISPLAY_0);
                }
                if (now - phase_start > 3000) {
                    demo_phase = 1;
                    phase_start = 0;
                }
                break;

            case 1:
                // Phase 1: Rainbow (3 seconds)
                if (phase_start == 0) {
                    phase_start = now;
                    gc9a01_fill(DISPLAY_0, COLOR_BLACK);
                    draw_rainbow(DISPLAY_0);
                    gc9a01_draw_string_centered(DISPLAY_0, 110, "GC9A01", COLOR_WHITE, COLOR_MAGENTA, 2);
                }
                if (now - phase_start > 3000) {
                    demo_phase = 2;
                    phase_start = 0;
                }
                break;

            case 2:
                // Phase 2: Hello World (3 seconds)
                if (phase_start == 0) {
                    phase_start = now;
                    draw_hello_screen(DISPLAY_0);
                }
                if (now - phase_start > 3000) {
                    demo_phase = 3;
                    phase_start = 0;
                }
                break;

            case 3:
                // Phase 3: Bouncing ball animation (10 seconds)
                if (phase_start == 0) {
                    phase_start = now;
                    gc9a01_fill(DISPLAY_0, COLOR_BLACK);
                    gc9a01_circle(DISPLAY_0, 120, 120, 118, COLOR_DARK_GRAY);
                }
                animate_ball(DISPLAY_0);
                sleep_ms(20);  // ~50 FPS

                if (now - phase_start > 10000) {
                    demo_phase = 0;  // Loop back
                    phase_start = 0;
                    frame = 0;
                }
                break;
        }

        // Toggle LED slowly as heartbeat
        static uint32_t last_led_toggle = 0;
        if (now - last_led_toggle > 500) {
            static bool led_state = false;
            led_state = !led_state;
            cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, led_state);
            last_led_toggle = now;
        }
    }

    return 0;
}
