/**
 * @file gc9a01.c
 * @brief GC9A01 round LCD driver implementation
 */

#include "gc9a01.h"
#include "config.h"

#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include <string.h>
#include <math.h>

// =============================================================================
// Display Pin Configuration
// =============================================================================

typedef struct {
    uint8_t cs_pin;
    uint8_t dc_pin;
    uint8_t rst_pin;
} display_pins_t;

static const display_pins_t g_displays[2] = {
    { LCD0_CS_PIN, LCD0_DC_PIN, LCD0_RST_PIN },
    { LCD1_CS_PIN, LCD1_DC_PIN, LCD1_RST_PIN }
};

// =============================================================================
// GC9A01 Commands
// =============================================================================

#define GC9A01_SWRESET  0x01
#define GC9A01_SLPIN    0x10
#define GC9A01_SLPOUT   0x11
#define GC9A01_INVOFF   0x20
#define GC9A01_INVON    0x21
#define GC9A01_DISPOFF  0x28
#define GC9A01_DISPON   0x29
#define GC9A01_CASET    0x2A
#define GC9A01_RASET    0x2B
#define GC9A01_RAMWR    0x2C
#define GC9A01_COLMOD   0x3A
#define GC9A01_MADCTL   0x36

// MADCTL bits
#define MADCTL_MY       0x80
#define MADCTL_MX       0x40
#define MADCTL_MV       0x20
#define MADCTL_ML       0x10
#define MADCTL_BGR      0x08
#define MADCTL_MH       0x04

// =============================================================================
// Simple 5x7 Font
// =============================================================================

static const uint8_t font5x7[][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, // space
    {0x00, 0x00, 0x5F, 0x00, 0x00}, // !
    {0x00, 0x07, 0x00, 0x07, 0x00}, // "
    {0x14, 0x7F, 0x14, 0x7F, 0x14}, // #
    {0x24, 0x2A, 0x7F, 0x2A, 0x12}, // $
    {0x23, 0x13, 0x08, 0x64, 0x62}, // %
    {0x36, 0x49, 0x55, 0x22, 0x50}, // &
    {0x00, 0x05, 0x03, 0x00, 0x00}, // '
    {0x00, 0x1C, 0x22, 0x41, 0x00}, // (
    {0x00, 0x41, 0x22, 0x1C, 0x00}, // )
    {0x08, 0x2A, 0x1C, 0x2A, 0x08}, // *
    {0x08, 0x08, 0x3E, 0x08, 0x08}, // +
    {0x00, 0x50, 0x30, 0x00, 0x00}, // ,
    {0x08, 0x08, 0x08, 0x08, 0x08}, // -
    {0x00, 0x60, 0x60, 0x00, 0x00}, // .
    {0x20, 0x10, 0x08, 0x04, 0x02}, // /
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 0
    {0x00, 0x42, 0x7F, 0x40, 0x00}, // 1
    {0x42, 0x61, 0x51, 0x49, 0x46}, // 2
    {0x21, 0x41, 0x45, 0x4B, 0x31}, // 3
    {0x18, 0x14, 0x12, 0x7F, 0x10}, // 4
    {0x27, 0x45, 0x45, 0x45, 0x39}, // 5
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 6
    {0x01, 0x71, 0x09, 0x05, 0x03}, // 7
    {0x36, 0x49, 0x49, 0x49, 0x36}, // 8
    {0x06, 0x49, 0x49, 0x29, 0x1E}, // 9
    {0x00, 0x36, 0x36, 0x00, 0x00}, // :
    {0x00, 0x56, 0x36, 0x00, 0x00}, // ;
    {0x00, 0x08, 0x14, 0x22, 0x41}, // <
    {0x14, 0x14, 0x14, 0x14, 0x14}, // =
    {0x41, 0x22, 0x14, 0x08, 0x00}, // >
    {0x02, 0x01, 0x51, 0x09, 0x06}, // ?
    {0x32, 0x49, 0x79, 0x41, 0x3E}, // @
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, // A
    {0x7F, 0x49, 0x49, 0x49, 0x36}, // B
    {0x3E, 0x41, 0x41, 0x41, 0x22}, // C
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, // D
    {0x7F, 0x49, 0x49, 0x49, 0x41}, // E
    {0x7F, 0x09, 0x09, 0x01, 0x01}, // F
    {0x3E, 0x41, 0x41, 0x51, 0x32}, // G
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, // H
    {0x00, 0x41, 0x7F, 0x41, 0x00}, // I
    {0x20, 0x40, 0x41, 0x3F, 0x01}, // J
    {0x7F, 0x08, 0x14, 0x22, 0x41}, // K
    {0x7F, 0x40, 0x40, 0x40, 0x40}, // L
    {0x7F, 0x02, 0x04, 0x02, 0x7F}, // M
    {0x7F, 0x04, 0x08, 0x10, 0x7F}, // N
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, // O
    {0x7F, 0x09, 0x09, 0x09, 0x06}, // P
    {0x3E, 0x41, 0x51, 0x21, 0x5E}, // Q
    {0x7F, 0x09, 0x19, 0x29, 0x46}, // R
    {0x46, 0x49, 0x49, 0x49, 0x31}, // S
    {0x01, 0x01, 0x7F, 0x01, 0x01}, // T
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, // U
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, // V
    {0x7F, 0x20, 0x18, 0x20, 0x7F}, // W
    {0x63, 0x14, 0x08, 0x14, 0x63}, // X
    {0x03, 0x04, 0x78, 0x04, 0x03}, // Y
    {0x61, 0x51, 0x49, 0x45, 0x43}, // Z
};

// =============================================================================
// Low-Level SPI
// =============================================================================

static void select_display(display_id_t display) {
    // Wait for any in-flight SPI to complete
    while (spi_is_busy(LCD_SPI_ID)) {
        tight_loop_contents();
    }

    // ALWAYS explicitly drive both CS pins - no optimization
    // This ensures the non-selected display CS is definitely HIGH
    gpio_put(g_displays[DISPLAY_0].cs_pin, 1);
    gpio_put(g_displays[DISPLAY_1].cs_pin, 1);

    // Delay to ensure CS lines fully settle
    for (volatile int i = 0; i < 20; i++) {
        __asm volatile ("nop");
    }

    // Now select the target display
    gpio_put(g_displays[display].cs_pin, 0);

    // Delay after selecting
    for (volatile int i = 0; i < 20; i++) {
        __asm volatile ("nop");
    }
}

static void deselect_display(display_id_t display) {
    // Wait for SPI to complete before deselecting
    while (spi_is_busy(LCD_SPI_ID)) {
        tight_loop_contents();
    }

    // Deselect the specified display
    gpio_put(g_displays[display].cs_pin, 1);

    // Also ensure the OTHER display is deselected
    display_id_t other = (display == DISPLAY_0) ? DISPLAY_1 : DISPLAY_0;
    gpio_put(g_displays[other].cs_pin, 1);
}

static void set_dc_command(display_id_t display) {
    gpio_put(g_displays[display].dc_pin, 0);
}

static void set_dc_data(display_id_t display) {
    gpio_put(g_displays[display].dc_pin, 1);
}

static void write_command(display_id_t display, uint8_t cmd) {
    select_display(display);
    set_dc_command(display);
    spi_write_blocking(LCD_SPI_ID, &cmd, 1);
    deselect_display(display);
}

static void write_data(display_id_t display, const uint8_t *data, size_t len) {
    select_display(display);
    set_dc_data(display);
    spi_write_blocking(LCD_SPI_ID, data, len);
    deselect_display(display);
}

static void write_command_data(display_id_t display, uint8_t cmd,
                                const uint8_t *data, size_t len) {
    write_command(display, cmd);
    if (len > 0 && data != NULL) {
        write_data(display, data, len);
    }
}

// =============================================================================
// Initialization
// =============================================================================

static void init_display_hw(display_id_t display) {
    const display_pins_t *pins = &g_displays[display];

    // Configure pins
    gpio_init(pins->cs_pin);
    gpio_init(pins->dc_pin);
    gpio_init(pins->rst_pin);

    gpio_set_dir(pins->cs_pin, GPIO_OUT);
    gpio_set_dir(pins->dc_pin, GPIO_OUT);
    gpio_set_dir(pins->rst_pin, GPIO_OUT);

    gpio_put(pins->cs_pin, 1);  // CS high (inactive)
    gpio_put(pins->dc_pin, 0);
    gpio_put(pins->rst_pin, 1);

    // Hardware reset (matching pot_display timing)
    sleep_ms(50);
    gpio_put(pins->rst_pin, 0);
    sleep_ms(50);
    gpio_put(pins->rst_pin, 1);
    sleep_ms(150);
}

// Additional GC9A01 registers for full Adafruit init
#define GC9A01_TEON      0x35
#define GC9A01_POWER2    0xC3
#define GC9A01_POWER3    0xC4
#define GC9A01_POWER4    0xC9
#define GC9A01_FRAMERATE 0xE8
#define GC9A01_INREGEN1  0xFE
#define GC9A01_INREGEN2  0xEF
#define GC9A01_GAMMA1    0xF0
#define GC9A01_GAMMA2    0xF1
#define GC9A01_GAMMA3    0xF2
#define GC9A01_GAMMA4    0xF3

static void init_display_registers(display_id_t display, bool rotate_180) {
    // Full Adafruit GC9A01A initialization sequence
    // Required for proper display operation - minimal init doesn't work

    // Internal register enable sequence
    write_command(display, GC9A01_INREGEN2);
    uint8_t eb_val = 0x14;
    write_command_data(display, 0xEB, &eb_val, 1);
    write_command(display, GC9A01_INREGEN1);
    write_command(display, GC9A01_INREGEN2);
    write_command_data(display, 0xEB, &eb_val, 1);

    // Undocumented registers 0x84-0x8F
    uint8_t r84 = 0x40; write_command_data(display, 0x84, &r84, 1);
    uint8_t r85 = 0xFF; write_command_data(display, 0x85, &r85, 1);
    uint8_t r86 = 0xFF; write_command_data(display, 0x86, &r86, 1);
    uint8_t r87 = 0xFF; write_command_data(display, 0x87, &r87, 1);
    uint8_t r88 = 0x0A; write_command_data(display, 0x88, &r88, 1);
    uint8_t r89 = 0x21; write_command_data(display, 0x89, &r89, 1);
    uint8_t r8a = 0x00; write_command_data(display, 0x8A, &r8a, 1);
    uint8_t r8b = 0x80; write_command_data(display, 0x8B, &r8b, 1);
    uint8_t r8c = 0x01; write_command_data(display, 0x8C, &r8c, 1);
    uint8_t r8d = 0x01; write_command_data(display, 0x8D, &r8d, 1);
    uint8_t r8e = 0xFF; write_command_data(display, 0x8E, &r8e, 1);
    uint8_t r8f = 0xFF; write_command_data(display, 0x8F, &r8f, 1);

    // Display function control
    uint8_t b6[] = {0x00, 0x00};
    write_command_data(display, 0xB6, b6, 2);

    // Memory access control (orientation)
    // 0x48 = MX | BGR (normal)
    // 0x88 = MY | BGR (Y-flip only, for Display 1)
    uint8_t madctl = rotate_180 ? 0x88 : 0x48;
    write_command_data(display, GC9A01_MADCTL, &madctl, 1);

    // Pixel format: 16-bit RGB565
    uint8_t colmod = 0x05;
    write_command_data(display, GC9A01_COLMOD, &colmod, 1);

    // Undocumented register 0x90
    uint8_t r90[] = {0x08, 0x08, 0x08, 0x08};
    write_command_data(display, 0x90, r90, 4);

    uint8_t rbd = 0x06; write_command_data(display, 0xBD, &rbd, 1);
    uint8_t rbc = 0x00; write_command_data(display, 0xBC, &rbc, 1);

    uint8_t rff[] = {0x60, 0x01, 0x04};
    write_command_data(display, 0xFF, rff, 3);

    // Power control
    uint8_t pc2 = 0x13; write_command_data(display, GC9A01_POWER2, &pc2, 1);
    uint8_t pc3 = 0x13; write_command_data(display, GC9A01_POWER3, &pc3, 1);
    uint8_t pc4 = 0x22; write_command_data(display, GC9A01_POWER4, &pc4, 1);

    uint8_t rbe = 0x11; write_command_data(display, 0xBE, &rbe, 1);

    uint8_t re1[] = {0x10, 0x0E};
    write_command_data(display, 0xE1, re1, 2);

    uint8_t rdf[] = {0x21, 0x0C, 0x02};
    write_command_data(display, 0xDF, rdf, 3);

    // Gamma settings
    uint8_t g1[] = {0x45, 0x09, 0x08, 0x08, 0x26, 0x2A};
    write_command_data(display, GC9A01_GAMMA1, g1, 6);

    uint8_t g2[] = {0x43, 0x70, 0x72, 0x36, 0x37, 0x6F};
    write_command_data(display, GC9A01_GAMMA2, g2, 6);

    write_command_data(display, GC9A01_GAMMA3, g1, 6);
    write_command_data(display, GC9A01_GAMMA4, g2, 6);

    uint8_t red[] = {0x1B, 0x0B};
    write_command_data(display, 0xED, red, 2);

    uint8_t rae = 0x77; write_command_data(display, 0xAE, &rae, 1);
    uint8_t rcd = 0x63; write_command_data(display, 0xCD, &rcd, 1);

    // Frame rate
    uint8_t fr = 0x34;
    write_command_data(display, GC9A01_FRAMERATE, &fr, 1);

    // More undocumented registers for display timing
    uint8_t r62[] = {0x18, 0x0D, 0x71, 0xED, 0x70, 0x70, 0x18, 0x0F, 0x71, 0xEF, 0x70, 0x70};
    write_command_data(display, 0x62, r62, 12);

    uint8_t r63[] = {0x18, 0x11, 0x71, 0xF1, 0x70, 0x70, 0x18, 0x13, 0x71, 0xF3, 0x70, 0x70};
    write_command_data(display, 0x63, r63, 12);

    uint8_t r64[] = {0x28, 0x29, 0xF1, 0x01, 0xF1, 0x00, 0x07};
    write_command_data(display, 0x64, r64, 7);

    uint8_t r66[] = {0x3C, 0x00, 0xCD, 0x67, 0x45, 0x45, 0x10, 0x00, 0x00, 0x00};
    write_command_data(display, 0x66, r66, 10);

    uint8_t r67[] = {0x00, 0x3C, 0x00, 0x00, 0x00, 0x01, 0x54, 0x10, 0x32, 0x98};
    write_command_data(display, 0x67, r67, 10);

    uint8_t r74[] = {0x10, 0x85, 0x80, 0x00, 0x00, 0x4E, 0x00};
    write_command_data(display, 0x74, r74, 7);

    uint8_t r98[] = {0x3E, 0x07};
    write_command_data(display, 0x98, r98, 2);

    // Tearing effect line on
    write_command(display, GC9A01_TEON);

    // Inversion on (required for correct colors)
    write_command(display, GC9A01_INVON);

    // Exit sleep mode
    write_command(display, GC9A01_SLPOUT);
    sleep_ms(150);

    // Display on
    write_command(display, GC9A01_DISPON);
    sleep_ms(150);
}

void gc9a01_init(void) {
    // Initialize SPI with explicit format (required for GC9A01)
    spi_init(LCD_SPI_ID, SPI1_BAUD);
    spi_set_format(LCD_SPI_ID, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST);

    gpio_set_function(SPI1_SCK_PIN, GPIO_FUNC_SPI);
    gpio_set_function(SPI1_MOSI_PIN, GPIO_FUNC_SPI);

    // Initialize GPIO pins for BOTH displays first (before any reset)
    // This is important because they share DC and RST pins
    for (int i = 0; i < 2; i++) {
        const display_pins_t *pins = &g_displays[i];
        gpio_init(pins->cs_pin);
        gpio_set_dir(pins->cs_pin, GPIO_OUT);
        gpio_put(pins->cs_pin, 1);  // CS high (inactive)
        // Enable pullup as backup to keep CS high when not driven
        gpio_pull_up(pins->cs_pin);
        // Set high drive strength for cleaner edges
        gpio_set_drive_strength(pins->cs_pin, GPIO_DRIVE_STRENGTH_12MA);
    }

    // Shared DC and RST - only init once
    gpio_init(LCD0_DC_PIN);
    gpio_set_dir(LCD0_DC_PIN, GPIO_OUT);
    gpio_put(LCD0_DC_PIN, 0);

    gpio_init(LCD0_RST_PIN);
    gpio_set_dir(LCD0_RST_PIN, GPIO_OUT);
    gpio_put(LCD0_RST_PIN, 1);

    // Single hardware reset for BOTH displays (shared RST line)
    sleep_ms(50);
    gpio_put(LCD0_RST_PIN, 0);
    sleep_ms(50);
    gpio_put(LCD0_RST_PIN, 1);
    sleep_ms(150);

    // Now initialize registers for each display (no more resets)
    // Display 0: normal orientation
    init_display_registers(DISPLAY_0, false);
    gc9a01_fill(DISPLAY_0, COLOR_BLACK);

    // Display 1: rotated 180 degrees
    init_display_registers(DISPLAY_1, true);
    gc9a01_fill(DISPLAY_1, COLOR_BLACK);
}

void gc9a01_init_display(display_id_t display) {
    init_display_hw(display);
    // Display 1 is rotated 180 degrees
    init_display_registers(display, display == DISPLAY_1);

    // Clear to black
    gc9a01_fill(display, COLOR_BLACK);
}

// =============================================================================
// Drawing
// =============================================================================

void gc9a01_set_window(display_id_t display, uint16_t x0, uint16_t y0,
                       uint16_t x1, uint16_t y1) {
    // IMPORTANT: Keep CS asserted throughout the entire window setup
    // The GC9A01 expects CS to stay low during CASET/RASET/RAMWR sequence
    // Toggling CS between commands can cause crosstalk issues with dual displays

    select_display(display);

    uint8_t data[4];

    // Column address set
    set_dc_command(display);
    uint8_t cmd = GC9A01_CASET;
    spi_write_blocking(LCD_SPI_ID, &cmd, 1);
    set_dc_data(display);
    data[0] = x0 >> 8;
    data[1] = x0 & 0xFF;
    data[2] = x1 >> 8;
    data[3] = x1 & 0xFF;
    spi_write_blocking(LCD_SPI_ID, data, 4);

    // Row address set
    set_dc_command(display);
    cmd = GC9A01_RASET;
    spi_write_blocking(LCD_SPI_ID, &cmd, 1);
    set_dc_data(display);
    data[0] = y0 >> 8;
    data[1] = y0 & 0xFF;
    data[2] = y1 >> 8;
    data[3] = y1 & 0xFF;
    spi_write_blocking(LCD_SPI_ID, data, 4);

    // RAM write command - leave CS asserted and DC in data mode
    // for the subsequent pixel writes
    set_dc_command(display);
    cmd = GC9A01_RAMWR;
    spi_write_blocking(LCD_SPI_ID, &cmd, 1);
    set_dc_data(display);

    // Note: CS stays LOW - caller must call deselect_display after writing pixels
}

void gc9a01_write_pixels(display_id_t display, const uint16_t *data, size_t len) {
    select_display(display);
    set_dc_data(display);

    // Convert to big-endian and send
    for (size_t i = 0; i < len; i++) {
        uint8_t buf[2] = { data[i] >> 8, data[i] & 0xFF };
        spi_write_blocking(LCD_SPI_ID, buf, 2);
    }

    deselect_display(display);
}

void gc9a01_fill(display_id_t display, color_t color) {
    gc9a01_set_window(display, 0, 0, LCD_WIDTH - 1, LCD_HEIGHT - 1);
    // set_window leaves CS asserted and DC in data mode

    uint8_t buf[2] = { color >> 8, color & 0xFF };
    for (uint32_t i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        spi_write_blocking(LCD_SPI_ID, buf, 2);
    }

    deselect_display(display);
}

void gc9a01_set_pixel(display_id_t display, uint16_t x, uint16_t y, color_t color) {
    if (x >= LCD_WIDTH || y >= LCD_HEIGHT) return;

    gc9a01_set_window(display, x, y, x, y);
    // set_window leaves CS asserted and DC in data mode

    uint8_t buf[2] = { color >> 8, color & 0xFF };
    spi_write_blocking(LCD_SPI_ID, buf, 2);

    deselect_display(display);
}

void gc9a01_fill_rect(display_id_t display, uint16_t x, uint16_t y,
                      uint16_t w, uint16_t h, color_t color) {
    if (x >= LCD_WIDTH || y >= LCD_HEIGHT) return;
    if (x + w > LCD_WIDTH) w = LCD_WIDTH - x;
    if (y + h > LCD_HEIGHT) h = LCD_HEIGHT - y;

    gc9a01_set_window(display, x, y, x + w - 1, y + h - 1);
    // set_window leaves CS asserted and DC in data mode

    uint8_t buf[2] = { color >> 8, color & 0xFF };
    for (uint32_t i = 0; i < w * h; i++) {
        spi_write_blocking(LCD_SPI_ID, buf, 2);
    }

    deselect_display(display);
}

void gc9a01_hline(display_id_t display, uint16_t x, uint16_t y,
                  uint16_t w, color_t color) {
    gc9a01_fill_rect(display, x, y, w, 1, color);
}

void gc9a01_vline(display_id_t display, uint16_t x, uint16_t y,
                  uint16_t h, color_t color) {
    gc9a01_fill_rect(display, x, y, 1, h, color);
}

void gc9a01_circle(display_id_t display, uint16_t cx, uint16_t cy,
                   uint16_t r, color_t color) {
    int16_t x = r;
    int16_t y = 0;
    int16_t err = 0;

    while (x >= y) {
        gc9a01_set_pixel(display, cx + x, cy + y, color);
        gc9a01_set_pixel(display, cx + y, cy + x, color);
        gc9a01_set_pixel(display, cx - y, cy + x, color);
        gc9a01_set_pixel(display, cx - x, cy + y, color);
        gc9a01_set_pixel(display, cx - x, cy - y, color);
        gc9a01_set_pixel(display, cx - y, cy - x, color);
        gc9a01_set_pixel(display, cx + y, cy - x, color);
        gc9a01_set_pixel(display, cx + x, cy - y, color);

        y++;
        err += 1 + 2 * y;
        if (2 * (err - x) + 1 > 0) {
            x--;
            err += 1 - 2 * x;
        }
    }
}

void gc9a01_fill_circle(display_id_t display, uint16_t cx, uint16_t cy,
                        uint16_t r, color_t color) {
    for (int16_t y = -r; y <= r; y++) {
        int16_t x = (int16_t)sqrt(r * r - y * y);
        gc9a01_hline(display, cx - x, cy + y, 2 * x + 1, color);
    }
}

// =============================================================================
// Text
// =============================================================================

void gc9a01_draw_char(display_id_t display, uint16_t x, uint16_t y,
                      char c, color_t fg, color_t bg, uint8_t size) {
    if (c < ' ' || c > 'Z') c = ' ';
    int idx = c - ' ';
    if (idx >= (int)(sizeof(font5x7) / sizeof(font5x7[0]))) idx = 0;

    for (uint8_t col = 0; col < 5; col++) {
        uint8_t line = font5x7[idx][col];
        for (uint8_t row = 0; row < 7; row++) {
            color_t color = (line & (1 << row)) ? fg : bg;
            if (size == 1) {
                gc9a01_set_pixel(display, x + col, y + row, color);
            } else {
                gc9a01_fill_rect(display, x + col * size, y + row * size,
                                 size, size, color);
            }
        }
    }

    // Inter-character space
    gc9a01_fill_rect(display, x + 5 * size, y, size, 7 * size, bg);
}

void gc9a01_draw_string(display_id_t display, uint16_t x, uint16_t y,
                        const char *str, color_t fg, color_t bg, uint8_t size) {
    while (*str) {
        gc9a01_draw_char(display, x, y, *str, fg, bg, size);
        x += 6 * size;  // 5 pixels + 1 space
        str++;
    }
}

void gc9a01_draw_string_centered(display_id_t display, uint16_t y,
                                  const char *str, color_t fg, color_t bg,
                                  uint8_t size) {
    int len = strlen(str);
    uint16_t width = len * 6 * size;
    uint16_t x = (LCD_WIDTH - width) / 2;
    gc9a01_draw_string(display, x, y, str, fg, bg, size);
}

// =============================================================================
// Control
// =============================================================================

void gc9a01_set_brightness(display_id_t display, uint8_t brightness) {
    (void)display;
    (void)brightness;
    // Note: GC9A01 doesn't have brightness control via SPI
    // Brightness control requires external PWM on backlight
}

void gc9a01_power(display_id_t display, bool on) {
    write_command(display, on ? GC9A01_DISPON : GC9A01_DISPOFF);
}
