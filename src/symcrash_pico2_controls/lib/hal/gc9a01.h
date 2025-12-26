/**
 * @file gc9a01.h
 * @brief GC9A01 round LCD driver for dual displays
 *
 * 240x240 RGB565 round displays on SPI1.
 */

#ifndef SYMCRASH_GC9A01_H
#define SYMCRASH_GC9A01_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "config.h"

// =============================================================================
// Types
// =============================================================================

/** Display ID */
typedef enum {
    DISPLAY_0 = 0,
    DISPLAY_1 = 1
} display_id_t;

/** RGB565 color */
typedef uint16_t color_t;

// =============================================================================
// Color Definitions (RGB565)
// =============================================================================

#define COLOR_BLACK     0x0000
#define COLOR_WHITE     0xFFFF
#define COLOR_RED       0xF800
#define COLOR_GREEN     0x07E0
#define COLOR_BLUE      0x001F
#define COLOR_YELLOW    0xFFE0
#define COLOR_CYAN      0x07FF
#define COLOR_MAGENTA   0xF81F
#define COLOR_ORANGE    0xFD20
#define COLOR_GRAY      0x8410
#define COLOR_DARK_GRAY 0x4208

/** Create RGB565 color from 8-bit components */
static inline color_t color_rgb(uint8_t r, uint8_t g, uint8_t b) {
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
}

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize both GC9A01 displays
 */
void gc9a01_init(void);

/**
 * @brief Initialize a single display
 *
 * @param display   Display ID
 */
void gc9a01_init_display(display_id_t display);

// =============================================================================
// Drawing
// =============================================================================

/**
 * @brief Fill entire display with a color
 *
 * @param display   Display ID
 * @param color     Fill color (RGB565)
 */
void gc9a01_fill(display_id_t display, color_t color);

/**
 * @brief Set a single pixel
 *
 * @param display   Display ID
 * @param x         X coordinate (0-239)
 * @param y         Y coordinate (0-239)
 * @param color     Pixel color
 */
void gc9a01_set_pixel(display_id_t display, uint16_t x, uint16_t y, color_t color);

/**
 * @brief Fill a rectangle
 *
 * @param display   Display ID
 * @param x         Top-left X
 * @param y         Top-left Y
 * @param w         Width
 * @param h         Height
 * @param color     Fill color
 */
void gc9a01_fill_rect(display_id_t display, uint16_t x, uint16_t y,
                      uint16_t w, uint16_t h, color_t color);

/**
 * @brief Draw a horizontal line
 *
 * @param display   Display ID
 * @param x         Start X
 * @param y         Y coordinate
 * @param w         Width
 * @param color     Line color
 */
void gc9a01_hline(display_id_t display, uint16_t x, uint16_t y,
                  uint16_t w, color_t color);

/**
 * @brief Draw a vertical line
 *
 * @param display   Display ID
 * @param x         X coordinate
 * @param y         Start Y
 * @param h         Height
 * @param color     Line color
 */
void gc9a01_vline(display_id_t display, uint16_t x, uint16_t y,
                  uint16_t h, color_t color);

/**
 * @brief Draw a circle outline
 *
 * @param display   Display ID
 * @param cx        Center X
 * @param cy        Center Y
 * @param r         Radius
 * @param color     Line color
 */
void gc9a01_circle(display_id_t display, uint16_t cx, uint16_t cy,
                   uint16_t r, color_t color);

/**
 * @brief Fill a circle
 *
 * @param display   Display ID
 * @param cx        Center X
 * @param cy        Center Y
 * @param r         Radius
 * @param color     Fill color
 */
void gc9a01_fill_circle(display_id_t display, uint16_t cx, uint16_t cy,
                        uint16_t r, color_t color);

// =============================================================================
// Text
// =============================================================================

/**
 * @brief Draw a character
 *
 * @param display   Display ID
 * @param x         X coordinate
 * @param y         Y coordinate
 * @param c         Character to draw
 * @param fg        Foreground color
 * @param bg        Background color
 * @param size      Font size multiplier (1, 2, 3...)
 */
void gc9a01_draw_char(display_id_t display, uint16_t x, uint16_t y,
                      char c, color_t fg, color_t bg, uint8_t size);

/**
 * @brief Draw a string
 *
 * @param display   Display ID
 * @param x         X coordinate
 * @param y         Y coordinate
 * @param str       String to draw
 * @param fg        Foreground color
 * @param bg        Background color
 * @param size      Font size multiplier
 */
void gc9a01_draw_string(display_id_t display, uint16_t x, uint16_t y,
                        const char *str, color_t fg, color_t bg, uint8_t size);

/**
 * @brief Draw a centered string
 *
 * @param display   Display ID
 * @param y         Y coordinate
 * @param str       String to draw
 * @param fg        Foreground color
 * @param bg        Background color
 * @param size      Font size multiplier
 */
void gc9a01_draw_string_centered(display_id_t display, uint16_t y,
                                  const char *str, color_t fg, color_t bg,
                                  uint8_t size);

// =============================================================================
// Raw Access
// =============================================================================

/**
 * @brief Set drawing window
 *
 * @param display   Display ID
 * @param x0        Start X
 * @param y0        Start Y
 * @param x1        End X
 * @param y1        End Y
 */
void gc9a01_set_window(display_id_t display, uint16_t x0, uint16_t y0,
                       uint16_t x1, uint16_t y1);

/**
 * @brief Write raw pixel data to current window
 *
 * @param display   Display ID
 * @param data      RGB565 pixel data
 * @param len       Number of pixels
 */
void gc9a01_write_pixels(display_id_t display, const uint16_t *data, size_t len);

// =============================================================================
// Control
// =============================================================================

/**
 * @brief Set display brightness (via PWM backlight, if available)
 *
 * @param display   Display ID
 * @param brightness    0-255
 */
void gc9a01_set_brightness(display_id_t display, uint8_t brightness);

/**
 * @brief Turn display on/off
 *
 * @param display   Display ID
 * @param on        true = on, false = off
 */
void gc9a01_power(display_id_t display, bool on);

#endif // SYMCRASH_GC9A01_H
