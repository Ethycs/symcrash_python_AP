/**
 * @file display_pages.c
 * @brief Display page rendering implementation
 */

#include "display_pages.h"
#include "menu.h"
#include "gc9a01.h"
#include "control_map.h"
#include "dual_uart_link.h"
#include "config.h"
#include "pico/stdlib.h"
#include <stdio.h>
#include <string.h>

// =============================================================================
// State
// =============================================================================

static bool g_connected = false;
static int16_t g_macro_values[8] = {0};
static uint8_t g_current_preset = 0;
static const char *g_preset_name = "Init";
static bool g_needs_redraw = true;

// Preset names (placeholder)
static const char *g_preset_names[] = {
    "Brass Stab",
    "Warm Pad",
    "Pluck",
    "Bass",
    "Lead",
    "Texture",
    "Drone",
    "FX"
};

// =============================================================================
// Drawing Helpers
// =============================================================================

static void draw_arc(display_id_t disp, uint16_t cx, uint16_t cy,
                     uint16_t r, int value, color_t color) {
    // Draw a simple value indicator (filled arc approximation)
    // Value is 0-32767 (Q15)

    // Convert to angle (0-270 degrees, starting at bottom left)
    int angle_deg = (value * 270) / 32767;

    // Draw as a thick arc (multiple circles of different radii)
    for (int dr = -2; dr <= 2; dr++) {
        gc9a01_circle(disp, cx, cy, r + dr, color);
    }

    // Draw indicator line
    // Simplified: just draw a filled segment
    int16_t end_x = cx + ((r - 10) * angle_deg) / 90;  // Rough approximation
    int16_t end_y = cy - ((r - 10) * (90 - angle_deg)) / 90;

    gc9a01_fill_rect(disp, end_x - 3, end_y - 3, 6, 6, color);
}

// =============================================================================
// Page Renderers
// =============================================================================

static void render_status_page(void) {
    // Display 0: Status page
    gc9a01_fill(DISPLAY_0, COLOR_BLACK);

    // Title
    gc9a01_draw_string_centered(DISPLAY_0, 20, "SYMCRASH", COLOR_CYAN, COLOR_BLACK, 2);

    // Connection status
    color_t status_color = g_connected ? COLOR_GREEN : COLOR_RED;
    const char *status_text = g_connected ? "CONNECTED" : "OFFLINE";
    gc9a01_draw_string_centered(DISPLAY_0, 60, status_text, status_color, COLOR_BLACK, 1);

    // Preset info
    char preset_str[32];
    snprintf(preset_str, sizeof(preset_str), "PRESET %d", g_current_preset + 1);
    gc9a01_draw_string_centered(DISPLAY_0, 100, preset_str, COLOR_WHITE, COLOR_BLACK, 2);

    if (g_current_preset < 8) {
        gc9a01_draw_string_centered(DISPLAY_0, 140, g_preset_names[g_current_preset],
                                     COLOR_YELLOW, COLOR_BLACK, 1);
    }

    // Draw decorative circle border
    gc9a01_circle(DISPLAY_0, LCD_WIDTH/2, LCD_HEIGHT/2, 115, COLOR_DARK_GRAY);
    gc9a01_circle(DISPLAY_0, LCD_WIDTH/2, LCD_HEIGHT/2, 116, COLOR_DARK_GRAY);
}

static void render_macros_page(void) {
    // Display 1: Macro values
    gc9a01_fill(DISPLAY_1, COLOR_BLACK);

    // Title
    gc9a01_draw_string_centered(DISPLAY_1, 10, "MACROS", COLOR_WHITE, COLOR_BLACK, 1);

    // Draw 6 macro values in a grid
    const char *macro_names[] = {"CUT", "RES", "DRV", "ATK", "REL", "MIX"};

    for (int i = 0; i < 6; i++) {
        int row = i / 3;
        int col = i % 3;

        int x = 30 + col * 70;
        int y = 40 + row * 90;

        // Macro name
        gc9a01_draw_string(DISPLAY_1, x, y, macro_names[i], COLOR_GRAY, COLOR_BLACK, 1);

        // Value bar
        int bar_width = (g_macro_values[i] * 50) / 32767;
        gc9a01_fill_rect(DISPLAY_1, x, y + 15, 50, 8, COLOR_DARK_GRAY);
        gc9a01_fill_rect(DISPLAY_1, x, y + 15, bar_width, 8, COLOR_CYAN);

        // Value text
        char val_str[8];
        int percent = (g_macro_values[i] * 100) / 32767;
        snprintf(val_str, sizeof(val_str), "%d%%", percent);
        gc9a01_draw_string(DISPLAY_1, x, y + 28, val_str, COLOR_WHITE, COLOR_BLACK, 1);
    }

    // Border
    gc9a01_circle(DISPLAY_1, LCD_WIDTH/2, LCD_HEIGHT/2, 115, COLOR_DARK_GRAY);
}

static void render_preset_page(void) {
    // Display 0: Preset browser
    gc9a01_fill(DISPLAY_0, COLOR_BLACK);

    gc9a01_draw_string_centered(DISPLAY_0, 20, "PRESETS", COLOR_WHITE, COLOR_BLACK, 1);

    // List presets with current highlighted
    for (int i = 0; i < 8; i++) {
        int y = 50 + i * 20;
        color_t fg = (i == g_current_preset) ? COLOR_BLACK : COLOR_WHITE;
        color_t bg = (i == g_current_preset) ? COLOR_CYAN : COLOR_BLACK;

        char line[32];
        snprintf(line, sizeof(line), "%d. %s", i + 1, g_preset_names[i]);

        // Draw with highlight
        if (i == g_current_preset) {
            gc9a01_fill_rect(DISPLAY_0, 30, y - 2, 180, 18, bg);
        }
        gc9a01_draw_string(DISPLAY_0, 35, y, line, fg, bg, 1);
    }
}

// =============================================================================
// Public API
// =============================================================================

void display_pages_init(void) {
    g_needs_redraw = true;
    g_connected = false;
    g_current_preset = 0;

    for (int i = 0; i < 8; i++) {
        g_macro_values[i] = 0;
    }
}

void display_pages_render(void) {
    if (!g_needs_redraw) {
        return;  // Only redraw when needed
    }

    menu_page_t page = menu_get_page();

    switch (page) {
        case PAGE_STATUS:
            render_status_page();
            render_macros_page();
            break;

        case PAGE_MACROS:
            render_status_page();
            render_macros_page();
            break;

        case PAGE_PRESET:
            render_preset_page();
            render_macros_page();
            break;

        case PAGE_SETTINGS:
            // TODO: Settings page
            render_status_page();
            render_macros_page();
            break;

        default:
            break;
    }

    g_needs_redraw = false;
}

void display_pages_invalidate(void) {
    g_needs_redraw = true;
}

void display_pages_set_connected(bool connected) {
    if (g_connected != connected) {
        g_connected = connected;
        g_needs_redraw = true;
    }
}

void display_pages_update_macro(uint8_t macro_id, int16_t value) {
    if (macro_id < 8 && g_macro_values[macro_id] != value) {
        g_macro_values[macro_id] = value;
        g_needs_redraw = true;
    }
}

void display_pages_set_preset(uint8_t preset_idx, const char *name) {
    if (preset_idx != g_current_preset) {
        g_current_preset = preset_idx;
        g_needs_redraw = true;
    }
    if (name != NULL) {
        g_preset_name = name;
    }
}
