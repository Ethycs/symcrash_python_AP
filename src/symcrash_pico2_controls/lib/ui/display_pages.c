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
static int16_t g_displayed_values[8] = {-1, -1, -1, -1, -1, -1, -1, -1};  // Track what's shown
static uint8_t g_current_preset = 0;
static const char *g_preset_name = "Init";
static bool g_needs_full_redraw = true;  // Full screen redraw needed

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

// Helper to draw a single pot row
static void draw_pot_row(int i, bool force) {
    // Skip if value hasn't changed (unless forced)
    if (!force && g_macro_values[i] == g_displayed_values[i]) {
        return;
    }
    g_displayed_values[i] = g_macro_values[i];

    const char *pot_names[] = {"P0", "P1", "P2", "P3", "P4", "P5"};
    uint16_t y = 60 + i * 28;

    // Convert Q15 to percentage
    int percent = (g_macro_values[i] * 100) / 32767;

    // Format line
    char line[20];
    snprintf(line, sizeof(line), "%s %3d%%", pot_names[i], percent);

    // Color based on value
    color_t color;
    if (percent < 25) color = COLOR_BLUE;
    else if (percent < 50) color = COLOR_CYAN;
    else if (percent < 75) color = COLOR_GREEN;
    else color = COLOR_YELLOW;

    // Clear text area and redraw
    gc9a01_fill_rect(DISPLAY_0, 20, y, 115, 20, COLOR_BLACK);
    gc9a01_draw_string(DISPLAY_0, 20, y, line, color, COLOR_BLACK, 2);

    // Draw bar graph
    uint16_t bar_x = 140;
    uint16_t bar_w = 80;
    uint16_t bar_h = 20;
    uint16_t filled = (g_macro_values[i] * bar_w) / 32767;

    // Background (full bar)
    gc9a01_fill_rect(DISPLAY_0, bar_x, y, bar_w, bar_h, COLOR_DARK_GRAY);
    // Filled portion
    if (filled > 0) {
        gc9a01_fill_rect(DISPLAY_0, bar_x, y, filled, bar_h, color);
    }
}

static void render_status_page(void) {
    if (g_needs_full_redraw) {
        // Full screen redraw
        gc9a01_fill(DISPLAY_0, COLOR_BLACK);

        // Title
        gc9a01_draw_string_centered(DISPLAY_0, 10, "SYMCRASH", COLOR_CYAN, COLOR_BLACK, 2);

        // Connection status
        color_t status_color = g_connected ? COLOR_GREEN : COLOR_RED;
        const char *status_text = g_connected ? "CONNECTED" : "OFFLINE";
        gc9a01_draw_string_centered(DISPLAY_0, 40, status_text, status_color, COLOR_BLACK, 1);

        // Draw all pot values
        for (int i = 0; i < 6; i++) {
            draw_pot_row(i, true);
        }

        g_needs_full_redraw = false;
    } else {
        // Incremental update - only redraw changed pots
        for (int i = 0; i < 6; i++) {
            draw_pot_row(i, false);
        }
    }
}

static void render_macros_page(void) {
    // Same as status page for now (single display mode)
    render_status_page();
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
    g_needs_full_redraw = true;
    g_connected = false;
    g_current_preset = 0;

    for (int i = 0; i < 8; i++) {
        g_macro_values[i] = 0;
        g_displayed_values[i] = -1;  // Force redraw on first render
    }
}

void display_pages_render(void) {
    // Always call render - it handles incremental updates internally
    menu_page_t page = menu_get_page();

    switch (page) {
        case PAGE_STATUS:
        case PAGE_MACROS:
        case PAGE_SETTINGS:
            render_status_page();
            break;

        case PAGE_PRESET:
            render_preset_page();
            break;

        default:
            render_status_page();
            break;
    }
}

void display_pages_invalidate(void) {
    g_needs_full_redraw = true;
}

void display_pages_set_connected(bool connected) {
    if (g_connected != connected) {
        g_connected = connected;
        g_needs_full_redraw = true;  // Connection change needs full redraw
    }
}

void display_pages_update_macro(uint8_t macro_id, int16_t value) {
    if (macro_id < 8) {
        g_macro_values[macro_id] = value;
        // Don't set g_needs_full_redraw - incremental update will handle it
    }
}

void display_pages_set_preset(uint8_t preset_idx, const char *name) {
    if (preset_idx != g_current_preset) {
        g_current_preset = preset_idx;
        g_needs_full_redraw = true;  // Preset change needs full redraw
    }
    if (name != NULL) {
        g_preset_name = name;
    }
}
