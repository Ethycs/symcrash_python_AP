/**
 * @file display_pages.h
 * @brief Display page rendering
 */

#ifndef SYMCRASH_DISPLAY_PAGES_H
#define SYMCRASH_DISPLAY_PAGES_H

#include <stdint.h>
#include "gc9a01.h"

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize display pages
 */
void display_pages_init(void);

// =============================================================================
// Rendering
// =============================================================================

/**
 * @brief Render current page to both displays
 *
 * Display 0: Main info (preset, status)
 * Display 1: Parameter values
 */
void display_pages_render(void);

/**
 * @brief Force full redraw of current page
 */
void display_pages_invalidate(void);

// =============================================================================
// Status Updates
// =============================================================================

/**
 * @brief Update connection status
 *
 * @param connected     true if connected
 */
void display_pages_set_connected(bool connected);

/**
 * @brief Update macro value for display
 *
 * @param macro_id      Macro ID (0-7)
 * @param value         Q15 value
 */
void display_pages_update_macro(uint8_t macro_id, int16_t value);

/**
 * @brief Update preset name
 *
 * @param preset_idx    Preset index
 * @param name          Preset name
 */
void display_pages_set_preset(uint8_t preset_idx, const char *name);

#endif // SYMCRASH_DISPLAY_PAGES_H
