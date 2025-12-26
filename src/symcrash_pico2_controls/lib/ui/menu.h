/**
 * @file menu.h
 * @brief Menu navigation system
 */

#ifndef SYMCRASH_MENU_H
#define SYMCRASH_MENU_H

#include <stdint.h>
#include <stdbool.h>

// =============================================================================
// Types
// =============================================================================

/** Menu page IDs */
typedef enum {
    PAGE_STATUS,        // Connection status, current preset
    PAGE_MACROS,        // Macro knob values
    PAGE_PRESET,        // Preset browser
    PAGE_SETTINGS,      // Configuration
    PAGE_COUNT
} menu_page_t;

// =============================================================================
// Navigation
// =============================================================================

/**
 * @brief Initialize menu system
 */
void menu_init(void);

/**
 * @brief Get current page
 */
menu_page_t menu_get_page(void);

/**
 * @brief Set current page
 */
void menu_set_page(menu_page_t page);

/**
 * @brief Navigate to next page
 */
void menu_next_page(void);

/**
 * @brief Navigate to previous page
 */
void menu_prev_page(void);

/**
 * @brief Handle encoder input for current page
 *
 * @param delta     Encoder delta
 */
void menu_encoder_input(int8_t delta);

/**
 * @brief Handle button input for current page
 *
 * @param action    Button action (enter/back/etc)
 */
void menu_button_input(uint8_t action);

#endif // SYMCRASH_MENU_H
