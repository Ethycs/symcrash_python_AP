/**
 * @file menu.c
 * @brief Menu navigation implementation
 */

#include "menu.h"

static menu_page_t g_current_page = PAGE_STATUS;

void menu_init(void) {
    g_current_page = PAGE_STATUS;
}

menu_page_t menu_get_page(void) {
    return g_current_page;
}

void menu_set_page(menu_page_t page) {
    if (page < PAGE_COUNT) {
        g_current_page = page;
    }
}

void menu_next_page(void) {
    g_current_page = (g_current_page + 1) % PAGE_COUNT;
}

void menu_prev_page(void) {
    if (g_current_page == 0) {
        g_current_page = PAGE_COUNT - 1;
    } else {
        g_current_page--;
    }
}

void menu_encoder_input(int8_t delta) {
    // Page-specific encoder handling
    // Currently just changes pages
    if (delta > 0) {
        menu_next_page();
    } else if (delta < 0) {
        menu_prev_page();
    }
}

void menu_button_input(uint8_t action) {
    (void)action;
    // Page-specific button handling
}
