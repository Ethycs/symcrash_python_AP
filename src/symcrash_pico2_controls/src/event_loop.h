/**
 * @file event_loop.h
 * @brief Main event loop
 */

#ifndef SYMCRASH_EVENT_LOOP_H
#define SYMCRASH_EVENT_LOOP_H

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Initialize event loop timers and callbacks
 */
void event_loop_init(void);

/**
 * @brief Run the main event loop (never returns)
 */
void event_loop_run(void);

/**
 * @brief Request a display update
 */
void event_loop_request_display_update(void);

#endif // SYMCRASH_EVENT_LOOP_H
