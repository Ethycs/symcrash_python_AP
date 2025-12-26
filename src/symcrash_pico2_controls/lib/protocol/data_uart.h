/**
 * @file data_uart.h
 * @brief D-UART (Data Channel) driver
 *
 * High-throughput streaming UART for voice data.
 * Uses UART1 at 3 Mbps with credit-based flow control.
 */

#ifndef SYMCRASH_DATA_UART_H
#define SYMCRASH_DATA_UART_H

#include <stdint.h>
#include <stdbool.h>
#include "frame.h"
#include "error_codes.h"

// =============================================================================
// Types
// =============================================================================

/** D-UART state */
typedef enum {
    DATA_STATE_IDLE,                    // Not streaming
    DATA_STATE_STREAMING                // Streaming active
} data_uart_state_t;

/** Callback for received data frames */
typedef void (*data_rx_callback_t)(const frame_t *frame);

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize D-UART hardware and buffers
 */
void data_uart_init(void);

/**
 * @brief Set callback for received data frames
 *
 * @param callback  Function to call when a data frame is received
 */
void data_uart_set_rx_callback(data_rx_callback_t callback);

// =============================================================================
// Transmission
// =============================================================================

/**
 * @brief Send a data frame on D-UART (non-blocking)
 *
 * @param frame     Frame to send
 * @return ERR_OK if sent, ERR_BUFFER_FULL if no credits available
 */
error_t data_uart_send(const frame_t *frame);

/**
 * @brief Check if D-UART has credits to send
 *
 * @return Number of available credits
 */
uint16_t data_uart_get_credits(void);

// =============================================================================
// Processing
// =============================================================================

/**
 * @brief Process D-UART (call from main loop)
 *
 * Handles:
 * - RX buffer parsing
 * - Credit management
 */
void data_uart_process(void);

/**
 * @brief Start streaming
 */
void data_uart_start_streaming(void);

/**
 * @brief Stop streaming
 */
void data_uart_stop_streaming(void);

/**
 * @brief Check if streaming is active
 */
bool data_uart_is_streaming(void);

// =============================================================================
// Status
// =============================================================================

/**
 * @brief Get current D-UART state
 */
data_uart_state_t data_uart_get_state(void);

/**
 * @brief Get number of frames sent since streaming started
 */
uint32_t data_uart_get_frame_count(void);

#endif // SYMCRASH_DATA_UART_H
