/**
 * @file ctrl_uart.h
 * @brief C-UART (Control Channel) driver
 *
 * Reliable, acknowledged UART communication for control messages.
 * Uses UART0 at 1 Mbps with ACK/retransmit logic.
 */

#ifndef SYMCRASH_CTRL_UART_H
#define SYMCRASH_CTRL_UART_H

#include <stdint.h>
#include <stdbool.h>
#include "frame.h"
#include "error_codes.h"

// =============================================================================
// Configuration
// =============================================================================

#define CTRL_ACK_TIMEOUT_MS     30      // ACK timeout in ms
#define CTRL_MAX_RETRIES        3       // Maximum retransmit attempts

// =============================================================================
// Types
// =============================================================================

/** C-UART state */
typedef enum {
    CTRL_STATE_IDLE,                    // Ready to send
    CTRL_STATE_WAIT_ACK,                // Waiting for ACK
    CTRL_STATE_CONNECTED                // Handshake complete
} ctrl_uart_state_t;

/** Callback for received frames */
typedef void (*ctrl_rx_callback_t)(const frame_t *frame);

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize C-UART hardware and buffers
 */
void ctrl_uart_init(void);

/**
 * @brief Set callback for received frames
 *
 * @param callback  Function to call when a frame is received
 */
void ctrl_uart_set_rx_callback(ctrl_rx_callback_t callback);

// =============================================================================
// Transmission
// =============================================================================

/**
 * @brief Send a frame on C-UART (blocking until ACK or timeout)
 *
 * @param frame     Frame to send
 * @return ERR_OK on success, ERR_TIMEOUT if no ACK, ERR_BUFFER_FULL if TX full
 */
error_t ctrl_uart_send(const frame_t *frame);

/**
 * @brief Send a frame on C-UART (non-blocking, queues for transmission)
 *
 * @param frame     Frame to send
 * @return ERR_OK if queued, ERR_BUFFER_FULL if queue full
 */
error_t ctrl_uart_send_async(const frame_t *frame);

/**
 * @brief Send raw bytes on C-UART (for debugging)
 *
 * @param data      Data to send
 * @param len       Length of data
 */
void ctrl_uart_send_raw(const uint8_t *data, size_t len);

// =============================================================================
// Processing
// =============================================================================

/**
 * @brief Process C-UART (call from main loop)
 *
 * Handles:
 * - RX buffer parsing
 * - ACK/NACK processing
 * - Retransmission timeouts
 */
void ctrl_uart_process(void);

/**
 * @brief Check if C-UART is connected (handshake complete)
 *
 * @return true if connected
 */
bool ctrl_uart_is_connected(void);

/**
 * @brief Initiate connection handshake
 *
 * Sends HELLO and waits for HELLO_ACK.
 *
 * @return ERR_OK on success, ERR_TIMEOUT on failure
 */
error_t ctrl_uart_connect(void);

// =============================================================================
// Status
// =============================================================================

/**
 * @brief Get current C-UART state
 */
ctrl_uart_state_t ctrl_uart_get_state(void);

/**
 * @brief Get number of frames pending in TX queue
 */
uint8_t ctrl_uart_tx_pending(void);

/**
 * @brief Get number of retransmits since last reset
 */
uint32_t ctrl_uart_get_retransmit_count(void);

#endif // SYMCRASH_CTRL_UART_H
