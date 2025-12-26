/**
 * @file dual_uart_link.h
 * @brief Dual-UART Link coordinator
 *
 * Manages both C-UART (control) and D-UART (data) channels.
 * Provides unified interface for sending messages.
 */

#ifndef SYMCRASH_DUAL_UART_LINK_H
#define SYMCRASH_DUAL_UART_LINK_H

#include <stdint.h>
#include <stdbool.h>
#include "types.h"
#include "frame.h"
#include "messages.h"
#include "error_codes.h"

// =============================================================================
// Types
// =============================================================================

/** Link state */
typedef enum {
    LINK_STATE_DISCONNECTED,            // Not connected
    LINK_STATE_CONNECTING,              // Handshake in progress
    LINK_STATE_CONNECTED                // Ready for communication
} link_state_t;

/** Link statistics */
typedef struct {
    uint32_t tx_frames_ctrl;            // Frames sent on C-UART
    uint32_t tx_frames_data;            // Frames sent on D-UART
    uint32_t rx_frames;                 // Total frames received
    uint32_t retransmits;               // Retransmit count
    uint32_t crc_errors;                // CRC validation failures
} link_stats_t;

// =============================================================================
// Initialization
// =============================================================================

/**
 * @brief Initialize dual-UART link (both channels)
 */
void dual_uart_init(void);

/**
 * @brief Connect to synthesis engine
 *
 * Performs HELLO/HELLO_ACK handshake.
 *
 * @return ERR_OK on success, ERR_TIMEOUT on failure
 */
error_t dual_uart_connect(void);

/**
 * @brief Disconnect from synthesis engine
 */
void dual_uart_disconnect(void);

// =============================================================================
// Message Sending (High-Level API)
// =============================================================================

/**
 * @brief Send a NoteEvent message
 *
 * @param note      MIDI note number
 * @param velocity  Note velocity
 * @param on        true for note on, false for note off
 * @return ERR_OK on success
 */
error_t dual_uart_send_note(midi_note_t note, velocity_t velocity, bool on);

/**
 * @brief Send a SetMacro message
 *
 * @param macro_id  Macro knob ID (0-7)
 * @param value     Q15 value (0-32767)
 * @return ERR_OK on success
 */
error_t dual_uart_send_macro(macro_id_t macro_id, q15_t value);

/**
 * @brief Send a SetTreeTarget message
 *
 * @param tree_index    Target tree slot (0-4)
 * @param tau_ms        Morph time constant
 * @return ERR_OK on success
 */
error_t dual_uart_send_tree_target(tree_index_t tree_index, uint16_t tau_ms);

// =============================================================================
// Processing
// =============================================================================

/**
 * @brief Process both UART channels (call from main loop)
 *
 * Must be called frequently to handle RX/TX and timeouts.
 */
void dual_uart_process(void);

// =============================================================================
// Block Timing
// =============================================================================

/**
 * @brief Get current block ID estimate
 *
 * Based on elapsed time since connection.
 *
 * @return Estimated current block ID
 */
block_id_t dual_uart_get_current_block(void);

/**
 * @brief Get apply block ID for new messages
 *
 * Returns current block + LEAD_BLOCKS for proper scheduling.
 *
 * @return Block ID to use in apply_block_id fields
 */
block_id_t dual_uart_get_apply_block(void);

/**
 * @brief Synchronize block counter with engine
 *
 * @param engine_block_id   Block ID received from engine
 */
void dual_uart_sync_block(block_id_t engine_block_id);

// =============================================================================
// Status
// =============================================================================

/**
 * @brief Get current link state
 */
link_state_t dual_uart_get_state(void);

/**
 * @brief Check if link is connected
 */
bool dual_uart_is_connected(void);

/**
 * @brief Get link statistics
 */
void dual_uart_get_stats(link_stats_t *stats);

/**
 * @brief Reset link statistics
 */
void dual_uart_reset_stats(void);

#endif // SYMCRASH_DUAL_UART_LINK_H
