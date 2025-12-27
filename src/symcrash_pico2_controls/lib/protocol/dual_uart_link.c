/**
 * @file dual_uart_link.c
 * @brief Dual-UART Link coordinator implementation
 */

#include "dual_uart_link.h"
#include "ctrl_uart.h"
#include "data_uart.h"
#include "config.h"

#include "pico/stdlib.h"

// =============================================================================
// State
// =============================================================================

static link_state_t g_state = LINK_STATE_DISCONNECTED;
static link_stats_t g_stats = {0};

// Block timing
static block_id_t g_base_block_id = 0;
static uint64_t g_base_time_us = 0;

// =============================================================================
// Initialization
// =============================================================================

void dual_uart_init(void) {
    ctrl_uart_init();
    data_uart_init();

    g_state = LINK_STATE_DISCONNECTED;
    dual_uart_reset_stats();
}

error_t dual_uart_connect(void) {
    g_state = LINK_STATE_CONNECTING;

    error_t err = ctrl_uart_connect();
    if (err != ERR_OK) {
        g_state = LINK_STATE_DISCONNECTED;
        return err;
    }

    g_state = LINK_STATE_CONNECTED;

    // Initialize block timing
    g_base_block_id = 0;
    g_base_time_us = time_us_64();

    // Start data streaming
    data_uart_start_streaming();

    return ERR_OK;
}

void dual_uart_disconnect(void) {
    data_uart_stop_streaming();
    g_state = LINK_STATE_DISCONNECTED;
}

// =============================================================================
// Message Sending
// =============================================================================

error_t dual_uart_send_note(midi_note_t note, velocity_t velocity, bool on) {
    if (g_state != LINK_STATE_CONNECTED) {
        return ERR_NOT_CONNECTED;
    }

    frame_t frame;
    msg_build_note_event(&frame, note, velocity, on, 0, NOTE_FLAG_NONE,
                         dual_uart_get_apply_block());

    error_t err = ctrl_uart_send_async(&frame);
    if (err == ERR_OK) {
        g_stats.tx_frames_ctrl++;
    }

    return err;
}

error_t dual_uart_send_macro(macro_id_t macro_id, q15_t value) {
    // Allow sending even when not fully connected (for testing/startup)
    // The transport layer will handle the actual send

    frame_t frame;
    msg_build_set_macro(&frame, macro_id, value, dual_uart_get_apply_block());

    error_t err = ctrl_uart_send_async(&frame);
    if (err == ERR_OK) {
        g_stats.tx_frames_ctrl++;
    }

    return err;
}

error_t dual_uart_send_tree_target(tree_index_t tree_index, uint16_t tau_ms) {
    if (g_state != LINK_STATE_CONNECTED) {
        return ERR_NOT_CONNECTED;
    }

    frame_t frame;
    msg_build_set_tree_target(&frame, tree_index, tau_ms, TREE_CURVE_EXPONENTIAL,
                              dual_uart_get_apply_block());

    error_t err = ctrl_uart_send_async(&frame);
    if (err == ERR_OK) {
        g_stats.tx_frames_ctrl++;
    }

    return err;
}

// =============================================================================
// Processing
// =============================================================================

void dual_uart_process(void) {
    ctrl_uart_process();
    data_uart_process();

    // Update stats
    g_stats.retransmits = ctrl_uart_get_retransmit_count();
}

// =============================================================================
// Block Timing
// =============================================================================

block_id_t dual_uart_get_current_block(void) {
    uint64_t now_us = time_us_64();
    uint64_t elapsed_us = now_us - g_base_time_us;

    // Blocks since base time
    uint64_t blocks = elapsed_us / BLOCK_DURATION_US;

    return g_base_block_id + (block_id_t)blocks;
}

block_id_t dual_uart_get_apply_block(void) {
    return dual_uart_get_current_block() + LEAD_BLOCKS;
}

void dual_uart_sync_block(block_id_t engine_block_id) {
    g_base_block_id = engine_block_id;
    g_base_time_us = time_us_64();
}

// =============================================================================
// Status
// =============================================================================

link_state_t dual_uart_get_state(void) {
    return g_state;
}

bool dual_uart_is_connected(void) {
    // In USB CDC mode, check if USB host has opened the CDC port
    // This is more useful than protocol state for display purposes
#if TRANSPORT_USB_MODE
    return ctrl_uart_is_connected();
#else
    return g_state == LINK_STATE_CONNECTED;
#endif
}

void dual_uart_get_stats(link_stats_t *stats) {
    if (stats != NULL) {
        *stats = g_stats;
    }
}

void dual_uart_reset_stats(void) {
    g_stats.tx_frames_ctrl = 0;
    g_stats.tx_frames_data = 0;
    g_stats.rx_frames = 0;
    g_stats.retransmits = 0;
    g_stats.crc_errors = 0;
}
