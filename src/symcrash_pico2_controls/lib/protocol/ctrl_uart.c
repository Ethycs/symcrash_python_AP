/**
 * @file ctrl_uart.c
 * @brief C-UART (Control Channel) driver implementation
 */

#include "ctrl_uart.h"
#include "cobs.h"
#include "crc32c.h"
#include "config.h"
#include "messages.h"

#include "pico/stdlib.h"
#include "hardware/uart.h"
#include "hardware/irq.h"
#include <string.h>

// =============================================================================
// Ring Buffer
// =============================================================================

typedef struct {
    uint8_t data[RX_BUFFER_SIZE];
    volatile uint16_t head;
    volatile uint16_t tail;
} ring_buffer_t;

static ring_buffer_t g_rx_buffer;
static ring_buffer_t g_tx_buffer;

static inline bool ring_is_empty(const ring_buffer_t *rb) {
    return rb->head == rb->tail;
}

static inline bool ring_is_full(const ring_buffer_t *rb) {
    return ((rb->head + 1) % RX_BUFFER_SIZE) == rb->tail;
}

static inline void ring_push(ring_buffer_t *rb, uint8_t byte) {
    if (!ring_is_full(rb)) {
        rb->data[rb->head] = byte;
        rb->head = (rb->head + 1) % RX_BUFFER_SIZE;
    }
}

static inline bool ring_pop(ring_buffer_t *rb, uint8_t *byte) {
    if (ring_is_empty(rb)) {
        return false;
    }
    *byte = rb->data[rb->tail];
    rb->tail = (rb->tail + 1) % RX_BUFFER_SIZE;
    return true;
}

// =============================================================================
// State
// =============================================================================

static ctrl_uart_state_t g_state = CTRL_STATE_IDLE;
static ctrl_rx_callback_t g_rx_callback = NULL;
static uint32_t g_retransmit_count = 0;

// Frame assembly buffer
static uint8_t g_frame_buf[MAX_PAYLOAD_SIZE + FRAME_HEADER_SIZE + 16];
static uint16_t g_frame_len = 0;

// Pending TX frame for retransmission
static frame_t g_pending_frame;
static bool g_has_pending = false;
static uint32_t g_send_time_ms = 0;
static uint8_t g_retry_count = 0;

// =============================================================================
// UART IRQ Handler
// =============================================================================

static void ctrl_uart_rx_handler(void) {
    while (uart_is_readable(C_UART_ID)) {
        uint8_t byte = uart_getc(C_UART_ID);
        ring_push(&g_rx_buffer, byte);
    }
}

// =============================================================================
// Initialization
// =============================================================================

void ctrl_uart_init(void) {
    // Initialize ring buffers
    memset(&g_rx_buffer, 0, sizeof(g_rx_buffer));
    memset(&g_tx_buffer, 0, sizeof(g_tx_buffer));

    // Initialize UART
    uart_init(C_UART_ID, C_UART_BAUD);

    // Set pin functions
    gpio_set_function(C_UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(C_UART_RX_PIN, GPIO_FUNC_UART);

    // Configure UART
    uart_set_hw_flow(C_UART_ID, false, false);
    uart_set_format(C_UART_ID, 8, 1, UART_PARITY_NONE);
    uart_set_fifo_enabled(C_UART_ID, true);

    // Enable RX interrupt
    int irq = C_UART_ID == uart0 ? UART0_IRQ : UART1_IRQ;
    irq_set_exclusive_handler(irq, ctrl_uart_rx_handler);
    irq_set_enabled(irq, true);
    uart_set_irq_enables(C_UART_ID, true, false);

    g_state = CTRL_STATE_IDLE;
}

void ctrl_uart_set_rx_callback(ctrl_rx_callback_t callback) {
    g_rx_callback = callback;
}

// =============================================================================
// Transmission
// =============================================================================

static void send_frame_internal(const frame_t *frame) {
    // Serialize frame
    uint8_t raw_buf[MAX_PAYLOAD_SIZE + FRAME_HEADER_SIZE];
    uint16_t raw_len = frame_serialize(frame, raw_buf, sizeof(raw_buf));

    if (raw_len == 0) {
        return;
    }

    // COBS encode
    uint8_t cobs_buf[MAX_PAYLOAD_SIZE + FRAME_HEADER_SIZE + 16];
    size_t cobs_len = cobs_encode(raw_buf, raw_len, cobs_buf);

    // Send COBS-encoded data + delimiter
    uart_write_blocking(C_UART_ID, cobs_buf, cobs_len);
    uart_putc_raw(C_UART_ID, COBS_DELIMITER);
}

error_t ctrl_uart_send(const frame_t *frame) {
    // Store for potential retransmission
    memcpy(&g_pending_frame, frame, sizeof(frame_t));
    g_has_pending = true;
    g_retry_count = 0;
    g_send_time_ms = to_ms_since_boot(get_absolute_time());

    send_frame_internal(frame);
    g_state = CTRL_STATE_WAIT_ACK;

    // Wait for ACK with timeout
    uint32_t start = to_ms_since_boot(get_absolute_time());
    while (g_state == CTRL_STATE_WAIT_ACK) {
        ctrl_uart_process();

        uint32_t elapsed = to_ms_since_boot(get_absolute_time()) - start;
        if (elapsed > CTRL_ACK_TIMEOUT_MS) {
            // Retry
            if (g_retry_count < CTRL_MAX_RETRIES) {
                g_retry_count++;
                g_retransmit_count++;
                send_frame_internal(&g_pending_frame);
                start = to_ms_since_boot(get_absolute_time());
            } else {
                g_state = CTRL_STATE_IDLE;
                g_has_pending = false;
                return ERR_TIMEOUT;
            }
        }

        tight_loop_contents();
    }

    g_has_pending = false;
    return ERR_OK;
}

error_t ctrl_uart_send_async(const frame_t *frame) {
    send_frame_internal(frame);
    return ERR_OK;
}

void ctrl_uart_send_raw(const uint8_t *data, size_t len) {
    uart_write_blocking(C_UART_ID, data, len);
}

// =============================================================================
// Processing
// =============================================================================

static void process_received_frame(const uint8_t *data, uint16_t len) {
    frame_t frame;

    if (!frame_parse(data, len, &frame)) {
        return;  // Invalid frame
    }

    // Handle ACK/NACK
    if (frame.header.msg_type == CTRL_MSG_ACK) {
        if (g_state == CTRL_STATE_WAIT_ACK) {
            g_state = CTRL_STATE_CONNECTED;
            g_has_pending = false;
        }
        return;
    }

    if (frame.header.msg_type == CTRL_MSG_NACK) {
        // Retransmit if we have a pending frame
        if (g_has_pending && g_retry_count < CTRL_MAX_RETRIES) {
            g_retry_count++;
            g_retransmit_count++;
            send_frame_internal(&g_pending_frame);
        }
        return;
    }

    // Handle HELLO_ACK
    if (frame.header.msg_type == CTRL_MSG_HELLO_ACK) {
        g_state = CTRL_STATE_CONNECTED;
        g_has_pending = false;
    }

    // Call user callback
    if (g_rx_callback != NULL) {
        g_rx_callback(&frame);
    }

    // Send ACK for received frame
    frame_t ack_frame;
    msg_build_ack(&ack_frame, frame.header.seq);
    send_frame_internal(&ack_frame);
}

void ctrl_uart_process(void) {
    uint8_t byte;

    // Process received bytes
    while (ring_pop(&g_rx_buffer, &byte)) {
        if (byte == COBS_DELIMITER) {
            // End of frame - decode and process
            if (g_frame_len > 0) {
                uint8_t decoded[MAX_PAYLOAD_SIZE + FRAME_HEADER_SIZE];
                size_t decoded_len = cobs_decode(g_frame_buf, g_frame_len, decoded);

                if (decoded_len > 0) {
                    process_received_frame(decoded, decoded_len);
                }
            }
            g_frame_len = 0;
        } else {
            // Accumulate frame bytes
            if (g_frame_len < sizeof(g_frame_buf)) {
                g_frame_buf[g_frame_len++] = byte;
            }
        }
    }

    // Check for retransmission timeout
    if (g_has_pending && g_state == CTRL_STATE_WAIT_ACK) {
        uint32_t elapsed = to_ms_since_boot(get_absolute_time()) - g_send_time_ms;
        if (elapsed > CTRL_ACK_TIMEOUT_MS) {
            if (g_retry_count < CTRL_MAX_RETRIES) {
                g_retry_count++;
                g_retransmit_count++;
                send_frame_internal(&g_pending_frame);
                g_send_time_ms = to_ms_since_boot(get_absolute_time());
            } else {
                g_state = CTRL_STATE_IDLE;
                g_has_pending = false;
            }
        }
    }
}

bool ctrl_uart_is_connected(void) {
    return g_state == CTRL_STATE_CONNECTED;
}

error_t ctrl_uart_connect(void) {
    frame_t hello;
    msg_build_hello(&hello, 0x0001);  // Build ID 1

    return ctrl_uart_send(&hello);
}

// =============================================================================
// Status
// =============================================================================

ctrl_uart_state_t ctrl_uart_get_state(void) {
    return g_state;
}

uint8_t ctrl_uart_tx_pending(void) {
    return g_has_pending ? 1 : 0;
}

uint32_t ctrl_uart_get_retransmit_count(void) {
    return g_retransmit_count;
}
