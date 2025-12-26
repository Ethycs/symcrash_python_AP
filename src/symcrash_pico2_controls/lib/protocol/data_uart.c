/**
 * @file data_uart.c
 * @brief D-UART (Data Channel) driver implementation
 */

#include "data_uart.h"
#include "cobs.h"
#include "config.h"

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

static data_uart_state_t g_state = DATA_STATE_IDLE;
static data_rx_callback_t g_rx_callback = NULL;
static uint16_t g_credits = 16;  // Initial credit pool
static uint32_t g_frame_count = 0;

// Frame assembly buffer
static uint8_t g_frame_buf[MAX_PAYLOAD_SIZE + FRAME_HEADER_SIZE + 16];
static uint16_t g_frame_len = 0;

// =============================================================================
// UART IRQ Handler
// =============================================================================

static void data_uart_rx_handler(void) {
    while (uart_is_readable(D_UART_ID)) {
        uint8_t byte = uart_getc(D_UART_ID);
        ring_push(&g_rx_buffer, byte);
    }
}

// =============================================================================
// Initialization
// =============================================================================

void data_uart_init(void) {
    // Initialize ring buffer
    memset(&g_rx_buffer, 0, sizeof(g_rx_buffer));

    // Initialize UART
    uart_init(D_UART_ID, D_UART_BAUD);

    // Set pin functions
    gpio_set_function(D_UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(D_UART_RX_PIN, GPIO_FUNC_UART);

    // Configure UART
    uart_set_hw_flow(D_UART_ID, false, false);
    uart_set_format(D_UART_ID, 8, 1, UART_PARITY_NONE);
    uart_set_fifo_enabled(D_UART_ID, true);

    // Enable RX interrupt
    int irq = D_UART_ID == uart0 ? UART0_IRQ : UART1_IRQ;
    irq_set_exclusive_handler(irq, data_uart_rx_handler);
    irq_set_enabled(irq, true);
    uart_set_irq_enables(D_UART_ID, true, false);

    g_state = DATA_STATE_IDLE;
}

void data_uart_set_rx_callback(data_rx_callback_t callback) {
    g_rx_callback = callback;
}

// =============================================================================
// Transmission
// =============================================================================

error_t data_uart_send(const frame_t *frame) {
    if (g_state != DATA_STATE_STREAMING) {
        return ERR_NOT_CONNECTED;
    }

    if (g_credits == 0) {
        return ERR_BUFFER_FULL;
    }

    // Serialize frame
    uint8_t raw_buf[MAX_PAYLOAD_SIZE + FRAME_HEADER_SIZE];
    uint16_t raw_len = frame_serialize(frame, raw_buf, sizeof(raw_buf));

    if (raw_len == 0) {
        return ERR_INVALID_PARAM;
    }

    // COBS encode
    uint8_t cobs_buf[MAX_PAYLOAD_SIZE + FRAME_HEADER_SIZE + 16];
    size_t cobs_len = cobs_encode(raw_buf, raw_len, cobs_buf);

    // Send COBS-encoded data + delimiter
    uart_write_blocking(D_UART_ID, cobs_buf, cobs_len);
    uart_putc_raw(D_UART_ID, COBS_DELIMITER);

    g_credits--;
    g_frame_count++;

    return ERR_OK;
}

uint16_t data_uart_get_credits(void) {
    return g_credits;
}

// =============================================================================
// Processing
// =============================================================================

static void process_received_frame(const uint8_t *data, uint16_t len) {
    frame_t frame;

    if (!frame_parse(data, len, &frame)) {
        return;  // Invalid frame
    }

    // Handle credit messages
    if (frame.header.msg_type == DATA_MSG_DATA_CREDIT) {
        // Extract credit count from payload
        if (frame.payload_len >= 2) {
            uint16_t new_credits = frame.payload[0] | (frame.payload[1] << 8);
            g_credits += new_credits;
        }
        return;
    }

    // Call user callback for other data messages
    if (g_rx_callback != NULL) {
        g_rx_callback(&frame);
    }
}

void data_uart_process(void) {
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
}

void data_uart_start_streaming(void) {
    g_state = DATA_STATE_STREAMING;
    g_frame_count = 0;
}

void data_uart_stop_streaming(void) {
    g_state = DATA_STATE_IDLE;
}

bool data_uart_is_streaming(void) {
    return g_state == DATA_STATE_STREAMING;
}

// =============================================================================
// Status
// =============================================================================

data_uart_state_t data_uart_get_state(void) {
    return g_state;
}

uint32_t data_uart_get_frame_count(void) {
    return g_frame_count;
}
