/**
 * @file frame.h
 * @brief WSE-STD-100 frame header handling
 *
 * Frame structure for Dual-UART Link protocol.
 */

#ifndef SYMCRASH_FRAME_H
#define SYMCRASH_FRAME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "config.h"

// =============================================================================
// Constants
// =============================================================================

#define FRAME_HEADER_SIZE   14      // Size of frame header in bytes

// Channel identifiers
typedef enum {
    CHANNEL_CONTROL = 0,            // C-UART: reliable control channel
    CHANNEL_DATA = 1                // D-UART: streaming data channel
} channel_t;

// Control message types (WSE-STD-100 Section 13.1)
typedef enum {
    CTRL_MSG_HELLO          = 0x01,
    CTRL_MSG_HELLO_ACK      = 0x02,
    CTRL_MSG_GET_STATUS     = 0x03,
    CTRL_MSG_STATUS         = 0x04,
    CTRL_MSG_SET_INSTRUMENT = 0x10,
    CTRL_MSG_SET_TREE_TARGET= 0x11,
    CTRL_MSG_SET_MACRO      = 0x12,
    CTRL_MSG_NOTE_EVENT     = 0x20,
    CTRL_MSG_VOICE_POLICY   = 0x21,
    CTRL_MSG_STREAM_CTRL    = 0x30,
    CTRL_MSG_ACK            = 0xF0,
    CTRL_MSG_NACK           = 0xF1
} ctrl_msg_type_t;

// Data message types
typedef enum {
    DATA_MSG_VOICE_DELTA    = 0x01,
    DATA_MSG_VOICE_BURST    = 0x02,
    DATA_MSG_DATA_CREDIT    = 0x03,
    DATA_MSG_STREAM_START   = 0x10,
    DATA_MSG_STREAM_STOP    = 0x11
} data_msg_type_t;

// =============================================================================
// Frame Header Structure
// =============================================================================

/**
 * @brief Frame header (14 bytes, little-endian on wire)
 */
typedef struct __attribute__((packed)) {
    uint16_t magic;         // 0xDEC0
    uint8_t  version;       // 0x01
    uint8_t  chan;          // 0=control, 1=data
    uint8_t  msg_type;      // Message type (enumerated)
    uint8_t  flags;         // Bitfield (reserved)
    uint16_t seq;           // Per-channel sequence number
    uint16_t length;        // Payload length in bytes
    uint32_t crc32;         // CRC-32C of header (with crc32=0) + payload
} frame_header_t;

/**
 * @brief Complete frame buffer
 */
typedef struct {
    frame_header_t header;
    uint8_t payload[MAX_PAYLOAD_SIZE];
    uint16_t payload_len;
} frame_t;

// =============================================================================
// Frame Building Functions
// =============================================================================

/**
 * @brief Initialize a frame with header defaults
 *
 * @param frame     Frame to initialize
 * @param chan      Channel (CHANNEL_CONTROL or CHANNEL_DATA)
 * @param msg_type  Message type
 */
void frame_init(frame_t *frame, channel_t chan, uint8_t msg_type);

/**
 * @brief Set frame payload
 *
 * @param frame     Frame to modify
 * @param data      Payload data
 * @param len       Payload length
 * @return true on success, false if payload too large
 */
bool frame_set_payload(frame_t *frame, const uint8_t *data, uint16_t len);

/**
 * @brief Serialize frame to wire format (with CRC, without COBS)
 *
 * @param frame     Frame to serialize
 * @param out_buf   Output buffer
 * @param out_size  Output buffer size
 * @return Number of bytes written, or 0 on error
 */
uint16_t frame_serialize(const frame_t *frame, uint8_t *out_buf, uint16_t out_size);

/**
 * @brief Parse frame from wire format
 *
 * @param data      Input buffer (header + payload, no COBS)
 * @param len       Input length
 * @param out_frame Parsed frame
 * @return true on success (valid magic, version, CRC), false otherwise
 */
bool frame_parse(const uint8_t *data, uint16_t len, frame_t *out_frame);

/**
 * @brief Update sequence number for channel
 *
 * @param chan      Channel
 * @return Next sequence number for that channel
 */
uint16_t frame_next_seq(channel_t chan);

/**
 * @brief Get current sequence number for channel
 *
 * @param chan      Channel
 * @return Current sequence number
 */
uint16_t frame_get_seq(channel_t chan);

#endif // SYMCRASH_FRAME_H
