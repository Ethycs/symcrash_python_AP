/**
 * @file frame.c
 * @brief WSE-STD-100 frame handling implementation
 */

#include "frame.h"
#include "crc32c.h"
#include <string.h>

// Per-channel sequence counters
static uint16_t g_seq_control = 0;
static uint16_t g_seq_data = 0;

void frame_init(frame_t *frame, channel_t chan, uint8_t msg_type) {
    memset(frame, 0, sizeof(frame_t));

    frame->header.magic = FRAME_MAGIC;
    frame->header.version = PROTOCOL_VERSION;
    frame->header.chan = (uint8_t)chan;
    frame->header.msg_type = msg_type;
    frame->header.flags = 0;
    frame->header.seq = frame_next_seq(chan);
    frame->header.length = 0;
    frame->header.crc32 = 0;

    frame->payload_len = 0;
}

bool frame_set_payload(frame_t *frame, const uint8_t *data, uint16_t len) {
    if (len > MAX_PAYLOAD_SIZE) {
        return false;
    }

    if (data != NULL && len > 0) {
        memcpy(frame->payload, data, len);
    }
    frame->payload_len = len;
    frame->header.length = len;

    return true;
}

uint16_t frame_serialize(const frame_t *frame, uint8_t *out_buf, uint16_t out_size) {
    uint16_t total_len = FRAME_HEADER_SIZE + frame->payload_len;

    if (out_size < total_len) {
        return 0;
    }

    // Copy header with CRC = 0 first
    frame_header_t header_copy = frame->header;
    header_copy.crc32 = 0;

    // Calculate CRC over header (with crc32=0) + payload
    uint32_t crc = 0xFFFFFFFF;
    crc = crc32c_update(crc, (const uint8_t *)&header_copy, FRAME_HEADER_SIZE);
    if (frame->payload_len > 0) {
        crc = crc32c_update(crc, frame->payload, frame->payload_len);
    }
    crc ^= 0xFFFFFFFF;

    // Set CRC in header
    header_copy.crc32 = crc;

    // Copy to output buffer
    memcpy(out_buf, &header_copy, FRAME_HEADER_SIZE);
    if (frame->payload_len > 0) {
        memcpy(out_buf + FRAME_HEADER_SIZE, frame->payload, frame->payload_len);
    }

    return total_len;
}

bool frame_parse(const uint8_t *data, uint16_t len, frame_t *out_frame) {
    if (len < FRAME_HEADER_SIZE) {
        return false;
    }

    // Copy header
    memcpy(&out_frame->header, data, FRAME_HEADER_SIZE);

    // Validate magic
    if (out_frame->header.magic != FRAME_MAGIC) {
        return false;
    }

    // Validate version
    if (out_frame->header.version != PROTOCOL_VERSION) {
        return false;
    }

    // Validate length
    if (out_frame->header.length > MAX_PAYLOAD_SIZE) {
        return false;
    }

    if (len < FRAME_HEADER_SIZE + out_frame->header.length) {
        return false;
    }

    // Copy payload
    out_frame->payload_len = out_frame->header.length;
    if (out_frame->payload_len > 0) {
        memcpy(out_frame->payload, data + FRAME_HEADER_SIZE, out_frame->payload_len);
    }

    // Verify CRC
    uint32_t received_crc = out_frame->header.crc32;

    // Calculate expected CRC (with crc32 field = 0)
    frame_header_t header_copy = out_frame->header;
    header_copy.crc32 = 0;

    uint32_t crc = 0xFFFFFFFF;
    crc = crc32c_update(crc, (const uint8_t *)&header_copy, FRAME_HEADER_SIZE);
    if (out_frame->payload_len > 0) {
        crc = crc32c_update(crc, out_frame->payload, out_frame->payload_len);
    }
    crc ^= 0xFFFFFFFF;

    if (crc != received_crc) {
        return false;
    }

    return true;
}

uint16_t frame_next_seq(channel_t chan) {
    if (chan == CHANNEL_CONTROL) {
        return g_seq_control++;
    } else {
        return g_seq_data++;
    }
}

uint16_t frame_get_seq(channel_t chan) {
    if (chan == CHANNEL_CONTROL) {
        return g_seq_control;
    } else {
        return g_seq_data;
    }
}
