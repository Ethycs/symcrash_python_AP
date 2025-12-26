/**
 * @file messages.c
 * @brief WSE-STD-100 message builders implementation
 */

#include "messages.h"
#include <string.h>

void msg_build_note_event(frame_t *frame,
                          midi_note_t note,
                          velocity_t velocity,
                          bool on,
                          midi_channel_t channel,
                          uint8_t flags,
                          block_id_t apply_block) {
    frame_init(frame, CHANNEL_CONTROL, CTRL_MSG_NOTE_EVENT);

    note_event_t evt = {
        .apply_block_id = apply_block,
        .note = note,
        .vel = velocity,
        .on = on ? 1 : 0,
        .chan = channel,
        .flags = flags
    };

    frame_set_payload(frame, (const uint8_t *)&evt, NOTE_EVENT_SIZE);
}

void msg_build_set_macro(frame_t *frame,
                         macro_id_t macro_id,
                         q15_t value,
                         block_id_t apply_block) {
    frame_init(frame, CHANNEL_CONTROL, CTRL_MSG_SET_MACRO);

    set_macro_t macro = {
        .apply_block_id = apply_block,
        .macro_id = macro_id,
        .value_q15 = (uint16_t)value,
        .reserved = 0
    };

    frame_set_payload(frame, (const uint8_t *)&macro, SET_MACRO_SIZE);
}

void msg_build_set_tree_target(frame_t *frame,
                               tree_index_t tree_index,
                               uint16_t tau_ms,
                               tree_morph_curve_t curve,
                               block_id_t apply_block) {
    frame_init(frame, CHANNEL_CONTROL, CTRL_MSG_SET_TREE_TARGET);

    set_tree_target_t tree = {
        .apply_block_id = apply_block,
        .tree_index = tree_index,
        .tau_ms = tau_ms,
        .curve = (uint8_t)curve
    };

    frame_set_payload(frame, (const uint8_t *)&tree, SET_TREE_TARGET_SIZE);
}

void msg_build_hello(frame_t *frame, uint16_t build_id) {
    frame_init(frame, CHANNEL_CONTROL, CTRL_MSG_HELLO);

    hello_t hello = {
        .build_id = build_id,
        .protocol_version = (PROTOCOL_VERSION << 8) | 0x00
    };

    frame_set_payload(frame, (const uint8_t *)&hello, HELLO_SIZE);
}

void msg_build_stream_ctrl(frame_t *frame, bool start, block_id_t block_id) {
    frame_init(frame, CHANNEL_CONTROL, CTRL_MSG_STREAM_CTRL);

    stream_ctrl_t ctrl = {
        .start_block_id = block_id
    };

    // Use different flags for start vs stop
    frame->header.flags = start ? 0x01 : 0x00;

    frame_set_payload(frame, (const uint8_t *)&ctrl, STREAM_CTRL_SIZE);
}

void msg_build_ack(frame_t *frame, uint16_t acked_seq) {
    frame_init(frame, CHANNEL_CONTROL, CTRL_MSG_ACK);

    ack_t ack = {
        .acked_seq = acked_seq
    };

    frame_set_payload(frame, (const uint8_t *)&ack, ACK_SIZE);
}

void msg_build_nack(frame_t *frame, uint16_t nacked_seq) {
    frame_init(frame, CHANNEL_CONTROL, CTRL_MSG_NACK);

    ack_t nack = {
        .acked_seq = nacked_seq
    };

    frame_set_payload(frame, (const uint8_t *)&nack, ACK_SIZE);
}
