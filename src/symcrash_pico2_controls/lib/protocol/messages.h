/**
 * @file messages.h
 * @brief WSE-STD-100 message structures and builders
 *
 * Control message payloads for NoteEvent, SetMacro, SetTreeTarget, etc.
 */

#ifndef SYMCRASH_MESSAGES_H
#define SYMCRASH_MESSAGES_H

#include <stdint.h>
#include <stdbool.h>
#include "types.h"
#include "frame.h"

// =============================================================================
// NoteEvent (9 bytes) - WSE-STD-100 Section 5.3.1
// =============================================================================

/** Note event flags */
typedef enum {
    NOTE_FLAG_NONE      = 0x00,
    NOTE_FLAG_LEGATO    = 0x01,     // Legato note (no attack)
    NOTE_FLAG_RETRIG    = 0x02      // Retrigger envelope
} note_event_flags_t;

/** Note event payload */
typedef struct __attribute__((packed)) {
    uint32_t apply_block_id;        // Block ID when event takes effect
    uint8_t  note;                  // MIDI note 0-127
    uint8_t  vel;                   // Velocity 0-127
    uint8_t  on;                    // 1=note on, 0=note off
    uint8_t  chan;                  // MIDI channel 0-15
    uint8_t  flags;                 // NoteEventFlags
} note_event_t;

#define NOTE_EVENT_SIZE     9

// =============================================================================
// SetMacro (8 bytes) - WSE-STD-100 Section 5.2
// =============================================================================

/** SetMacro payload */
typedef struct __attribute__((packed)) {
    uint32_t apply_block_id;        // Block ID when change takes effect
    uint8_t  macro_id;              // Macro index 0-7
    uint16_t value_q15;             // Q15 value 0-32767
    uint8_t  reserved;              // Reserved for future use
} set_macro_t;

#define SET_MACRO_SIZE      8

// =============================================================================
// SetTreeTarget (8 bytes) - WSE-STD-100 Section 5.3.2
// =============================================================================

/** Tree morph curves */
typedef enum {
    TREE_CURVE_EXPONENTIAL = 0,     // Exponential decay
    TREE_CURVE_MIN_JERK = 1         // Minimum jerk trajectory
} tree_morph_curve_t;

/** SetTreeTarget payload */
typedef struct __attribute__((packed)) {
    uint32_t apply_block_id;        // Block ID when morph starts
    uint8_t  tree_index;            // Target tree slot 0-4
    uint16_t tau_ms;                // Morph time constant in ms
    uint8_t  curve;                 // TreeMorphCurve
} set_tree_target_t;

#define SET_TREE_TARGET_SIZE    8

// =============================================================================
// Hello (4 bytes)
// =============================================================================

typedef struct __attribute__((packed)) {
    uint16_t build_id;              // Controller build ID
    uint16_t protocol_version;      // Protocol version (0x0100)
} hello_t;

#define HELLO_SIZE          4

// =============================================================================
// StreamCtrl (4 bytes)
// =============================================================================

typedef struct __attribute__((packed)) {
    uint32_t start_block_id;        // Block ID to start/stop streaming
} stream_ctrl_t;

#define STREAM_CTRL_SIZE    4

// =============================================================================
// ACK/NACK (2 bytes)
// =============================================================================

typedef struct __attribute__((packed)) {
    uint16_t acked_seq;             // Sequence number being acknowledged
} ack_t;

#define ACK_SIZE            2

// =============================================================================
// Message Building Functions
// =============================================================================

/**
 * @brief Build a NoteEvent frame
 *
 * @param frame         Frame to populate
 * @param note          MIDI note number (0-127)
 * @param velocity      Note velocity (0-127)
 * @param on            true for note on, false for note off
 * @param channel       MIDI channel (0-15)
 * @param flags         Note event flags
 * @param apply_block   Block ID when event takes effect
 */
void msg_build_note_event(frame_t *frame,
                          midi_note_t note,
                          velocity_t velocity,
                          bool on,
                          midi_channel_t channel,
                          uint8_t flags,
                          block_id_t apply_block);

/**
 * @brief Build a SetMacro frame
 *
 * @param frame         Frame to populate
 * @param macro_id      Macro knob ID (0-7)
 * @param value         Q15 value (0-32767)
 * @param apply_block   Block ID when change takes effect
 */
void msg_build_set_macro(frame_t *frame,
                         macro_id_t macro_id,
                         q15_t value,
                         block_id_t apply_block);

/**
 * @brief Build a SetTreeTarget frame
 *
 * @param frame         Frame to populate
 * @param tree_index    Target tree slot (0-4)
 * @param tau_ms        Morph time constant in milliseconds
 * @param curve         Morph curve type
 * @param apply_block   Block ID when morph starts
 */
void msg_build_set_tree_target(frame_t *frame,
                               tree_index_t tree_index,
                               uint16_t tau_ms,
                               tree_morph_curve_t curve,
                               block_id_t apply_block);

/**
 * @brief Build a Hello frame
 *
 * @param frame         Frame to populate
 * @param build_id      Controller build identifier
 */
void msg_build_hello(frame_t *frame, uint16_t build_id);

/**
 * @brief Build a StreamCtrl frame
 *
 * @param frame         Frame to populate
 * @param start         true to start streaming, false to stop
 * @param block_id      Block ID for stream start/stop
 */
void msg_build_stream_ctrl(frame_t *frame, bool start, block_id_t block_id);

/**
 * @brief Build an ACK frame
 *
 * @param frame         Frame to populate
 * @param acked_seq     Sequence number being acknowledged
 */
void msg_build_ack(frame_t *frame, uint16_t acked_seq);

/**
 * @brief Build a NACK frame
 *
 * @param frame         Frame to populate
 * @param nacked_seq    Sequence number being negatively acknowledged
 */
void msg_build_nack(frame_t *frame, uint16_t nacked_seq);

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Create a NoteEvent structure
 */
static inline note_event_t note_event_create(midi_note_t note,
                                              velocity_t vel,
                                              bool on,
                                              midi_channel_t chan,
                                              block_id_t apply_block) {
    note_event_t evt = {
        .apply_block_id = apply_block,
        .note = note,
        .vel = vel,
        .on = on ? 1 : 0,
        .chan = chan,
        .flags = NOTE_FLAG_NONE
    };
    return evt;
}

/**
 * @brief Create a SetMacro structure
 */
static inline set_macro_t set_macro_create(macro_id_t id,
                                            q15_t value,
                                            block_id_t apply_block) {
    set_macro_t macro = {
        .apply_block_id = apply_block,
        .macro_id = id,
        .value_q15 = (uint16_t)value,
        .reserved = 0
    };
    return macro;
}

/**
 * @brief Create a SetTreeTarget structure
 */
static inline set_tree_target_t set_tree_target_create(tree_index_t idx,
                                                        uint16_t tau_ms,
                                                        tree_morph_curve_t curve,
                                                        block_id_t apply_block) {
    set_tree_target_t tree = {
        .apply_block_id = apply_block,
        .tree_index = idx,
        .tau_ms = tau_ms,
        .curve = (uint8_t)curve
    };
    return tree;
}

#endif // SYMCRASH_MESSAGES_H
