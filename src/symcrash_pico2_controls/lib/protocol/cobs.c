/**
 * @file cobs.c
 * @brief COBS encoder/decoder implementation
 *
 * Matches the Python implementation in dual_uart_link.py.
 */

#include "cobs.h"

size_t cobs_encode(const uint8_t *input, size_t input_len, uint8_t *output) {
    if (input == NULL || output == NULL) {
        return 0;
    }

    if (input_len == 0) {
        output[0] = 0x01;
        return 1;
    }

    size_t read_idx = 0;
    size_t write_idx = 0;
    size_t code_idx = 0;
    uint8_t code = 1;

    // Placeholder for first code byte
    output[write_idx++] = 0;

    while (read_idx < input_len) {
        if (input[read_idx] == 0) {
            // Found zero byte - write code and start new block
            output[code_idx] = code;
            code_idx = write_idx++;
            output[code_idx] = 0;  // Placeholder
            code = 1;
        } else {
            // Non-zero byte - copy to output
            output[write_idx++] = input[read_idx];
            code++;

            // Maximum code value reached - start new block
            if (code == 0xFF) {
                output[code_idx] = code;
                code_idx = write_idx++;
                output[code_idx] = 0;  // Placeholder
                code = 1;
            }
        }
        read_idx++;
    }

    // Write final code byte
    output[code_idx] = code;

    return write_idx;
}

size_t cobs_decode(const uint8_t *input, size_t input_len, uint8_t *output) {
    if (input == NULL || output == NULL || input_len == 0) {
        return 0;
    }

    size_t read_idx = 0;
    size_t write_idx = 0;

    while (read_idx < input_len) {
        uint8_t code = input[read_idx];

        if (code == 0) {
            // Invalid: zero byte in COBS-encoded data
            return 0;
        }

        read_idx++;

        // Copy (code - 1) data bytes
        for (uint8_t i = 1; i < code; i++) {
            if (read_idx >= input_len) {
                // Unexpected end of data
                return 0;
            }
            output[write_idx++] = input[read_idx++];
        }

        // Add zero byte between groups, but not after the last group
        // and not if code was 0xFF (special case: no implicit zero)
        if (code < 0xFF && read_idx < input_len) {
            output[write_idx++] = 0;
        }
    }

    return write_idx;
}
