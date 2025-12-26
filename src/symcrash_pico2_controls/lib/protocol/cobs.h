/**
 * @file cobs.h
 * @brief COBS (Consistent Overhead Byte Stuffing) encoder/decoder
 *
 * COBS eliminates all zero bytes from data, allowing 0x00 to be used
 * as an unambiguous frame delimiter (per WSE-STD-100 Section 4.1).
 */

#ifndef SYMCRASH_COBS_H
#define SYMCRASH_COBS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/**
 * @brief Encode data using COBS
 *
 * @param input     Source data buffer
 * @param input_len Length of source data
 * @param output    Destination buffer (must be at least input_len + (input_len/254) + 1)
 * @return Number of bytes written to output, or 0 on error
 *
 * @note Output does NOT include trailing delimiter (0x00).
 *       Caller should append delimiter if needed.
 */
size_t cobs_encode(const uint8_t *input, size_t input_len, uint8_t *output);

/**
 * @brief Decode COBS-encoded data
 *
 * @param input     COBS-encoded data (without trailing delimiter)
 * @param input_len Length of encoded data
 * @param output    Destination buffer (can be same size as input)
 * @return Number of decoded bytes, or 0 on error (invalid encoding)
 */
size_t cobs_decode(const uint8_t *input, size_t input_len, uint8_t *output);

/**
 * @brief Calculate maximum encoded size for given input length
 *
 * @param input_len Length of data to encode
 * @return Maximum size of COBS-encoded output
 */
static inline size_t cobs_max_encoded_size(size_t input_len) {
    return input_len + (input_len / 254) + 1;
}

#endif // SYMCRASH_COBS_H
