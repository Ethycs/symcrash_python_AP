/**
 * @file crc32c.h
 * @brief CRC-32C (Castagnoli) implementation
 *
 * Uses polynomial 0x82F63B78, matching WSE-STD-100.
 */

#ifndef SYMCRASH_CRC32C_H
#define SYMCRASH_CRC32C_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Calculate CRC-32C checksum
 *
 * @param data   Data buffer
 * @param len    Length of data
 * @return 32-bit CRC checksum
 */
uint32_t crc32c(const uint8_t *data, size_t len);

/**
 * @brief Continue CRC-32C calculation (for incremental computation)
 *
 * @param crc    Previous CRC value (use 0xFFFFFFFF for initial call)
 * @param data   Data buffer
 * @param len    Length of data
 * @return Updated CRC value (XOR with 0xFFFFFFFF for final result)
 */
uint32_t crc32c_update(uint32_t crc, const uint8_t *data, size_t len);

#endif // SYMCRASH_CRC32C_H
