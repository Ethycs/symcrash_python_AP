/**
 * @file error_codes.h
 * @brief Error code definitions
 */

#ifndef SYMCRASH_ERROR_CODES_H
#define SYMCRASH_ERROR_CODES_H

typedef enum {
    ERR_OK = 0,             // Success
    ERR_INVALID_PARAM,      // Invalid parameter
    ERR_BUFFER_FULL,        // Buffer overflow
    ERR_BUFFER_EMPTY,       // Buffer underflow
    ERR_TIMEOUT,            // Operation timed out
    ERR_CRC_MISMATCH,       // CRC validation failed
    ERR_FRAME_INVALID,      // Invalid frame format
    ERR_NOT_CONNECTED,      // Not connected to synthesis engine
    ERR_HARDWARE,           // Hardware error
} error_t;

#endif // SYMCRASH_ERROR_CODES_H
