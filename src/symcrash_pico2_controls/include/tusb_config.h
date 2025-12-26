/**
 * @file tusb_config.h
 * @brief TinyUSB configuration for dual CDC (Control + Data channels)
 *
 * This configures USB with two CDC interfaces:
 * - CDC 0: Control channel (C-UART equivalent)
 * - CDC 1: Data channel (D-UART equivalent)
 */

#ifndef TUSB_CONFIG_H
#define TUSB_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

//--------------------------------------------------------------------
// COMMON CONFIGURATION
//--------------------------------------------------------------------

// Defined by compiler flags for Pico SDK
// For RP2350 (Pico 2), the SDK defines this automatically
// Do NOT override - let the SDK set it based on PICO_BOARD

#define CFG_TUSB_RHPORT0_MODE   OPT_MODE_DEVICE

#ifndef CFG_TUSB_OS
#define CFG_TUSB_OS             OPT_OS_PICO
#endif

#ifndef CFG_TUSB_MEM_SECTION
#define CFG_TUSB_MEM_SECTION
#endif

#ifndef CFG_TUSB_MEM_ALIGN
#define CFG_TUSB_MEM_ALIGN      __attribute__ ((aligned(4)))
#endif

//--------------------------------------------------------------------
// DEVICE CONFIGURATION
//--------------------------------------------------------------------

#ifndef CFG_TUD_ENDPOINT0_SIZE
#define CFG_TUD_ENDPOINT0_SIZE  64
#endif

//--------------------------------------------------------------------
// CLASS DRIVER CONFIGURATION
//--------------------------------------------------------------------

// Enable dual CDC
#define CFG_TUD_CDC             2
#define CFG_TUD_MSC             0
#define CFG_TUD_HID             0
#define CFG_TUD_MIDI            0
#define CFG_TUD_VENDOR          0

// CDC FIFO sizes
#define CFG_TUD_CDC_RX_BUFSIZE  256
#define CFG_TUD_CDC_TX_BUFSIZE  256

// Enable line coding callback for baud rate changes
#define CFG_TUD_CDC_LINE_CODING_ON_EACH_SET 0

#ifdef __cplusplus
}
#endif

#endif // TUSB_CONFIG_H
