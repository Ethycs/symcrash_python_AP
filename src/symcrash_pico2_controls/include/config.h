/**
 * @file config.h
 * @brief Hardware configuration and pin assignments
 *
 * Pin mapping for Symcrash Pico2 Hardware UI Controller.
 * Adjust these values to match your actual hardware wiring.
 */

#ifndef SYMCRASH_CONFIG_H
#define SYMCRASH_CONFIG_H

// =============================================================================
// Transport Mode Selection
// =============================================================================
// Set to 1 for USB CDC mode (dual virtual COM ports over USB)
// Set to 0 for hardware UART mode (physical GPIO pins)
#define TRANSPORT_USB_MODE  1

// =============================================================================
// C-UART (Control Channel) - UART0
// Reliable, acknowledged communication for control messages
// =============================================================================

#define C_UART_ID           uart0
#define C_UART_TX_PIN       0       // GP0 - UART0 TX
#define C_UART_RX_PIN       1       // GP1 - UART0 RX
#define C_UART_BAUD         1000000 // 1 Mbps

// =============================================================================
// D-UART (Data Channel) - UART1
// High-throughput streaming for voice data
// =============================================================================

#define D_UART_ID           uart1
#define D_UART_TX_PIN       8       // GP8 - UART1 TX
#define D_UART_RX_PIN       9       // GP9 - UART1 RX
#define D_UART_BAUD         3000000 // 3 Mbps

// =============================================================================
// MCP3008 ADCs on SPI0 (10 potentiometers total)
// =============================================================================

#define MCP3008_SPI_ID      spi0
#define SPI0_SCK_PIN        18      // GP18 - SPI0 SCK
#define SPI0_MOSI_PIN       19      // GP19 - SPI0 TX (MOSI)
#define SPI0_MISO_PIN       16      // GP16 - SPI0 RX (MISO)
#define SPI0_BAUD           1000000 // 1 MHz SPI clock

// MCP3008 #0 - First 6 pots (ADSR + Filter)
#define MCP3008_0_CS_PIN    17      // GP17 - Chip Select for ADC 0

// MCP3008 #1 - Additional 4 pots (Oscillator control)
#define MCP3008_1_CS_PIN    27      // GP27 - Chip Select for ADC 1

// Legacy alias for single-ADC code
#define MCP3008_CS_PIN      MCP3008_0_CS_PIN

// Channel Assignments - ADC 0 (6 pots on channels 0-5)
#define POT_MACRO_0         0       // CH0 - Attack
#define POT_MACRO_1         1       // CH1 - Decay
#define POT_MACRO_2         2       // CH2 - Sustain
#define POT_MACRO_3         3       // CH3 - Release
#define POT_MACRO_4         4       // CH4 - Filter Cutoff
#define POT_MACRO_5         5       // CH5 - Filter Resonance

// Channel Assignments - ADC 1 (4 pots on channels 0-3)
#define POT_MACRO_6         6       // ADC1:CH0 - Harmonic Mix / Brightness
#define POT_MACRO_7         7       // ADC1:CH1 - Eta (Dry/Wet)
#define POT_MACRO_8         8       // ADC1:CH2 - Detune / Spread
#define POT_MACRO_9         9       // ADC1:CH3 - Master Gain

#define NUM_POTS_ADC0       6       // Pots on first MCP3008
#define NUM_POTS_ADC1       4       // Pots on second MCP3008
#define NUM_POTS            10      // Total pots

// ADC scanning parameters
#define ADC_SCAN_RATE_HZ    1000    // 1 kHz scanning rate
#define ADC_SMOOTHING       8       // Exponential smoothing factor (1-16)
#define ADC_DEADBAND        4       // Deadband for change detection (0-1023)

// =============================================================================
// GC9A01 Round LCDs (2 displays on SPI1)
// =============================================================================

#define LCD_SPI_ID          spi1
#define SPI1_SCK_PIN        10      // GP10 - SPI1 SCK
#define SPI1_MOSI_PIN       11      // GP11 - SPI1 TX (MOSI/SDA)
#define SPI1_BAUD           62500000 // 62.5 MHz SPI clock (max stable)

// Display 0 - Actual wiring
#define LCD0_CS_PIN         13      // GP13 - Display 0 CS
#define LCD0_DC_PIN         5       // GP5  - Display 0 DC (Data/Command)
#define LCD0_RST_PIN        9       // GP9  - Display 0 Reset
#define LCD0_BL_PIN         8       // GP8  - Display 0 Backlight (optional)

// Display 1 (second screen)
#define LCD1_CS_PIN         2       // GP2  - Display 1 CS
#define LCD1_DC_PIN         5       // GP5  - Display 1 DC (shared DC possible)
#define LCD1_RST_PIN        9       // GP9  - Display 1 Reset (shared RST possible)

// Display parameters
#define LCD_WIDTH           240     // GC9A01 is 240x240 round
#define LCD_HEIGHT          240
#define LCD_REFRESH_HZ      30      // Display refresh rate

// =============================================================================
// Rotary Encoder
// =============================================================================

// Encoder pins - moved to avoid conflicts
// Original: GP2, GP3, GP4 - caused issues with LCD1
// New: GP20, GP21, GP22 - unused GPIOs
#define ENC0_A_PIN          20      // GP20 - Encoder A
#define ENC0_B_PIN          21      // GP21 - Encoder B
#define ENC0_SW_PIN         22      // GP22 - Encoder Switch
#define NUM_ENCODERS        1

// Encoder parameters
#define ENCODER_PULSES_PER_DETENT   4   // Typical for most encoders

// =============================================================================
// Buttons (Active Low with internal pull-up)
// =============================================================================

#define BTN_0               5       // GP5
#define BTN_1               6       // GP6
#define BTN_2               7       // GP7
#define BTN_3               15      // GP15
#define NUM_BUTTONS         4

// Button debounce parameters
#define BUTTON_DEBOUNCE_MS      20      // Debounce time
#define BUTTON_LONG_PRESS_MS    500     // Long press threshold
#define BUTTON_REPEAT_MS        100     // Auto-repeat rate (0 = disabled)

// =============================================================================
// Status LEDs
// =============================================================================

// On Pico 2 W, onboard LED is controlled via CYW43 WiFi chip
// Use cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, value) instead of gpio_put
// For standard Pico 2, LED is on GP25
#ifdef PICO_CYW43_SUPPORTED
#define LED_STATUS          CYW43_WL_GPIO_LED_PIN  // Pico 2 W - via WiFi chip
#else
#define LED_STATUS          25      // Standard Pico 2 - GP25
#endif

// LED blink patterns (in ms)
#define LED_BLINK_FAST      100
#define LED_BLINK_SLOW      500
#define LED_BLINK_HEARTBEAT 1000

// =============================================================================
// Protocol Constants
// =============================================================================

#define FRAME_MAGIC         0xDEC0  // WSE-STD-100 frame magic
#define PROTOCOL_VERSION    0x01
#define COBS_DELIMITER      0x00

// Timing
#define LEAD_BLOCKS         8       // Schedule commands this many blocks ahead
#define BLOCK_SAMPLES       64      // Samples per block
#define SAMPLE_RATE         48000   // Audio sample rate

// Block duration in microseconds: (64 / 48000) * 1_000_000 = 1333.33
#define BLOCK_DURATION_US   1333

// =============================================================================
// Buffer Sizes
// =============================================================================

#define TX_BUFFER_SIZE      256     // UART TX ring buffer
#define RX_BUFFER_SIZE      512     // UART RX ring buffer
#define MAX_PAYLOAD_SIZE    128     // Maximum frame payload

#endif // SYMCRASH_CONFIG_H
