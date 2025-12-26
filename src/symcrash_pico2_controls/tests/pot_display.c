/**
 * @file pot_display.c
 * @brief Display MCP3008 potentiometer values on GC9A01 LCD
 *
 * LCD Wiring (SPI1):
 *   GP10 - SCL (SCK)
 *   GP11 - SDA (MOSI)
 *   GP13 - CS
 *   GP5  - DC
 *   GP9  - RST
 *
 * MCP3008 Wiring (SPI0):
 *   GP18 - SCK
 *   GP19 - MOSI
 *   GP16 - MISO
 *   GP17 - CS
 */

#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include <stdio.h>
#include <string.h>

// =============================================================================
// LCD Pin definitions - SPI1
// =============================================================================
#define LCD_SPI     spi1
#define LCD_SCK     10      // GP10 - SPI1 SCK
#define LCD_MOSI    11      // GP11 - SPI1 TX (MOSI/SDA)
#define LCD_CS      13      // GP13 - CS
#define LCD_DC      5       // GP5  - DC (Data/Command)
#define LCD_RST     9       // GP9  - Reset

// =============================================================================
// MCP3008 Pin definitions - SPI0
// =============================================================================
#define ADC_SPI     spi0
#define ADC_SCK     18      // GP18 - SPI0 SCK
#define ADC_MOSI    19      // GP19 - SPI0 TX (MOSI)
#define ADC_MISO    16      // GP16 - SPI0 RX (MISO)
#define ADC_CS      17      // GP17 - CS
#define NUM_POTS    6       // Number of pots to read

// =============================================================================
// GC9A01 commands
// =============================================================================
#define GC9A01_SWRESET   0x01
#define GC9A01_SLPOUT    0x11
#define GC9A01_INVON     0x21
#define GC9A01_DISPON    0x29
#define GC9A01_CASET     0x2A
#define GC9A01_RASET     0x2B
#define GC9A01_RAMWR     0x2C
#define GC9A01_TEON      0x35
#define GC9A01_MADCTL    0x36
#define GC9A01_COLMOD    0x3A
#define GC9A01_POWER2    0xC3
#define GC9A01_POWER3    0xC4
#define GC9A01_POWER4    0xC9
#define GC9A01_FRAMERATE 0xE8
#define GC9A01_INREGEN1  0xFE
#define GC9A01_INREGEN2  0xEF
#define GC9A01_GAMMA1    0xF0
#define GC9A01_GAMMA2    0xF1
#define GC9A01_GAMMA3    0xF2
#define GC9A01_GAMMA4    0xF3

// =============================================================================
// Colors (RGB565)
// =============================================================================
#define BLACK   0x0000
#define WHITE   0xFFFF
#define RED     0xF800
#define GREEN   0x07E0
#define BLUE    0x001F
#define YELLOW  0xFFE0
#define CYAN    0x07FF
#define GRAY    0x8410
#define DARKGRAY 0x4208

// =============================================================================
// Simple 5x7 font (ASCII 32-127)
// =============================================================================
static const uint8_t font5x7[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, // 32 (space)
    0x00, 0x00, 0x5F, 0x00, 0x00, // 33 !
    0x00, 0x07, 0x00, 0x07, 0x00, // 34 "
    0x14, 0x7F, 0x14, 0x7F, 0x14, // 35 #
    0x24, 0x2A, 0x7F, 0x2A, 0x12, // 36 $
    0x23, 0x13, 0x08, 0x64, 0x62, // 37 %
    0x36, 0x49, 0x55, 0x22, 0x50, // 38 &
    0x00, 0x05, 0x03, 0x00, 0x00, // 39 '
    0x00, 0x1C, 0x22, 0x41, 0x00, // 40 (
    0x00, 0x41, 0x22, 0x1C, 0x00, // 41 )
    0x08, 0x2A, 0x1C, 0x2A, 0x08, // 42 *
    0x08, 0x08, 0x3E, 0x08, 0x08, // 43 +
    0x00, 0x50, 0x30, 0x00, 0x00, // 44 ,
    0x08, 0x08, 0x08, 0x08, 0x08, // 45 -
    0x00, 0x60, 0x60, 0x00, 0x00, // 46 .
    0x20, 0x10, 0x08, 0x04, 0x02, // 47 /
    0x3E, 0x51, 0x49, 0x45, 0x3E, // 48 0
    0x00, 0x42, 0x7F, 0x40, 0x00, // 49 1
    0x42, 0x61, 0x51, 0x49, 0x46, // 50 2
    0x21, 0x41, 0x45, 0x4B, 0x31, // 51 3
    0x18, 0x14, 0x12, 0x7F, 0x10, // 52 4
    0x27, 0x45, 0x45, 0x45, 0x39, // 53 5
    0x3C, 0x4A, 0x49, 0x49, 0x30, // 54 6
    0x01, 0x71, 0x09, 0x05, 0x03, // 55 7
    0x36, 0x49, 0x49, 0x49, 0x36, // 56 8
    0x06, 0x49, 0x49, 0x29, 0x1E, // 57 9
    0x00, 0x36, 0x36, 0x00, 0x00, // 58 :
    0x00, 0x56, 0x36, 0x00, 0x00, // 59 ;
    0x00, 0x08, 0x14, 0x22, 0x41, // 60 <
    0x14, 0x14, 0x14, 0x14, 0x14, // 61 =
    0x41, 0x22, 0x14, 0x08, 0x00, // 62 >
    0x02, 0x01, 0x51, 0x09, 0x06, // 63 ?
    0x32, 0x49, 0x79, 0x41, 0x3E, // 64 @
    0x7E, 0x11, 0x11, 0x11, 0x7E, // 65 A
    0x7F, 0x49, 0x49, 0x49, 0x36, // 66 B
    0x3E, 0x41, 0x41, 0x41, 0x22, // 67 C
    0x7F, 0x41, 0x41, 0x22, 0x1C, // 68 D
    0x7F, 0x49, 0x49, 0x49, 0x41, // 69 E
    0x7F, 0x09, 0x09, 0x01, 0x01, // 70 F
    0x3E, 0x41, 0x41, 0x51, 0x32, // 71 G
    0x7F, 0x08, 0x08, 0x08, 0x7F, // 72 H
    0x00, 0x41, 0x7F, 0x41, 0x00, // 73 I
    0x20, 0x40, 0x41, 0x3F, 0x01, // 74 J
    0x7F, 0x08, 0x14, 0x22, 0x41, // 75 K
    0x7F, 0x40, 0x40, 0x40, 0x40, // 76 L
    0x7F, 0x02, 0x04, 0x02, 0x7F, // 77 M
    0x7F, 0x04, 0x08, 0x10, 0x7F, // 78 N
    0x3E, 0x41, 0x41, 0x41, 0x3E, // 79 O
    0x7F, 0x09, 0x09, 0x09, 0x06, // 80 P
    0x3E, 0x41, 0x51, 0x21, 0x5E, // 81 Q
    0x7F, 0x09, 0x19, 0x29, 0x46, // 82 R
    0x46, 0x49, 0x49, 0x49, 0x31, // 83 S
    0x01, 0x01, 0x7F, 0x01, 0x01, // 84 T
    0x3F, 0x40, 0x40, 0x40, 0x3F, // 85 U
    0x1F, 0x20, 0x40, 0x20, 0x1F, // 86 V
    0x7F, 0x20, 0x18, 0x20, 0x7F, // 87 W
    0x63, 0x14, 0x08, 0x14, 0x63, // 88 X
    0x03, 0x04, 0x78, 0x04, 0x03, // 89 Y
    0x61, 0x51, 0x49, 0x45, 0x43, // 90 Z
    0x00, 0x00, 0x7F, 0x41, 0x41, // 91 [
    0x02, 0x04, 0x08, 0x10, 0x20, // 92 backslash
    0x41, 0x41, 0x7F, 0x00, 0x00, // 93 ]
    0x04, 0x02, 0x01, 0x02, 0x04, // 94 ^
    0x40, 0x40, 0x40, 0x40, 0x40, // 95 _
    0x00, 0x01, 0x02, 0x04, 0x00, // 96 `
    0x20, 0x54, 0x54, 0x54, 0x78, // 97 a
    0x7F, 0x48, 0x44, 0x44, 0x38, // 98 b
    0x38, 0x44, 0x44, 0x44, 0x20, // 99 c
    0x38, 0x44, 0x44, 0x48, 0x7F, // 100 d
    0x38, 0x54, 0x54, 0x54, 0x18, // 101 e
    0x08, 0x7E, 0x09, 0x01, 0x02, // 102 f
    0x08, 0x14, 0x54, 0x54, 0x3C, // 103 g
    0x7F, 0x08, 0x04, 0x04, 0x78, // 104 h
    0x00, 0x44, 0x7D, 0x40, 0x00, // 105 i
    0x20, 0x40, 0x44, 0x3D, 0x00, // 106 j
    0x00, 0x7F, 0x10, 0x28, 0x44, // 107 k
    0x00, 0x41, 0x7F, 0x40, 0x00, // 108 l
    0x7C, 0x04, 0x18, 0x04, 0x78, // 109 m
    0x7C, 0x08, 0x04, 0x04, 0x78, // 110 n
    0x38, 0x44, 0x44, 0x44, 0x38, // 111 o
    0x7C, 0x14, 0x14, 0x14, 0x08, // 112 p
    0x08, 0x14, 0x14, 0x18, 0x7C, // 113 q
    0x7C, 0x08, 0x04, 0x04, 0x08, // 114 r
    0x48, 0x54, 0x54, 0x54, 0x20, // 115 s
    0x04, 0x3F, 0x44, 0x40, 0x20, // 116 t
    0x3C, 0x40, 0x40, 0x20, 0x7C, // 117 u
    0x1C, 0x20, 0x40, 0x20, 0x1C, // 118 v
    0x3C, 0x40, 0x30, 0x40, 0x3C, // 119 w
    0x44, 0x28, 0x10, 0x28, 0x44, // 120 x
    0x0C, 0x50, 0x50, 0x50, 0x3C, // 121 y
    0x44, 0x64, 0x54, 0x4C, 0x44, // 122 z
    0x00, 0x08, 0x36, 0x41, 0x00, // 123 {
    0x00, 0x00, 0x7F, 0x00, 0x00, // 124 |
    0x00, 0x41, 0x36, 0x08, 0x00, // 125 }
    0x08, 0x08, 0x2A, 0x1C, 0x08, // 126 ~
    0x08, 0x1C, 0x2A, 0x08, 0x08, // 127
};

// =============================================================================
// LCD Functions
// =============================================================================

static void lcd_write_cmd(uint8_t cmd) {
    gpio_put(LCD_CS, 0);
    gpio_put(LCD_DC, 0);
    spi_write_blocking(LCD_SPI, &cmd, 1);
    gpio_put(LCD_CS, 1);
}

static void lcd_write_data(const uint8_t *data, size_t len) {
    gpio_put(LCD_CS, 0);
    gpio_put(LCD_DC, 1);
    spi_write_blocking(LCD_SPI, data, len);
    gpio_put(LCD_CS, 1);
}

static void lcd_write_data_byte(uint8_t data) {
    lcd_write_data(&data, 1);
}

static void lcd_set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1) {
    uint8_t data[4];

    lcd_write_cmd(GC9A01_CASET);
    data[0] = x0 >> 8; data[1] = x0 & 0xFF;
    data[2] = x1 >> 8; data[3] = x1 & 0xFF;
    lcd_write_data(data, 4);

    lcd_write_cmd(GC9A01_RASET);
    data[0] = y0 >> 8; data[1] = y0 & 0xFF;
    data[2] = y1 >> 8; data[3] = y1 & 0xFF;
    lcd_write_data(data, 4);

    lcd_write_cmd(GC9A01_RAMWR);
}

static void lcd_fill_rect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t color) {
    lcd_set_window(x, y, x + w - 1, y + h - 1);

    uint8_t hi = color >> 8;
    uint8_t lo = color & 0xFF;
    uint8_t line[480];  // Max 240 pixels per line

    int line_bytes = w * 2;
    if (line_bytes > 480) line_bytes = 480;

    for (int i = 0; i < line_bytes; i += 2) {
        line[i] = hi;
        line[i + 1] = lo;
    }

    gpio_put(LCD_CS, 0);
    gpio_put(LCD_DC, 1);
    for (uint16_t row = 0; row < h; row++) {
        spi_write_blocking(LCD_SPI, line, line_bytes);
    }
    gpio_put(LCD_CS, 1);
}

static void lcd_fill(uint16_t color) {
    lcd_fill_rect(0, 0, 240, 240, color);
}

static void lcd_draw_char(uint16_t x, uint16_t y, char c, uint16_t fg, uint16_t bg, uint8_t scale) {
    if (c < 32 || c > 127) c = '?';

    const uint8_t *glyph = &font5x7[(c - 32) * 5];
    uint16_t char_w = 5 * scale;
    uint16_t char_h = 7 * scale;

    lcd_set_window(x, y, x + char_w - 1, y + char_h - 1);

    uint8_t fg_hi = fg >> 8, fg_lo = fg & 0xFF;
    uint8_t bg_hi = bg >> 8, bg_lo = bg & 0xFF;

    gpio_put(LCD_CS, 0);
    gpio_put(LCD_DC, 1);

    for (int row = 0; row < 7; row++) {
        for (int sy = 0; sy < scale; sy++) {
            for (int col = 0; col < 5; col++) {
                bool pixel = (glyph[col] >> row) & 1;
                uint8_t hi = pixel ? fg_hi : bg_hi;
                uint8_t lo = pixel ? fg_lo : bg_lo;
                for (int sx = 0; sx < scale; sx++) {
                    uint8_t px[2] = {hi, lo};
                    spi_write_blocking(LCD_SPI, px, 2);
                }
            }
        }
    }

    gpio_put(LCD_CS, 1);
}

static void lcd_draw_string(uint16_t x, uint16_t y, const char *str, uint16_t fg, uint16_t bg, uint8_t scale) {
    uint16_t char_w = 6 * scale;  // 5 pixels + 1 space
    while (*str) {
        lcd_draw_char(x, y, *str++, fg, bg, scale);
        x += char_w;
    }
}

static void lcd_init(void) {
    // Control pins
    gpio_init(LCD_CS);
    gpio_init(LCD_DC);
    gpio_init(LCD_RST);
    gpio_set_dir(LCD_CS, GPIO_OUT);
    gpio_set_dir(LCD_DC, GPIO_OUT);
    gpio_set_dir(LCD_RST, GPIO_OUT);

    gpio_put(LCD_CS, 1);
    gpio_put(LCD_DC, 0);
    gpio_put(LCD_RST, 1);

    // Hardware reset
    sleep_ms(50);
    gpio_put(LCD_RST, 0);
    sleep_ms(50);
    gpio_put(LCD_RST, 1);
    sleep_ms(150);

    // Initialize SPI1 for LCD
    spi_init(LCD_SPI, 62500000);  // 62.5 MHz - max stable for GC9A01
    spi_set_format(LCD_SPI, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST);
    gpio_set_function(LCD_SCK, GPIO_FUNC_SPI);
    gpio_set_function(LCD_MOSI, GPIO_FUNC_SPI);

    // Full Adafruit init sequence
    lcd_write_cmd(GC9A01_INREGEN2);
    lcd_write_cmd(0xEB); lcd_write_data_byte(0x14);
    lcd_write_cmd(GC9A01_INREGEN1);
    lcd_write_cmd(GC9A01_INREGEN2);
    lcd_write_cmd(0xEB); lcd_write_data_byte(0x14);

    lcd_write_cmd(0x84); lcd_write_data_byte(0x40);
    lcd_write_cmd(0x85); lcd_write_data_byte(0xFF);
    lcd_write_cmd(0x86); lcd_write_data_byte(0xFF);
    lcd_write_cmd(0x87); lcd_write_data_byte(0xFF);
    lcd_write_cmd(0x88); lcd_write_data_byte(0x0A);
    lcd_write_cmd(0x89); lcd_write_data_byte(0x21);
    lcd_write_cmd(0x8A); lcd_write_data_byte(0x00);
    lcd_write_cmd(0x8B); lcd_write_data_byte(0x80);
    lcd_write_cmd(0x8C); lcd_write_data_byte(0x01);
    lcd_write_cmd(0x8D); lcd_write_data_byte(0x01);
    lcd_write_cmd(0x8E); lcd_write_data_byte(0xFF);
    lcd_write_cmd(0x8F); lcd_write_data_byte(0xFF);

    lcd_write_cmd(0xB6);
    uint8_t b6[] = {0x00, 0x00};
    lcd_write_data(b6, 2);

    lcd_write_cmd(GC9A01_MADCTL);
    lcd_write_data_byte(0x48);

    lcd_write_cmd(GC9A01_COLMOD);
    lcd_write_data_byte(0x05);

    lcd_write_cmd(0x90);
    uint8_t r90[] = {0x08, 0x08, 0x08, 0x08};
    lcd_write_data(r90, 4);

    lcd_write_cmd(0xBD); lcd_write_data_byte(0x06);
    lcd_write_cmd(0xBC); lcd_write_data_byte(0x00);

    lcd_write_cmd(0xFF);
    uint8_t rff[] = {0x60, 0x01, 0x04};
    lcd_write_data(rff, 3);

    lcd_write_cmd(GC9A01_POWER2); lcd_write_data_byte(0x13);
    lcd_write_cmd(GC9A01_POWER3); lcd_write_data_byte(0x13);
    lcd_write_cmd(GC9A01_POWER4); lcd_write_data_byte(0x22);

    lcd_write_cmd(0xBE); lcd_write_data_byte(0x11);
    lcd_write_cmd(0xE1);
    uint8_t re1[] = {0x10, 0x0E};
    lcd_write_data(re1, 2);
    lcd_write_cmd(0xDF);
    uint8_t rdf[] = {0x21, 0x0C, 0x02};
    lcd_write_data(rdf, 3);

    lcd_write_cmd(GC9A01_GAMMA1);
    uint8_t g1[] = {0x45, 0x09, 0x08, 0x08, 0x26, 0x2A};
    lcd_write_data(g1, 6);
    lcd_write_cmd(GC9A01_GAMMA2);
    uint8_t g2[] = {0x43, 0x70, 0x72, 0x36, 0x37, 0x6F};
    lcd_write_data(g2, 6);
    lcd_write_cmd(GC9A01_GAMMA3);
    lcd_write_data(g1, 6);
    lcd_write_cmd(GC9A01_GAMMA4);
    lcd_write_data(g2, 6);

    lcd_write_cmd(0xED);
    uint8_t red[] = {0x1B, 0x0B};
    lcd_write_data(red, 2);
    lcd_write_cmd(0xAE); lcd_write_data_byte(0x77);
    lcd_write_cmd(0xCD); lcd_write_data_byte(0x63);

    lcd_write_cmd(GC9A01_FRAMERATE);
    lcd_write_data_byte(0x34);

    lcd_write_cmd(0x62);
    uint8_t r62[] = {0x18, 0x0D, 0x71, 0xED, 0x70, 0x70, 0x18, 0x0F, 0x71, 0xEF, 0x70, 0x70};
    lcd_write_data(r62, 12);
    lcd_write_cmd(0x63);
    uint8_t r63[] = {0x18, 0x11, 0x71, 0xF1, 0x70, 0x70, 0x18, 0x13, 0x71, 0xF3, 0x70, 0x70};
    lcd_write_data(r63, 12);
    lcd_write_cmd(0x64);
    uint8_t r64[] = {0x28, 0x29, 0xF1, 0x01, 0xF1, 0x00, 0x07};
    lcd_write_data(r64, 7);
    lcd_write_cmd(0x66);
    uint8_t r66[] = {0x3C, 0x00, 0xCD, 0x67, 0x45, 0x45, 0x10, 0x00, 0x00, 0x00};
    lcd_write_data(r66, 10);
    lcd_write_cmd(0x67);
    uint8_t r67[] = {0x00, 0x3C, 0x00, 0x00, 0x00, 0x01, 0x54, 0x10, 0x32, 0x98};
    lcd_write_data(r67, 10);
    lcd_write_cmd(0x74);
    uint8_t r74[] = {0x10, 0x85, 0x80, 0x00, 0x00, 0x4E, 0x00};
    lcd_write_data(r74, 7);
    lcd_write_cmd(0x98);
    uint8_t r98[] = {0x3E, 0x07};
    lcd_write_data(r98, 2);

    lcd_write_cmd(GC9A01_TEON);
    lcd_write_cmd(GC9A01_INVON);
    lcd_write_cmd(GC9A01_SLPOUT);
    sleep_ms(150);
    lcd_write_cmd(GC9A01_DISPON);
    sleep_ms(150);
}

// =============================================================================
// MCP3008 Functions
// =============================================================================

static void adc_init_hw(void) {
    // Initialize SPI0 for MCP3008
    spi_init(ADC_SPI, 1000000);  // 1 MHz
    spi_set_format(ADC_SPI, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST);

    gpio_set_function(ADC_SCK, GPIO_FUNC_SPI);
    gpio_set_function(ADC_MOSI, GPIO_FUNC_SPI);
    gpio_set_function(ADC_MISO, GPIO_FUNC_SPI);

    gpio_init(ADC_CS);
    gpio_set_dir(ADC_CS, GPIO_OUT);
    gpio_put(ADC_CS, 1);
}

static uint16_t adc_read_channel(uint8_t channel) {
    if (channel > 7) return 0;

    uint8_t tx[3] = {
        0x01,                       // Start bit
        (0x80 | (channel << 4)),    // Single-ended, channel select
        0x00                        // Dummy
    };
    uint8_t rx[3] = {0};

    gpio_put(ADC_CS, 0);
    spi_write_read_blocking(ADC_SPI, tx, rx, 3);
    gpio_put(ADC_CS, 1);

    // Extract 10-bit result
    return ((rx[1] & 0x03) << 8) | rx[2];
}

// =============================================================================
// Main
// =============================================================================

int main(void) {
    stdio_init_all();

    // Initialize CYW43 for LED
    if (cyw43_arch_init()) {
        while (1) tight_loop_contents();
    }

    // Blink LED
    for (int i = 0; i < 3; i++) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(100);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(100);
    }

    // Initialize hardware
    lcd_init();
    adc_init_hw();

    // Clear screen
    lcd_fill(BLACK);

    // Draw title
    lcd_draw_string(60, 30, "POT VALUES", CYAN, BLACK, 2);

    // Draw table header
    lcd_draw_string(40, 60, "CH  RAW   PCT", WHITE, BLACK, 2);

    // LED on
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);

    // Pot names
    const char *pot_names[] = {"P0", "P1", "P2", "P3", "P4", "P5"};

    uint32_t frame = 0;

    while (true) {
        // Read and display each pot
        for (int ch = 0; ch < NUM_POTS; ch++) {
            uint16_t raw = adc_read_channel(ch);
            uint8_t pct = (raw * 100) / 1023;

            // Format: "Px  1023  100%"
            char line[20];
            snprintf(line, sizeof(line), "%s %4d  %3d%%", pot_names[ch], raw, pct);

            uint16_t y = 85 + ch * 22;

            // Color based on value
            uint16_t color;
            if (pct < 25) color = BLUE;
            else if (pct < 50) color = CYAN;
            else if (pct < 75) color = GREEN;
            else color = YELLOW;

            lcd_draw_string(30, y, line, color, BLACK, 2);

            // Draw bar graph
            uint16_t bar_x = 170;
            uint16_t bar_w = 50;
            uint16_t bar_h = 16;
            uint16_t filled = (raw * bar_w) / 1023;

            // Background
            lcd_fill_rect(bar_x, y, bar_w, bar_h, DARKGRAY);
            // Filled portion
            if (filled > 0) {
                lcd_fill_rect(bar_x, y, filled, bar_h, color);
            }
        }

        // Frame counter
        char fbuf[16];
        snprintf(fbuf, sizeof(fbuf), "F:%lu", (unsigned long)frame++);
        lcd_draw_string(10, 220, fbuf, GRAY, BLACK, 1);

        // Toggle LED
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, frame & 1);

        sleep_ms(50);  // ~20 Hz update
    }

    return 0;
}
