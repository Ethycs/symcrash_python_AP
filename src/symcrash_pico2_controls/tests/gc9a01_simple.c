/**
 * @file gc9a01_simple.c
 * @brief Minimal GC9A01 test - just fill screen with colors
 *
 * Wiring:
 *   GP10 - SCL (SCK)
 *   GP11 - SDA (MOSI)
 *   GP13 - CS
 *   GP7  - DC
 *   GP9  - RST
 */

#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"

// Pin definitions - SPI1 on Pico 2
// GP10/GP11 are valid SPI1 pins (SCK=GP10, TX=GP11)
#define LCD_SPI     spi1
#define PIN_SCK     10      // GP10 - SPI1 SCK
#define PIN_MOSI    11      // GP11 - SPI1 TX (MOSI/SDA)
#define PIN_CS      13      // GP13 - CS
#define PIN_DC      5       // GP5  - DC (Data/Command)
#define PIN_RST     9       // GP9  - Reset
#define PIN_BL      8       // GP8  - Backlight (optional, unused)

// GC9A01 commands (from Adafruit library)
#define GC9A01_SWRESET   0x01
#define GC9A01_SLPOUT    0x11
#define GC9A01_INVON     0x21
#define GC9A01_DISPOFF   0x28
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

// Colors (RGB565)
#define RED     0xF800
#define GREEN   0x07E0
#define BLUE    0x001F
#define WHITE   0xFFFF
#define BLACK   0x0000
#define YELLOW  0xFFE0
#define CYAN    0x07FF
#define MAGENTA 0xF81F

static void lcd_cs_select(void) {
    gpio_put(PIN_CS, 0);
}

static void lcd_cs_deselect(void) {
    gpio_put(PIN_CS, 1);
}

static void lcd_dc_command(void) {
    gpio_put(PIN_DC, 0);
}

static void lcd_dc_data(void) {
    gpio_put(PIN_DC, 1);
}

static void lcd_write_cmd(uint8_t cmd) {
    lcd_cs_select();
    lcd_dc_command();
    spi_write_blocking(LCD_SPI, &cmd, 1);
    lcd_cs_deselect();
}

static void lcd_write_data(const uint8_t *data, size_t len) {
    lcd_cs_select();
    lcd_dc_data();
    spi_write_blocking(LCD_SPI, data, len);
    lcd_cs_deselect();
}

static void lcd_write_data_byte(uint8_t data) {
    lcd_write_data(&data, 1);
}

static void lcd_init(void) {
    // Control pins as GPIO - initialize FIRST before SPI
    gpio_init(PIN_CS);
    gpio_init(PIN_DC);
    gpio_init(PIN_RST);
    gpio_init(PIN_BL);
    gpio_set_dir(PIN_CS, GPIO_OUT);
    gpio_set_dir(PIN_DC, GPIO_OUT);
    gpio_set_dir(PIN_RST, GPIO_OUT);
    gpio_set_dir(PIN_BL, GPIO_OUT);

    gpio_put(PIN_CS, 1);  // CS high (inactive)
    gpio_put(PIN_DC, 0);
    gpio_put(PIN_RST, 1);
    gpio_put(PIN_BL, 1);  // Backlight ON

    // Hardware reset - proper sequence
    sleep_ms(50);
    gpio_put(PIN_RST, 0);
    sleep_ms(50);
    gpio_put(PIN_RST, 1);
    sleep_ms(150);

    // Initialize SPI - 62.5 MHz for fast display updates
    spi_init(LCD_SPI, 62500000);
    // Mode 0 (CPOL=0, CPHA=0) - standard for GC9A01
    spi_set_format(LCD_SPI, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST);

    gpio_set_function(PIN_SCK, GPIO_FUNC_SPI);
    gpio_set_function(PIN_MOSI, GPIO_FUNC_SPI);

    // =========================================================================
    // Full Adafruit GC9A01A initialization sequence
    // =========================================================================

    // Inter register enable
    lcd_write_cmd(GC9A01_INREGEN2);
    lcd_write_cmd(0xEB); lcd_write_data_byte(0x14);
    lcd_write_cmd(GC9A01_INREGEN1);
    lcd_write_cmd(GC9A01_INREGEN2);
    lcd_write_cmd(0xEB); lcd_write_data_byte(0x14);

    // Undocumented power/timing registers
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

    // Display function control
    lcd_write_cmd(0xB6);
    uint8_t b6_data[] = {0x00, 0x00};
    lcd_write_data(b6_data, 2);

    // Memory access control: MX | BGR
    lcd_write_cmd(GC9A01_MADCTL);
    lcd_write_data_byte(0x48);

    // Pixel format: 16-bit (RGB565)
    lcd_write_cmd(GC9A01_COLMOD);
    lcd_write_data_byte(0x05);

    // More undocumented registers
    lcd_write_cmd(0x90);
    uint8_t r90_data[] = {0x08, 0x08, 0x08, 0x08};
    lcd_write_data(r90_data, 4);

    lcd_write_cmd(0xBD); lcd_write_data_byte(0x06);
    lcd_write_cmd(0xBC); lcd_write_data_byte(0x00);

    lcd_write_cmd(0xFF);
    uint8_t rff_data[] = {0x60, 0x01, 0x04};
    lcd_write_data(rff_data, 3);

    // Power control
    lcd_write_cmd(GC9A01_POWER2); lcd_write_data_byte(0x13);
    lcd_write_cmd(GC9A01_POWER3); lcd_write_data_byte(0x13);
    lcd_write_cmd(GC9A01_POWER4); lcd_write_data_byte(0x22);

    lcd_write_cmd(0xBE); lcd_write_data_byte(0x11);

    lcd_write_cmd(0xE1);
    uint8_t re1_data[] = {0x10, 0x0E};
    lcd_write_data(re1_data, 2);

    lcd_write_cmd(0xDF);
    uint8_t rdf_data[] = {0x21, 0x0C, 0x02};
    lcd_write_data(rdf_data, 3);

    // Gamma curves
    lcd_write_cmd(GC9A01_GAMMA1);
    uint8_t gamma1[] = {0x45, 0x09, 0x08, 0x08, 0x26, 0x2A};
    lcd_write_data(gamma1, 6);

    lcd_write_cmd(GC9A01_GAMMA2);
    uint8_t gamma2[] = {0x43, 0x70, 0x72, 0x36, 0x37, 0x6F};
    lcd_write_data(gamma2, 6);

    lcd_write_cmd(GC9A01_GAMMA3);
    uint8_t gamma3[] = {0x45, 0x09, 0x08, 0x08, 0x26, 0x2A};
    lcd_write_data(gamma3, 6);

    lcd_write_cmd(GC9A01_GAMMA4);
    uint8_t gamma4[] = {0x43, 0x70, 0x72, 0x36, 0x37, 0x6F};
    lcd_write_data(gamma4, 6);

    // More undocumented registers
    lcd_write_cmd(0xED);
    uint8_t red_data[] = {0x1B, 0x0B};
    lcd_write_data(red_data, 2);

    lcd_write_cmd(0xAE); lcd_write_data_byte(0x77);
    lcd_write_cmd(0xCD); lcd_write_data_byte(0x63);

    // Frame rate
    lcd_write_cmd(GC9A01_FRAMERATE);
    lcd_write_data_byte(0x34);

    // Long undocumented sequences (display timing)
    lcd_write_cmd(0x62);
    uint8_t r62_data[] = {0x18, 0x0D, 0x71, 0xED, 0x70, 0x70,
                          0x18, 0x0F, 0x71, 0xEF, 0x70, 0x70};
    lcd_write_data(r62_data, 12);

    lcd_write_cmd(0x63);
    uint8_t r63_data[] = {0x18, 0x11, 0x71, 0xF1, 0x70, 0x70,
                          0x18, 0x13, 0x71, 0xF3, 0x70, 0x70};
    lcd_write_data(r63_data, 12);

    lcd_write_cmd(0x64);
    uint8_t r64_data[] = {0x28, 0x29, 0xF1, 0x01, 0xF1, 0x00, 0x07};
    lcd_write_data(r64_data, 7);

    lcd_write_cmd(0x66);
    uint8_t r66_data[] = {0x3C, 0x00, 0xCD, 0x67, 0x45, 0x45,
                          0x10, 0x00, 0x00, 0x00};
    lcd_write_data(r66_data, 10);

    lcd_write_cmd(0x67);
    uint8_t r67_data[] = {0x00, 0x3C, 0x00, 0x00, 0x00, 0x01,
                          0x54, 0x10, 0x32, 0x98};
    lcd_write_data(r67_data, 10);

    lcd_write_cmd(0x74);
    uint8_t r74_data[] = {0x10, 0x85, 0x80, 0x00, 0x00, 0x4E, 0x00};
    lcd_write_data(r74_data, 7);

    lcd_write_cmd(0x98);
    uint8_t r98_data[] = {0x3E, 0x07};
    lcd_write_data(r98_data, 2);

    // Tearing effect on
    lcd_write_cmd(GC9A01_TEON);

    // Inversion on (required for correct colors)
    lcd_write_cmd(GC9A01_INVON);

    // Exit sleep
    lcd_write_cmd(GC9A01_SLPOUT);
    sleep_ms(150);

    // Display on
    lcd_write_cmd(GC9A01_DISPON);
    sleep_ms(150);
}

static void lcd_set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1) {
    uint8_t data[4];

    lcd_write_cmd(GC9A01_CASET);
    data[0] = x0 >> 8;
    data[1] = x0 & 0xFF;
    data[2] = x1 >> 8;
    data[3] = x1 & 0xFF;
    lcd_write_data(data, 4);

    lcd_write_cmd(GC9A01_RASET);
    data[0] = y0 >> 8;
    data[1] = y0 & 0xFF;
    data[2] = y1 >> 8;
    data[3] = y1 & 0xFF;
    lcd_write_data(data, 4);

    lcd_write_cmd(GC9A01_RAMWR);
}

static void lcd_fill(uint16_t color) {
    lcd_set_window(0, 0, 239, 239);

    // Prepare a line buffer (240 pixels = 480 bytes)
    uint8_t line[480];
    uint8_t hi = color >> 8;
    uint8_t lo = color & 0xFF;

    for (int i = 0; i < 480; i += 2) {
        line[i] = hi;
        line[i + 1] = lo;
    }

    lcd_cs_select();
    lcd_dc_data();

    // Send 240 lines
    for (int y = 0; y < 240; y++) {
        spi_write_blocking(LCD_SPI, line, 480);
    }

    lcd_cs_deselect();
}

int main(void) {
    stdio_init_all();

    // Initialize CYW43 for LED
    if (cyw43_arch_init()) {
        while (1) tight_loop_contents();
    }

    // Blink LED to show we're alive
    for (int i = 0; i < 5; i++) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
        sleep_ms(100);
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
        sleep_ms(100);
    }

    // Initialize LCD
    lcd_init();

    // LED on after init
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);

    // Cycle through colors
    uint16_t colors[] = {RED, GREEN, BLUE, WHITE, YELLOW, CYAN, MAGENTA, BLACK};
    int color_idx = 0;

    while (true) {
        lcd_fill(colors[color_idx]);

        // Toggle LED each color change
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, color_idx & 1);

        color_idx = (color_idx + 1) % 8;
        sleep_ms(1000);
    }

    return 0;
}
