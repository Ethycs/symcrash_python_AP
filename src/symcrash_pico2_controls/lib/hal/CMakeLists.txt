# Hardware Abstraction Layer library

add_library(symcrash_hal STATIC
    mcp3008.c
    gpio_buttons.c
    rotary_encoder.c
    gc9a01.c
)

# Note: rotary_encoder.pio is available for future PIO-based implementation
# Currently using GPIO polling in rotary_encoder.c for simplicity

target_include_directories(symcrash_hal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(symcrash_hal
    pico_stdlib
    hardware_spi
    hardware_gpio
    hardware_dma
    hardware_timer
)
