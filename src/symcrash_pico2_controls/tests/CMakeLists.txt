# Test executables for Symcrash Pico2 Controls

# GC9A01 Hello World Test
add_executable(gc9a01_hello
    gc9a01_hello.c
)

target_include_directories(gc9a01_hello PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib/hal
)

target_link_libraries(gc9a01_hello
    pico_stdlib
    pico_cyw43_arch_none
    hardware_spi
    hardware_gpio
    symcrash_hal
)

# Disable stdio - standalone test
pico_enable_stdio_usb(gc9a01_hello 0)
pico_enable_stdio_uart(gc9a01_hello 0)

# Create UF2
pico_add_extra_outputs(gc9a01_hello)

# ============================================================================
# GC9A01 Simple Test - minimal, self-contained
# ============================================================================
add_executable(gc9a01_simple
    gc9a01_simple.c
)

target_link_libraries(gc9a01_simple
    pico_stdlib
    pico_cyw43_arch_none
    hardware_spi
    hardware_gpio
)

pico_enable_stdio_usb(gc9a01_simple 0)
pico_enable_stdio_uart(gc9a01_simple 0)

pico_add_extra_outputs(gc9a01_simple)

# ============================================================================
# Pot Display Test - MCP3008 values on GC9A01 LCD
# ============================================================================
add_executable(pot_display
    pot_display.c
)

target_link_libraries(pot_display
    pico_stdlib
    pico_cyw43_arch_none
    hardware_spi
    hardware_gpio
)

pico_enable_stdio_usb(pot_display 0)
pico_enable_stdio_uart(pot_display 0)

pico_add_extra_outputs(pot_display)
