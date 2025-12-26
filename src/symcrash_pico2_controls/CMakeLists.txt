# Symcrash Pico2 Hardware UI Controller
# CMake build configuration for Raspberry Pi Pico2 (RP2350)

cmake_minimum_required(VERSION 3.13)

# Include Pico SDK (must be before project())
include(pico_sdk_import.cmake)

project(symcrash_pico2_controls C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize the Pico SDK
pico_sdk_init()

# =============================================================================
# Library subdirectories
# =============================================================================

add_subdirectory(lib/protocol)
add_subdirectory(lib/hal)
add_subdirectory(lib/controls)
add_subdirectory(lib/ui)

# =============================================================================
# Main application
# =============================================================================

add_executable(symcrash_controls
    src/main.c
    src/init.c
    src/event_loop.c
)

target_include_directories(symcrash_controls PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(symcrash_controls
    pico_stdlib
    hardware_spi
    hardware_uart
    hardware_dma
    hardware_pio
    hardware_timer
    hardware_gpio
    symcrash_protocol
    symcrash_hal
    symcrash_controls_lib
    symcrash_ui
)

# Enable USB output for debug (stdio over USB)
pico_enable_stdio_usb(symcrash_controls 1)
pico_enable_stdio_uart(symcrash_controls 0)

# Create UF2 file for flashing
pico_add_extra_outputs(symcrash_controls)

# =============================================================================
# Build configuration
# =============================================================================

# Add compiler warnings
target_compile_options(symcrash_controls PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# Debug build options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(symcrash_controls PRIVATE DEBUG=1)
endif()
