# Symcrash Pico2 Hardware UI Controller
# CMake build configuration for Raspberry Pi Pico2 (RP2350)

cmake_minimum_required(VERSION 3.13)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================

# Set board to Pico 2 (RP2350)
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Include Pico SDK (must be before project())
include(pico_sdk_import.cmake)

# Project definition
project(symcrash_controls C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize Pico SDK
pico_sdk_init()

# Add subdirectories for libraries
add_subdirectory(lib/protocol)
add_subdirectory(lib/hal)
add_subdirectory(lib/controls)
add_subdirectory(lib/ui)

# Main executable
add_executable(symcrash_controls
    src/main.c
    src/init.c
    src/event_loop.c
)

# Include directories
target_include_directories(symcrash_controls PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(symcrash_controls
    pico_stdlib
    symcrash_protocol
    symcrash_hal
    symcrash_controls_lib
    symcrash_ui
)

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

# Enable USB output, disable UART for stdio (using UARTs for protocol)
pico_enable_stdio_usb(symcrash_controls 1)
pico_enable_stdio_uart(symcrash_controls 0)

# Create UF2 file for flashing
pico_add_extra_outputs(symcrash_controls)
