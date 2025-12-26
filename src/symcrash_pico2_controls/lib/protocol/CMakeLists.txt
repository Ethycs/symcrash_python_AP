# Protocol library - UART communication with synthesis engine

add_library(symcrash_protocol STATIC
    cobs.c
    crc32c.c
    frame.c
    messages.c
    ctrl_uart.c
    data_uart.c
    dual_uart_link.c
)

target_include_directories(symcrash_protocol PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Base libraries
target_link_libraries(symcrash_protocol
    pico_stdlib
    hardware_uart
    hardware_dma
)

# USB CDC support - TRANSPORT_USB_MODE is defined in config.h
# TinyUSB is needed for USB CDC mode
target_link_libraries(symcrash_protocol
    tinyusb_device
    tinyusb_board
)
