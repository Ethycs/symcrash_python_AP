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

target_link_libraries(symcrash_protocol
    pico_stdlib
    hardware_uart
    hardware_dma
)
