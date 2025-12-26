# Control surface logic library

add_library(symcrash_controls_lib STATIC
    control_map.c
    knob.c
    button.c
)

target_include_directories(symcrash_controls_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(symcrash_controls_lib
    pico_stdlib
    symcrash_hal
    symcrash_protocol
)
