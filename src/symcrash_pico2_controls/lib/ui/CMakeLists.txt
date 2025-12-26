# User interface library

add_library(symcrash_ui STATIC
    menu.c
    display_pages.c
)

target_include_directories(symcrash_ui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(symcrash_ui
    pico_stdlib
    symcrash_hal
    symcrash_controls_lib
    symcrash_protocol
)
