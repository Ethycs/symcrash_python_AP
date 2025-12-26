# Symcrash Pico2 Hardware UI Controller

Raspberry Pi Pico2 (RP2350) firmware for a hardware control surface that interfaces with the symcrash Python audio synthesis engine.

## Hardware Requirements

- **MCU**: Raspberry Pi Pico2 (RP2350)
- **ADC**: MCP3008 (8-channel SPI ADC)
- **Potentiometers**: 6x linear pots on MCP3008 channels 0-5
- **Buttons**: 4x momentary switches (active low)
- **Encoder**: 1x rotary encoder with switch
- **Displays**: 2x GC9A01 240x240 round LCD

## Pin Assignments

| Function | Pin | GPIO |
|----------|-----|------|
| C-UART TX | 1 | GP0 |
| C-UART RX | 2 | GP1 |
| D-UART TX | 11 | GP8 |
| D-UART RX | 12 | GP9 |
| Encoder A | 4 | GP2 |
| Encoder B | 5 | GP3 |
| Encoder SW | 6 | GP4 |
| Button 0 | 7 | GP5 |
| Button 1 | 9 | GP6 |
| Button 2 | 10 | GP7 |
| Button 3 | 20 | GP15 |
| LCD SPI SCK | 14 | GP10 |
| LCD SPI TX | 15 | GP11 |
| LCD0 CS | 16 | GP12 |
| LCD0 DC | 17 | GP13 |
| LCD0 RST | 19 | GP14 |
| LCD1 CS | 26 | GP20 |
| LCD1 DC | 27 | GP21 |
| LCD1 RST | 29 | GP22 |
| MCP3008 MISO | 21 | GP16 |
| MCP3008 CS | 22 | GP17 |
| MCP3008 SCK | 24 | GP18 |
| MCP3008 MOSI | 25 | GP19 |
| Status LED | 25 | GP25 |

## Building

1. Set up Pico SDK:
   ```bash
   export PICO_SDK_PATH=/path/to/pico-sdk
   ```

2. Build:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. Flash to Pico2:
   - Hold BOOTSEL while connecting USB
   - Copy `symcrash_controls.uf2` to the mounted drive

## Protocol

Uses WSE-STD-100 Dual-UART Link protocol:

- **C-UART** (UART0 @ 1 Mbps): Reliable control messages with ACK
- **D-UART** (UART1 @ 3 Mbps): High-throughput streaming data

### Control Messages

| Message | Size | Description |
|---------|------|-------------|
| NoteEvent | 9 bytes | Note on/off with velocity |
| SetMacro | 8 bytes | Knob value change (Q15) |
| SetTreeTarget | 8 bytes | Preset/tree morphing |

### Frame Format

```
Magic (0xDEC0) | Version | Channel | MsgType | Flags | Seq | Length | CRC32
```

All frames are COBS-encoded with 0x00 delimiter.

## Architecture

```
main.c
  └── system_init() (init.c)
        ├── HAL: mcp3008, buttons, encoder, gc9a01
        ├── Protocol: dual_uart_link, ctrl_uart, data_uart
        ├── Controls: control_map
        └── UI: menu, display_pages

  └── event_loop_run() (event_loop.c)
        ├── Timer: ADC scan @ 1 kHz
        ├── Timer: Button/encoder @ 200 Hz
        ├── Timer: Display @ 30 Hz
        └── Loop: UART processing (continuous)
```

## License

Part of the symcrash project.
