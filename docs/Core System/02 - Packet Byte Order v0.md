## VoiceFrame Packet v0b (136 bytes)

```
Offset  Size  Field
------  ----  ------------------------------------------------------------
0x0000  2     magic u16                 = 0x5646  ("VF")
0x0002  1     version u8                = 1
0x0003  1     packet_len u8             = 136
0x0004  2     program_id u16
0x0006  2     voice_id u16
0x0008  4     frame_seq u32             (monotonic per voice)
0x000C  2     reject_code u16           (0=accepted, else reject reason)
0x000E  2     hdr_pad u16               (MUST=0)

0x0010  2     flags u16                 (gate/gate_rise/gate_fall/etc)
0x0012  1     regime_id u8
0x0013  1     degradation_level u8
0x0014  1     note_u8 u8                (0..127)
0x0015  1     pad0 u8                   (MUST=0)
0x0016  2     cents_i16 i16

0x0018  2     gain_q15 u16              (Q0.16 or UQ15 by convention)
0x001A  2     activity_q15 u16
0x001C  2     pan_q15 i16               (Q1.15 bipolar)
0x001E  2     sendA_q15 u16
0x0020  2     sendB_q15 u16

0x0022  16    w_family_q15[8] u16[8]    (8 families, fixed order)

0x0032  1     tree_a_id u8
0x0033  1     tree_b_id u8
0x0034  2     tree_blend_q15 u16        ($\beta$)

0x0036  1     event_mask u8
0x0037  1     pad1 u8                   (MUST=0)
0x0038  16    event_strength_q15[8] u16[8]

0x0048  64    event_payload_u64[8] u64[8]
              lane $i$ payload is valid iff event_mask bit $i = 1$
              if bit $i = 0$ $\Rightarrow$ strength$[i]=0$ AND payload$[i]=0$ (strict)

0x0088  ----  END (0x0088 = 136 bytes)
```

---

### Event payload u64 bit layout (per lane, big-endian integer)

For $event\_payload\_u64[i]$ when lane $i$ active:

- **63..60** lane_id (MUST equal $i$)
- **59..56** variant
- **55..48** strength_q8 (reserved=0; use $event\_strength\_q15[i]$)
- **47..40** target_kind
- **39..32** target_id
- **31..24** pattern_id
- **23..16** param0_q8
- **15..8** param1_q8
- **7..0** param2_q8

---

If you want, I can also emit this same diagram as a machine-readable schema (tiny IDL) so you can auto-generate C/Python/Verilog pack/unpack + validators directly from the canonical offsets.