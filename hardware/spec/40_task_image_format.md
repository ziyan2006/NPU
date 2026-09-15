# NPU task image format

Document version: `1.0`

The host submits one contiguous, 64-byte aligned task image.  `TASK_BASE`,
`TASK_BYTES`, and `TASK_TAG` are therefore sufficient to launch work; internal
section addresses are not part of the software-visible register ABI.

## Header

The fixed 256-byte header is little endian.

| Offset | Field | Meaning |
|---:|---|---|
| `0x00` | `magic[8]` | `STEMNPU\0` |
| `0x08` | `header_bytes`, `total_bytes` | Header and complete image size |
| `0x10` | format major/minor, ISA major/minor | Compatibility tuple |
| `0x18` | `flags`, `command_count` | Flag bit 0 means build-image CRC is present |
| `0x20` | command offset/bytes | Tile-level 128-bit command stream |
| `0x28` | tensor descriptor offset/bytes | 64-byte descriptors |
| `0x30` | operator descriptor offset/bytes | 64-byte tile descriptors |
| `0x38` | quant descriptor offset/bytes | 32-byte descriptors |
| `0x40` | segment offset/bytes | 8-byte concat view records |
| `0x48` | activation offset/bytes | Mutable, zero-initialized arena |
| `0x50` | weight offset/bytes | O8I8 packed INT8 weights |
| `0x58` | bias offset/bytes | Packed INT32 bias |
| `0x60` | quant parameter offset/bytes | Q31 multiplier/shift records |
| `0x68` | tanh LUT offset/bytes | 4096 signed INT12 entries in INT16 storage |
| `0x70` | input/output tensor index | Host entry points |
| `0x78` | payload/header CRC32 | IEEE CRC-32 |
| `0x80` | source manifest SHA-256 | Model provenance |
| `0xA0` | reserved | Must be zero |

Every section starts on a 64-byte boundary.  An offset/size pair must lie inside
`total_bytes`; the command section must contain an integral number of 16-byte
commands and agree with `command_count`.  Unknown incompatible major versions
are rejected.  Minor versions are backward compatible only when unsupported
flags remain zero.

The payload CRC covers the pristine bytes from `header_bytes` through
`total_bytes`.  Host software verifies it before writing the input tensor into
the mutable activation arena.  The header CRC is calculated over all 256 header
bytes with the `header_crc32` field cleared.  RTL validates structural fields;
full-image CRC and SHA-256 validation belong to the driver because the activation
arena changes before execution.

`scripts/33_build_npu_task_image.py` is the normative builder.  It emits
`task_image.bin` and a readable `task_image.json` section map.
