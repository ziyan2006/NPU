"""Executable primitives for the draft STEM NPU v1 ISA.

This module is deliberately dependency-light.  It is shared by the offline
compiler, structural tests, and the future instruction interpreter so the
binary layout and fixed-point rules have one source of truth.
"""
from __future__ import annotations

import enum
import math
import struct
from dataclasses import dataclass


ISA_MAJOR = 1
ISA_MINOR = 0
NONE_INDEX = 0xFFFF
COMMAND_STRUCT = struct.Struct("<BB7H")
TENSOR_STRUCT = struct.Struct("<QI4H4IBBHHHI16x")
OPERATOR_STRUCT = struct.Struct("<16H8I")
QUANT_DESC_STRUCT = struct.Struct("<IIiiHH12x")
QUANT_PARAM_STRUCT = struct.Struct("<iB3xii")
SEGMENT_STRUCT = struct.Struct("<4H")

TENSOR_FLAG_CONSTANT = 1 << 0
TENSOR_FLAG_EXTERNAL = 1 << 1
TENSOR_FLAG_VIEW = 1 << 2

IMM_EVENT_LSB = 0
IMM_EVENT_WIDTH = 8
IMM_ACTIVATION_BANK_BIT = 8
IMM_WEIGHT_BANK_BIT = 9
IMM_ACCUMULATOR_BANK_BIT = 10
# DMA_LOAD never addresses the accumulator bank, so bit 10 is reused there to
# select the scalar/vector quant slot instead of the convolution quant slot.
IMM_DMA_VECTOR_QUANT_BIT = IMM_ACCUMULATOR_BANK_BIT
IMM_OUTPUT_BANK_BIT = 11
IMM_SEGMENT_LSB = 12
IMM_SEGMENT_WIDTH = 2
IMM_SEGMENTED_BIT = 14


class Opcode(enum.IntEnum):
    NOP = 0x00
    WAIT = 0x01
    SIGNAL = 0x02
    END = 0x03
    DMA_LOAD = 0x10
    DMA_STORE = 0x11
    DMA_COPY2D = 0x12
    CONV2D = 0x20
    VEC_ADD = 0x30
    VEC_SUB = 0x31
    VEC_MUL = 0x32
    VEC_MINMAX = 0x33
    ACT = 0x40
    REQUANT = 0x41
    UPSAMPLE2X = 0x42
    POOL2D = 0x43
    COPY_LAYOUT = 0x44


class CommandFlag(enum.IntFlag):
    NONE = 0
    ASYNC = 1 << 0
    IRQ = 1 << 1
    SATURATE = 1 << 2
    FUSED_POST_OP = 1 << 3


class Event(enum.IntFlag):
    A0_READY = 1 << 0
    A1_READY = 1 << 1
    W0_READY = 1 << 2
    W1_READY = 1 << 3
    C0_DONE = 1 << 4
    C1_DONE = 1 << 5
    S0_DONE = 1 << 6
    S1_DONE = 1 << 7


class ErrorCode(enum.IntEnum):
    NONE = 0x0000
    ILLEGAL_OPCODE = 0x0001
    ILLEGAL_FLAGS = 0x0002
    ILLEGAL_FIELDS = 0x0003
    EXECUTION_ERROR = 0x0004
    WATCHDOG = 0x0005


class DType(enum.IntEnum):
    INT8 = 1
    INT12_IN_INT16 = 2
    INT16 = 3
    INT32 = 4
    UINT8 = 5


class Layout(enum.IntEnum):
    LINEAR = 0
    NHWC8 = 1
    WEIGHT_O8I8 = 2
    SEGMENTED = 3


class PostOp(enum.IntEnum):
    NONE = 0
    RELU = 1
    LEAKY_RELU_0P1 = 2
    TANH_LUT = 3


@dataclass(frozen=True)
class Command:
    opcode: Opcode
    flags: int = 0
    tag: int = 0
    dst_td: int = NONE_INDEX
    src0_td: int = NONE_INDEX
    src1_td: int = NONE_INDEX
    op_desc: int = NONE_INDEX
    quant_desc: int = NONE_INDEX
    imm: int = 0

    def pack(self) -> bytes:
        fields = (
            int(self.opcode), self.flags, self.tag, self.dst_td, self.src0_td,
            self.src1_td, self.op_desc, self.quant_desc, self.imm,
        )
        if not all(0 <= value <= (0xFF if i < 2 else 0xFFFF)
                   for i, value in enumerate(fields)):
            raise ValueError(f"command field out of range: {fields}")
        return COMMAND_STRUCT.pack(*fields)

    @classmethod
    def unpack(cls, payload: bytes) -> "Command":
        if len(payload) != COMMAND_STRUCT.size:
            raise ValueError(f"command must be {COMMAND_STRUCT.size} bytes")
        opcode, flags, *words = COMMAND_STRUCT.unpack(payload)
        return cls(Opcode(opcode), flags, *words)


def encode_control_imm(*, event: int = 0, activation: int = 0,
                       weight: int = 0, accumulator: int = 0,
                       output: int = 0, segmented: bool = False,
                       segment: int = 0,
                       dma_vector_quant: bool = False) -> int:
    """Pack the P3 proposed event and scratchpad routing immediate."""
    if event & ~0xFF or not 0 <= segment < (1 << IMM_SEGMENT_WIDTH):
        raise ValueError("event or segment does not fit the proposed immediate")
    if dma_vector_quant and accumulator:
        raise ValueError("DMA vector-quant and accumulator select share bit 10")
    auxiliary = int(bool(dma_vector_quant)) if dma_vector_quant else (accumulator & 1)
    return (event | ((activation & 1) << IMM_ACTIVATION_BANK_BIT)
            | ((weight & 1) << IMM_WEIGHT_BANK_BIT)
            | (auxiliary << IMM_ACCUMULATOR_BANK_BIT)
            | ((output & 1) << IMM_OUTPUT_BANK_BIT)
            | ((segment & 3) << IMM_SEGMENT_LSB)
            | (int(segmented) << IMM_SEGMENTED_BIT))


def round_shift_rne(value: int, shift: int) -> int:
    """Signed divide by 2**shift using round-to-nearest, ties-to-even.

    Python's arbitrary-width integers make this the normative definition for
    RTL corner cases, including negative halfway values.  A negative shift is
    an exact left shift.
    """
    if shift < 0:
        return int(value) << -shift
    if shift == 0:
        return int(value)
    sign = -1 if value < 0 else 1
    magnitude = abs(int(value))
    quotient, remainder = divmod(magnitude, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return sign * quotient


def saturate(value: int, bits: int, *, signed: bool = True) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    if signed:
        low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        low, high = 0, (1 << bits) - 1
    return min(max(int(value), low), high)


def quantize_scale(real_scale: float) -> tuple[int, int]:
    """Approximate a positive real scale as multiplier / 2**shift.

    The multiplier is a positive signed-Q31 value.  Keeping it below 2**31
    makes the representation directly usable by a signed 32-bit multiplier.
    """
    if not math.isfinite(real_scale) or real_scale < 0:
        raise ValueError(f"scale must be finite and non-negative: {real_scale}")
    if real_scale == 0:
        return 0, 0
    mantissa, exponent = math.frexp(real_scale)
    multiplier = int(round(mantissa * (1 << 31)))
    if multiplier == 1 << 31:
        multiplier >>= 1
        exponent += 1
    shift = 31 - exponent
    if not 0 < multiplier <= 0x7FFFFFFF or not 0 <= shift <= 63:
        raise OverflowError(
            f"scale {real_scale} is outside Q31/shift representation")
    return multiplier, shift


def dequantize_scale(multiplier: int, shift: int) -> float:
    return float(multiplier) / float(1 << shift)


def requantize(value: int, multiplier: int, shift: int,
               clamp_min: int, clamp_max: int) -> int:
    if clamp_min > clamp_max:
        raise ValueError("invalid clamp interval")
    scaled = round_shift_rne(int(value) * int(multiplier), int(shift))
    return min(max(scaled, int(clamp_min)), int(clamp_max))


def leaky_relu_q(value: int) -> int:
    """LeakyReLU with exact draft slope 0.1 and RNE integer output."""
    return int(value) if value >= 0 else _round_ratio_rne(int(value), 1, 10)


def _round_ratio_rne(value: int, numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ratio must be non-negative with a positive denominator")
    sign = -1 if value < 0 else 1
    total = abs(int(value)) * numerator
    quotient, remainder = divmod(total, denominator)
    twice = remainder * 2
    if twice > denominator or (twice == denominator and quotient & 1):
        quotient += 1
    return sign * quotient


def align_up(value: int, alignment: int = 64) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (int(value) + alignment - 1) & -alignment


def nhwc8_nbytes(shape_chw: list[int] | tuple[int, int, int],
                 element_bytes: int = 2) -> int:
    channels, height, width = map(int, shape_chw)
    padded_channels = align_up(channels, 8)
    return height * width * padded_channels * element_bytes
