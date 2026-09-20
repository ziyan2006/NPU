/* SPDX-License-Identifier: MIT */
#ifndef STEM_AUDIO_HW_H
#define STEM_AUDIO_HW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_HW_BASE_ADDRESS       0x43c10000u
#define AUDIO_HW_IP_ID              0x31445541u
#define AUDIO_HW_FIFO_CAPACITY      16384u

#define AUDIO_HW_REG_IP_ID          0x000u
#define AUDIO_HW_REG_CONTROL        0x008u
#define AUDIO_HW_REG_STATUS         0x00cu
#define AUDIO_HW_REG_MIX_FRAME      0x010u
#define AUDIO_HW_REG_VOCAL_FRAME    0x014u
#define AUDIO_HW_REG_FIFO_LEVEL     0x018u
#define AUDIO_HW_REG_FIFO_CAPACITY  0x01cu
#define AUDIO_HW_REG_UNDERFLOW_COUNT 0x020u
#define AUDIO_HW_REG_OVERFLOW_COUNT  0x024u
#define AUDIO_HW_REG_PLAYED_FRAMES   0x028u
#define AUDIO_HW_REG_STEM_STATE       0x02cu
#define AUDIO_HW_REG_CODEC_STATUS    0x030u
#define AUDIO_HW_REG_TONE_CONTROL    0x034u

#define AUDIO_HW_CONTROL_ENABLE      (1u << 0)
#define AUDIO_HW_CONTROL_SOFT_RESET  (1u << 1)
#define AUDIO_HW_CONTROL_TONE_ENABLE (1u << 3)

#define AUDIO_HW_STATUS_CODEC_DONE   (1u << 2)
#define AUDIO_HW_STATUS_CODEC_ERROR  (1u << 3)
#define AUDIO_HW_STEM_TARGET          (1u << 0)
#define AUDIO_HW_STEM_RAMPING         (1u << 1)

#if defined(XPAR_AUDIO_OUT_AXI_0_BASEADDR)
_Static_assert(XPAR_AUDIO_OUT_AXI_0_BASEADDR == AUDIO_HW_BASE_ADDRESS,
               "Vivado audio base address differs from the software contract");
#endif

typedef enum {
    AUDIO_HW_OK = 0,
    AUDIO_HW_E_INVALID = -1,
    AUDIO_HW_E_INCOMPATIBLE = -2,
    AUDIO_HW_E_FULL = -3
} audio_hw_result_t;

typedef struct {
    int16_t mix_left;
    int16_t mix_right;
    int16_t vocal_left;
    int16_t vocal_right;
} audio_frame_t;

typedef uint32_t (*audio_hw_read32_fn)(uintptr_t address, void *context);
typedef void (*audio_hw_write32_fn)(uintptr_t address, uint32_t value,
                                    void *context);

typedef struct {
    audio_hw_read32_fn read32;
    audio_hw_write32_fn write32;
    void *context;
} audio_hw_io_t;

typedef struct {
    uintptr_t base_address;
    audio_hw_io_t io;
    uint32_t submitted_frames;
} audio_hw_t;

audio_hw_result_t audio_hw_init(audio_hw_t *hardware, uintptr_t base_address);
audio_hw_result_t audio_hw_init_with_io(audio_hw_t *hardware,
                                        uintptr_t base_address,
                                        const audio_hw_io_t *io);
size_t audio_hw_space(const audio_hw_t *hardware);
audio_hw_result_t audio_hw_write_frame(audio_hw_t *hardware,
                                       const audio_frame_t *frame);

#ifdef __cplusplus
}
#endif

#endif /* STEM_AUDIO_HW_H */
