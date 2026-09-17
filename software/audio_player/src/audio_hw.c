/* SPDX-License-Identifier: MIT */
#include "audio_hw.h"

static uint32_t direct_read32(uintptr_t address, void *context)
{
    (void)context;
    return *(volatile const uint32_t *)address;
}

static void direct_write32(uintptr_t address, uint32_t value, void *context)
{
    (void)context;
    *(volatile uint32_t *)address = value;
}

static uint32_t audio_read(const audio_hw_t *hardware, uint32_t offset)
{
    return hardware->io.read32(hardware->base_address + offset,
                               hardware->io.context);
}

static void audio_write(audio_hw_t *hardware, uint32_t offset, uint32_t value)
{
    hardware->io.write32(hardware->base_address + offset, value,
                         hardware->io.context);
}

static uint32_t pack_stereo(int16_t left, int16_t right)
{
    return (uint32_t)(uint16_t)left | ((uint32_t)(uint16_t)right << 16);
}

audio_hw_result_t audio_hw_init_with_io(audio_hw_t *hardware,
                                        uintptr_t base_address,
                                        const audio_hw_io_t *io)
{
    if (hardware == NULL || base_address == (uintptr_t)0 || io == NULL
        || io->read32 == NULL || io->write32 == NULL)
        return AUDIO_HW_E_INVALID;

    hardware->base_address = base_address;
    hardware->io = *io;
    hardware->submitted_frames = 0u;
    if (audio_read(hardware, AUDIO_HW_REG_IP_ID) != AUDIO_HW_IP_ID
        || audio_read(hardware, AUDIO_HW_REG_FIFO_CAPACITY)
           != AUDIO_HW_FIFO_CAPACITY)
        return AUDIO_HW_E_INCOMPATIBLE;
    return AUDIO_HW_OK;
}

audio_hw_result_t audio_hw_init(audio_hw_t *hardware, uintptr_t base_address)
{
    const audio_hw_io_t io = {direct_read32, direct_write32, NULL};
    return audio_hw_init_with_io(hardware, base_address, &io);
}

size_t audio_hw_space(const audio_hw_t *hardware)
{
    uint32_t level;
    uint32_t capacity;

    if (hardware == NULL || hardware->io.read32 == NULL)
        return 0u;
    level = audio_read(hardware, AUDIO_HW_REG_FIFO_LEVEL);
    capacity = audio_read(hardware, AUDIO_HW_REG_FIFO_CAPACITY);
    return level < capacity ? (size_t)(capacity - level) : 0u;
}

audio_hw_result_t audio_hw_write_frame(audio_hw_t *hardware,
                                       const audio_frame_t *frame)
{
    if (hardware == NULL || frame == NULL || hardware->io.write32 == NULL)
        return AUDIO_HW_E_INVALID;
    if (audio_hw_space(hardware) == 0u)
        return AUDIO_HW_E_FULL;

    audio_write(hardware, AUDIO_HW_REG_MIX_FRAME,
                pack_stereo(frame->mix_left, frame->mix_right));
    audio_write(hardware, AUDIO_HW_REG_VOCAL_FRAME,
                pack_stereo(frame->vocal_left, frame->vocal_right));
    ++hardware->submitted_frames;
    return AUDIO_HW_OK;
}
