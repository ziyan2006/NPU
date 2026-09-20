/* SPDX-License-Identifier: MIT */
#include "audio_hw.h"
#include "pcm_ring.h"
#include "wav_source.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(XPAR_AUDIO_OUT_AXI_0_BASEADDR == 0x43C10000u,
               "audio base address must match the Vivado design");

typedef struct {
    uint32_t registers[64];
    uint32_t write_offsets[4];
    uint32_t write_values[4];
    size_t write_count;
} mmio_log_t;

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t position;
    size_t maximum_chunk;
} memory_reader_t;

static uint32_t recorded_read(uintptr_t address, void *context)
{
    mmio_log_t *log = (mmio_log_t *)context;
    return log->registers[(address & 0xfffu) / sizeof(uint32_t)];
}

static void recorded_write(uintptr_t address, uint32_t value, void *context)
{
    mmio_log_t *log = (mmio_log_t *)context;
    assert(log->write_count < 4u);
    log->write_offsets[log->write_count] = (uint32_t)(address & 0xfffu);
    log->write_values[log->write_count] = value;
    ++log->write_count;
}

static size_t memory_read(void *context, void *destination, size_t bytes)
{
    memory_reader_t *reader = (memory_reader_t *)context;
    size_t remaining = reader->size - reader->position;
    size_t count = bytes;

    if (count > reader->maximum_chunk)
        count = reader->maximum_chunk;
    if (count > remaining)
        count = remaining;
    memcpy(destination, reader->data + reader->position, count);
    reader->position += count;
    return count;
}

static void put_u16(uint8_t *data, size_t *position, uint16_t value)
{
    data[(*position)++] = (uint8_t)value;
    data[(*position)++] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t *data, size_t *position, uint32_t value)
{
    data[(*position)++] = (uint8_t)value;
    data[(*position)++] = (uint8_t)(value >> 8);
    data[(*position)++] = (uint8_t)(value >> 16);
    data[(*position)++] = (uint8_t)(value >> 24);
}

static size_t make_wav(uint8_t *data, uint16_t channels,
                       uint32_t sample_rate, uint16_t bits_per_sample)
{
    static const int16_t samples[] = {100, -100, 200, -200};
    const uint16_t block_align = (uint16_t)(channels * bits_per_sample / 8u);
    const uint32_t data_bytes = (uint32_t)sizeof(samples);
    size_t position = 0u;

    memcpy(data + position, "RIFF", 4u); position += 4u;
    put_u32(data, &position, 0u);
    memcpy(data + position, "WAVE", 4u); position += 4u;
    memcpy(data + position, "JUNK", 4u); position += 4u;
    put_u32(data, &position, 3u);
    data[position++] = 0x11u;
    data[position++] = 0x22u;
    data[position++] = 0x33u;
    data[position++] = 0u;
    memcpy(data + position, "fmt ", 4u); position += 4u;
    put_u32(data, &position, 16u);
    put_u16(data, &position, 1u);
    put_u16(data, &position, channels);
    put_u32(data, &position, sample_rate);
    put_u32(data, &position, sample_rate * block_align);
    put_u16(data, &position, block_align);
    put_u16(data, &position, bits_per_sample);
    memcpy(data + position, "data", 4u); position += 4u;
    put_u32(data, &position, data_bytes);
    memcpy(data + position, samples, data_bytes); position += data_bytes;
    data[4] = (uint8_t)(position - 8u);
    data[5] = (uint8_t)((position - 8u) >> 8);
    data[6] = (uint8_t)((position - 8u) >> 16);
    data[7] = (uint8_t)((position - 8u) >> 24);
    return position;
}

static void test_audio_hw(void)
{
    uint32_t direct_registers[64] = {0};
    mmio_log_t log;
    audio_hw_t hardware;
    audio_hw_io_t io;
    audio_frame_t frame = {0x1234, -2, 0x3456, -4};

    direct_registers[AUDIO_HW_REG_IP_ID / 4u] = AUDIO_HW_IP_ID;
    direct_registers[AUDIO_HW_REG_FIFO_CAPACITY / 4u] =
        AUDIO_HW_FIFO_CAPACITY;
    assert(audio_hw_init(&hardware, (uintptr_t)direct_registers)
           == AUDIO_HW_OK);
    direct_registers[AUDIO_HW_REG_IP_ID / 4u] = 0u;
    assert(audio_hw_init(&hardware, (uintptr_t)direct_registers)
           == AUDIO_HW_E_INCOMPATIBLE);
    assert(audio_hw_init(NULL, (uintptr_t)direct_registers)
           == AUDIO_HW_E_INVALID);

    memset(&log, 0, sizeof(log));
    log.registers[AUDIO_HW_REG_IP_ID / 4u] = AUDIO_HW_IP_ID;
    log.registers[AUDIO_HW_REG_FIFO_CAPACITY / 4u] =
        AUDIO_HW_FIFO_CAPACITY;
    log.registers[AUDIO_HW_REG_FIFO_LEVEL / 4u] = 17u;
    io.read32 = recorded_read;
    io.write32 = recorded_write;
    io.context = &log;
    assert(audio_hw_init_with_io(&hardware, 0x43c10000u, &io)
           == AUDIO_HW_OK);
    assert(audio_hw_space(&hardware) == AUDIO_HW_FIFO_CAPACITY - 17u);
    assert(audio_hw_write_frame(&hardware, &frame) == AUDIO_HW_OK);
    assert(log.write_count == 2u);
    assert(log.write_offsets[0] == AUDIO_HW_REG_MIX_FRAME);
    assert(log.write_offsets[1] == AUDIO_HW_REG_VOCAL_FRAME);
    assert(log.write_values[0] == 0xfffe1234u);
    assert(log.write_values[1] == 0xfffc3456u);
    assert(hardware.submitted_frames == 1u);

    log.registers[AUDIO_HW_REG_FIFO_LEVEL / 4u] = AUDIO_HW_FIFO_CAPACITY;
    assert(audio_hw_write_frame(&hardware, &frame) == AUDIO_HW_E_FULL);
    assert(log.write_count == 2u);
    assert(hardware.submitted_frames == 1u);
}

static void test_pcm_ring(void)
{
    pcm_stereo_frame_t storage[8];
    pcm_stereo_frame_t first[6] = {
        {0, 100}, {1, 101}, {2, 102},
        {3, 103}, {4, 104}, {5, 105},
    };
    pcm_stereo_frame_t second[6] = {
        {10, 110}, {11, 111}, {12, 112},
        {13, 113}, {14, 114}, {15, 115},
    };
    pcm_stereo_frame_t output[8];
    pcm_ring_t ring;

    assert(pcm_ring_init(&ring, storage, 6u) == PCM_RING_E_INVALID);
    assert(pcm_ring_init(&ring, storage, 8u) == PCM_RING_OK);
    assert(pcm_ring_push(&ring, first, 6u) == 6u);
    assert(pcm_ring_size(&ring) == 6u && pcm_ring_space(&ring) == 2u);
    assert(pcm_ring_peek(&ring, output, 4u) == 4u);
    assert(memcmp(output, first, 4u * sizeof(*output)) == 0);
    assert(pcm_ring_drop(&ring, 4u) == 4u);
    assert(pcm_ring_push(&ring, second, 6u) == 6u);
    assert(pcm_ring_push(&ring, second, 1u) == 0u);
    assert(pcm_ring_peek(&ring, output, 8u) == 8u);
    assert(memcmp(output, first + 4, 2u * sizeof(*output)) == 0);
    assert(memcmp(output + 2, second, 6u * sizeof(*output)) == 0);
    assert(pcm_ring_drop(&ring, 20u) == 8u);
    assert(pcm_ring_size(&ring) == 0u);
}

static void test_wav_source(void)
{
    uint8_t wav[128];
    int16_t samples[6] = {0};
    memory_reader_t reader;
    wav_source_t source;
    size_t size;

    size = make_wav(wav, 2u, 44100u, 16u);
    reader = (memory_reader_t){wav, size, 0u, 3u};
    assert(wav_source_open(&source, memory_read, &reader) == WAV_SOURCE_OK);
    assert(source.total_frames == 2u);
    assert(wav_source_read(&source, samples, 3u) == 2u);
    assert(samples[0] == 100 && samples[1] == -100);
    assert(samples[2] == 200 && samples[3] == -200);
    assert(wav_source_read(&source, samples, 1u) == 0u);

    size = make_wav(wav, 1u, 44100u, 16u);
    reader = (memory_reader_t){wav, size, 0u, sizeof(wav)};
    assert(wav_source_open(&source, memory_read, &reader)
           == WAV_SOURCE_E_UNSUPPORTED);
    size = make_wav(wav, 2u, 48000u, 16u);
    reader = (memory_reader_t){wav, size, 0u, sizeof(wav)};
    assert(wav_source_open(&source, memory_read, &reader)
           == WAV_SOURCE_E_UNSUPPORTED);
    size = make_wav(wav, 2u, 44100u, 24u);
    reader = (memory_reader_t){wav, size, 0u, sizeof(wav)};
    assert(wav_source_open(&source, memory_read, &reader)
           == WAV_SOURCE_E_UNSUPPORTED);
}

int main(void)
{
    test_audio_hw();
    test_pcm_ring();
    test_wav_source();
    puts("audio foundation: PASS");
    return 0;
}
