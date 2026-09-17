/* SPDX-License-Identifier: MIT */
#include "wav_source.h"

#include <string.h>

#define WAV_FRAME_BYTES 4u

static uint16_t read_u16(const uint8_t *data)
{
    return (uint16_t)((uint16_t)data[0] | ((uint16_t)data[1] << 8));
}

static uint32_t read_u32(const uint8_t *data)
{
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8)
        | ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static int read_exact(wav_source_t *source, void *destination, size_t bytes)
{
    uint8_t *cursor = (uint8_t *)destination;
    size_t count;

    while (bytes != 0u) {
        count = source->read(source->context, cursor, bytes);
        if (count == 0u || count > bytes)
            return 0;
        cursor += count;
        bytes -= count;
    }
    return 1;
}

static int skip_exact(wav_source_t *source, uint32_t bytes)
{
    uint8_t scratch[32];

    while (bytes != 0u) {
        size_t count = bytes < sizeof(scratch) ? (size_t)bytes : sizeof(scratch);
        if (!read_exact(source, scratch, count))
            return 0;
        bytes -= (uint32_t)count;
    }
    return 1;
}

wav_source_result_t wav_source_open(wav_source_t *source,
                                    wav_read_fn read,
                                    void *context)
{
    uint8_t header[16];
    uint32_t remaining;
    int format_seen = 0;

    if (source == NULL || read == NULL)
        return WAV_SOURCE_E_INVALID;
    memset(source, 0, sizeof(*source));
    source->read = read;
    source->context = context;
    if (!read_exact(source, header, 12u))
        return WAV_SOURCE_E_IO;
    if (memcmp(header, "RIFF", 4u) != 0 || memcmp(header + 8, "WAVE", 4u) != 0)
        return WAV_SOURCE_E_FORMAT;
    if (read_u32(header + 4) < 4u)
        return WAV_SOURCE_E_FORMAT;
    remaining = read_u32(header + 4) - 4u;

    while (remaining >= 8u) {
        uint32_t chunk_bytes;
        uint32_t padded_bytes;

        if (!read_exact(source, header, 8u))
            return WAV_SOURCE_E_IO;
        remaining -= 8u;
        chunk_bytes = read_u32(header + 4);
        padded_bytes = chunk_bytes + (chunk_bytes & 1u);
        if (padded_bytes < chunk_bytes || padded_bytes > remaining)
            return WAV_SOURCE_E_FORMAT;

        if (memcmp(header, "fmt ", 4u) == 0) {
            uint16_t channels;
            uint16_t block_align;
            uint16_t bits_per_sample;
            uint32_t sample_rate;
            uint32_t byte_rate;

            if (chunk_bytes < 16u || !read_exact(source, header, 16u))
                return WAV_SOURCE_E_FORMAT;
            channels = read_u16(header + 2);
            sample_rate = read_u32(header + 4);
            byte_rate = read_u32(header + 8);
            block_align = read_u16(header + 12);
            bits_per_sample = read_u16(header + 14);
            if (read_u16(header) != 1u || channels != 2u
                || sample_rate != 44100u || bits_per_sample != 16u
                || block_align != WAV_FRAME_BYTES
                || byte_rate != 44100u * WAV_FRAME_BYTES)
                return WAV_SOURCE_E_UNSUPPORTED;
            if (!skip_exact(source, padded_bytes - 16u))
                return WAV_SOURCE_E_IO;
            format_seen = 1;
        } else if (memcmp(header, "data", 4u) == 0) {
            if (!format_seen || (chunk_bytes % WAV_FRAME_BYTES) != 0u)
                return WAV_SOURCE_E_FORMAT;
            source->data_bytes_remaining = chunk_bytes;
            source->total_frames = chunk_bytes / WAV_FRAME_BYTES;
            return WAV_SOURCE_OK;
        } else if (!skip_exact(source, padded_bytes)) {
            return WAV_SOURCE_E_IO;
        }
        remaining -= padded_bytes;
    }
    return WAV_SOURCE_E_FORMAT;
}

size_t wav_source_read(wav_source_t *source, int16_t *interleaved_lr,
                       size_t frame_count)
{
    size_t available;
    size_t wanted;
    size_t received = 0u;
    uint8_t *destination = (uint8_t *)interleaved_lr;

    if (source == NULL || interleaved_lr == NULL || source->error != 0u)
        return 0u;
    available = source->data_bytes_remaining / WAV_FRAME_BYTES;
    if (frame_count > available)
        frame_count = available;
    wanted = frame_count * WAV_FRAME_BYTES;
    while (received < wanted) {
        size_t count = source->read(source->context, destination + received,
                                    wanted - received);
        if (count == 0u || count > wanted - received) {
            source->error = 1u;
            break;
        }
        received += count;
    }
    received -= received % WAV_FRAME_BYTES;
    source->data_bytes_remaining -= (uint32_t)received;
    return received / WAV_FRAME_BYTES;
}
