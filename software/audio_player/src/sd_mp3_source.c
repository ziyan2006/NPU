/* SPDX-License-Identifier: MIT */
#define MINIMP3_IMPLEMENTATION
#include "sd_mp3_source.h"

#include <limits.h>
#include <string.h>

#define ID3_HEADER_BYTES 10u

static size_t buffered_bytes(const mp3_source_t *source)
{
    return source->buffer_end - source->buffer_begin;
}

static void consume_bytes(mp3_source_t *source, size_t bytes)
{
    size_t available = buffered_bytes(source);

    if (bytes > available)
        bytes = available;
    source->buffer_begin += bytes;
    if (source->buffer_begin == source->buffer_end)
        source->buffer_begin = source->buffer_end = 0u;
}

static mp3_result_t refill(mp3_source_t *source)
{
    size_t available;

    if (source->buffer_begin != 0u) {
        available = buffered_bytes(source);
        memmove(source->buffer, source->buffer + source->buffer_begin,
                available);
        source->buffer_begin = 0u;
        source->buffer_end = available;
    }
    while (!source->end_of_stream
           && source->buffer_end < sizeof(source->buffer)) {
        size_t received = source->io.read(
            source->io.context, source->buffer + source->buffer_end,
            sizeof(source->buffer) - source->buffer_end);
        if (received == SIZE_MAX
            || received > sizeof(source->buffer) - source->buffer_end) {
            source->io_error = 1u;
            return MP3_E_IO;
        }
        if (received == 0u) {
            source->end_of_stream = 1u;
            break;
        }
        source->buffer_end += received;
    }
    return MP3_OK;
}

static mp3_result_t skip_stream_bytes(mp3_source_t *source, size_t bytes)
{
    while (bytes != 0u) {
        size_t available;
        size_t chunk;
        mp3_result_t result = refill(source);

        if (result != MP3_OK)
            return result;
        available = buffered_bytes(source);
        if (available == 0u)
            return MP3_E_SYNC;
        chunk = bytes < available ? bytes : available;
        consume_bytes(source, chunk);
        bytes -= chunk;
    }
    return MP3_OK;
}

static mp3_result_t skip_id3v2(mp3_source_t *source)
{
    const uint8_t *header;
    size_t tag_bytes;
    mp3_result_t result = refill(source);

    if (result != MP3_OK)
        return result;
    if (buffered_bytes(source) < ID3_HEADER_BYTES
        || memcmp(source->buffer + source->buffer_begin, "ID3", 3u) != 0)
        return MP3_OK;
    header = source->buffer + source->buffer_begin;
    if ((header[6] | header[7] | header[8] | header[9]) >= 0x80u)
        return MP3_E_SYNC;
    tag_bytes = ID3_HEADER_BYTES
        + ((size_t)header[6] << 21)
        + ((size_t)header[7] << 14)
        + ((size_t)header[8] << 7)
        + (size_t)header[9];
    if ((header[5] & 0x10u) != 0u)
        tag_bytes += ID3_HEADER_BYTES;
    return skip_stream_bytes(source, tag_bytes);
}

static mp3_result_t validate_format(const mp3dec_frame_info_t *info)
{
    if (info->layer != 3)
        return MP3_E_LAYER;
    if (info->hz != 44100)
        return MP3_E_RATE;
    if (info->channels != 2)
        return MP3_E_CHANNELS;
    return MP3_OK;
}

static mp3_result_t count_skipped(mp3_source_t *source, size_t bytes)
{
    if (bytes > MP3_SOURCE_MAX_RESYNC_BYTES - source->skipped_bytes)
        return MP3_E_SYNC;
    source->skipped_bytes += (uint32_t)bytes;
    return MP3_OK;
}

static mp3_result_t load_frame(mp3_source_t *source)
{
    for (;;) {
        mp3dec_frame_info_t info;
        mp3_result_t result;
        size_t available;
        size_t consume;
        int samples;

        result = refill(source);
        if (result != MP3_OK)
            return result;
        available = buffered_bytes(source);
        if (available < 4u && source->end_of_stream)
            return source->decoded_frames == 0u ? MP3_E_SYNC : MP3_EOF;
        if (available > (size_t)INT_MAX)
            return MP3_E_INVALID;
        memset(&info, 0, sizeof(info));
        samples = mp3dec_decode_frame(
            &source->decoder, source->buffer + source->buffer_begin,
            (int)available, source->pending_pcm, &info);
        if (samples > 0) {
            result = count_skipped(source, (size_t)info.frame_offset);
            if (result != MP3_OK)
                return result;
            result = validate_format(&info);
            if (result != MP3_OK)
                return result;
            if (samples > (int)MP3_SOURCE_FRAME_SAMPLES)
                return MP3_E_INVALID;
            consume_bytes(source, (size_t)info.frame_bytes);
            source->pending_begin = 0u;
            source->pending_frames = (size_t)samples;
            source->skipped_bytes = 0u;
            source->format_locked = 1u;
            ++source->decoded_frames;
            return MP3_OK;
        }

        if (source->end_of_stream) {
            consume = available;
        } else if (info.frame_bytes > 0
                   && (size_t)info.frame_bytes < available) {
            consume = (size_t)info.frame_bytes;
        } else {
            consume = available > 3u ? available - 3u : 0u;
        }
        if (consume == 0u)
            return source->decoded_frames == 0u ? MP3_E_SYNC : MP3_EOF;
        result = count_skipped(source, consume);
        if (result != MP3_OK)
            return result;
        consume_bytes(source, consume);
        if (source->end_of_stream && buffered_bytes(source) == 0u)
            return source->decoded_frames == 0u ? MP3_E_SYNC : MP3_EOF;
    }
}

mp3_result_t mp3_source_open(mp3_source_t *source, const mp3_io_t *io)
{
    mp3_result_t result;

    if (source == NULL || io == NULL || io->read == NULL)
        return MP3_E_INVALID;
    memset(source, 0, sizeof(*source));
    source->io = *io;
    mp3dec_init(&source->decoder);
    result = skip_id3v2(source);
    if (result != MP3_OK)
        return result;
    return load_frame(source);
}

mp3_result_t mp3_source_decode(mp3_source_t *source,
                               int16_t *interleaved_lr,
                               size_t frame_capacity,
                               size_t *frames_out)
{
    size_t frames;
    mp3_result_t result;

    if (frames_out != NULL)
        *frames_out = 0u;
    if (source == NULL || interleaved_lr == NULL || frames_out == NULL
        || frame_capacity == 0u || source->io.read == NULL)
        return MP3_E_INVALID;
    if (source->io_error)
        return MP3_E_IO;
    if (source->pending_begin == source->pending_frames) {
        source->pending_begin = source->pending_frames = 0u;
        result = load_frame(source);
        if (result != MP3_OK)
            return result;
    }
    frames = source->pending_frames - source->pending_begin;
    if (frames > frame_capacity)
        frames = frame_capacity;
    memcpy(interleaved_lr, source->pending_pcm + source->pending_begin * 2u,
           frames * 2u * sizeof(interleaved_lr[0]));
    source->pending_begin += frames;
    *frames_out = frames;
    return MP3_OK;
}
