/* SPDX-License-Identifier: MIT */
#include "sd_mp3_source.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t position;
    size_t maximum_chunk;
    int fail_read;
} memory_stream_t;

static size_t memory_read(void *context, void *destination, size_t bytes)
{
    memory_stream_t *stream = (memory_stream_t *)context;
    size_t remaining;

    if (stream->fail_read)
        return SIZE_MAX;
    remaining = stream->size - stream->position;
    if (bytes > stream->maximum_chunk)
        bytes = stream->maximum_chunk;
    if (bytes > remaining)
        bytes = remaining;
    memcpy(destination, stream->data + stream->position, bytes);
    stream->position += bytes;
    return bytes;
}

static uint8_t *read_file_from(const char *environment_name,
                               const char *name, size_t *size_out)
{
    char path[512];
    FILE *file;
    long size;
    uint8_t *data;
    const char *fixture_directory = getenv(environment_name);

    assert(fixture_directory != NULL);
    assert(snprintf(path, sizeof(path), "%s/%s", fixture_directory, name)
           < (int)sizeof(path));
    file = fopen(path, "rb");
    assert(file != NULL);
    assert(fseek(file, 0, SEEK_END) == 0);
    size = ftell(file);
    assert(size > 0 && fseek(file, 0, SEEK_SET) == 0);
    data = (uint8_t *)malloc((size_t)size);
    assert(data != NULL);
    assert(fread(data, 1u, (size_t)size, file) == (size_t)size);
    assert(fclose(file) == 0);
    *size_out = (size_t)size;
    return data;
}

static uint8_t *read_fixture(const char *name, size_t *size_out)
{
    return read_file_from("MP3_FIXTURE_DIR", name, size_out);
}

static uint8_t *read_vector(const char *name, size_t *size_out)
{
    return read_file_from("MP3_VECTOR_DIR", name, size_out);
}

static uint32_t fnv1a_update(uint32_t value, const void *data, size_t bytes)
{
    const uint8_t *cursor = (const uint8_t *)data;
    size_t index;

    for (index = 0u; index < bytes; ++index) {
        value ^= cursor[index];
        value *= 0x01000193u;
    }
    return value;
}

static void decode_fixture(const uint8_t *data, size_t size,
                           size_t maximum_chunk, size_t output_capacity,
                           uint32_t expected_frames,
                           uint32_t expected_hash)
{
    int16_t pcm[1152u * 2u];
    memory_stream_t stream = {data, size, 0u, maximum_chunk, 0};
    mp3_io_t io = {memory_read, &stream};
    mp3_source_t source;
    uint32_t frames = 0u;
    uint32_t hash = 0x811c9dc5u;

    assert(mp3_source_open(&source, &io) == MP3_OK);
    for (;;) {
        size_t produced = 0u;
        mp3_result_t result = mp3_source_decode(
            &source, pcm, output_capacity, &produced);
        if (result == MP3_EOF)
            break;
        assert(result == MP3_OK && produced != 0u);
        frames += (uint32_t)produced;
        hash = fnv1a_update(hash, pcm, produced * 2u * sizeof(pcm[0]));
    }
    if (frames != expected_frames || hash != expected_hash)
        fprintf(stderr, "decoded frames=%lu hash=0x%08lx\n",
                (unsigned long)frames, (unsigned long)hash);
    assert(frames == expected_frames);
    assert(hash == expected_hash);
}

static void test_cbr_vbr_and_split_sync(void)
{
    size_t cbr_size;
    size_t vbr_size;
    uint8_t *cbr = read_fixture("test_44100_stereo_cbr.mp3", &cbr_size);
    uint8_t *vbr = read_fixture("test_44100_stereo_vbr.mp3", &vbr_size);

    decode_fixture(cbr, cbr_size, cbr_size, MP3_SOURCE_FRAME_SAMPLES,
                   134784u, 0xf0f5edd2u);
    decode_fixture(vbr, vbr_size, 7u, 127u,
                   134784u, 0x23df9715u);
    free(vbr);
    free(cbr);
}

static void test_id3_prefix(void)
{
    size_t mp3_size;
    uint8_t *mp3 = read_fixture("test_44100_stereo_cbr.mp3", &mp3_size);
    const size_t payload = 19u;
    uint8_t *tagged = (uint8_t *)calloc(1u, 10u + payload + mp3_size);

    assert(tagged != NULL);
    memcpy(tagged, "ID3\x04\x00\x00", 6u);
    tagged[9] = (uint8_t)payload;
    memset(tagged + 10u, 0x5au, payload);
    memcpy(tagged + 10u + payload, mp3, mp3_size);
    decode_fixture(tagged, 10u + payload + mp3_size, 3u, 128u,
                   134784u, 0xf0f5edd2u);
    free(tagged);
    free(mp3);
}

static void test_format_rejection_and_io_error(void)
{
    uint8_t data[128] = {0};
    memory_stream_t stream = {data, sizeof(data), 0u, sizeof(data), 0};
    mp3_io_t io = {memory_read, &stream};
    mp3_source_t source;

    assert(mp3_source_open(&source, &io) == MP3_E_SYNC);
    stream = (memory_stream_t){data, sizeof(data), 0u, sizeof(data), 1};
    assert(mp3_source_open(&source, &io) == MP3_E_IO);
}

static mp3_result_t open_bytes(const uint8_t *data, size_t size,
                               size_t maximum_chunk)
{
    memory_stream_t stream = {data, size, 0u, maximum_chunk, 0};
    mp3_io_t io = {memory_read, &stream};
    mp3_source_t source;
    return mp3_source_open(&source, &io);
}

static void test_strict_format_rejection(void)
{
    static const struct {
        const char *name;
        mp3_result_t expected;
    } cases[] = {
        {"invalid_layer2.mp2", MP3_E_LAYER},
        {"invalid_48000_stereo.mp3", MP3_E_RATE},
        {"invalid_44100_mono.mp3", MP3_E_CHANNELS},
        {"invalid_mpeg2_stereo.mp3", MP3_E_RATE},
    };
    size_t index;

    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        size_t size;
        uint8_t *data = read_vector(cases[index].name, &size);
        assert(open_bytes(data, size, 5u) == cases[index].expected);
        free(data);
    }
}

static size_t find_frame_after(const uint8_t *data, size_t size, size_t start)
{
    size_t index;

    for (index = start; index + 4u <= size; ++index) {
        if (data[index] == 0xffu && (data[index + 1u] & 0xe0u) == 0xe0u)
            return index;
    }
    return size;
}

static void test_corruption_recovers_and_sync_is_bounded(void)
{
    int16_t pcm[MP3_SOURCE_FRAME_SAMPLES * 2u];
    size_t size;
    uint8_t *mp3 = read_fixture("test_44100_stereo_cbr.mp3", &size);
    size_t corrupt = find_frame_after(mp3, size, 1000u);
    memory_stream_t stream;
    mp3_io_t io;
    mp3_source_t source;
    uint32_t frames = 0u;
    uint32_t hash = 0x811c9dc5u;

    assert(corrupt + 4u <= size);
    memset(mp3 + corrupt, 0u, 4u);
    stream = (memory_stream_t){mp3, size, 0u, 11u, 0};
    io = (mp3_io_t){memory_read, &stream};
    assert(mp3_source_open(&source, &io) == MP3_OK);
    for (;;) {
        size_t produced = 0u;
        mp3_result_t result = mp3_source_decode(
            &source, pcm, MP3_SOURCE_FRAME_SAMPLES, &produced);
        if (result == MP3_EOF)
            break;
        assert(result == MP3_OK);
        frames += (uint32_t)produced;
        hash = fnv1a_update(hash, pcm, produced * 2u * sizeof(pcm[0]));
    }
    assert(frames >= 100u * MP3_SOURCE_FRAME_SAMPLES);
    assert(frames <= 134784u);
    assert(hash != 0xf0f5edd2u);
    free(mp3);

    size = MP3_SOURCE_MAX_RESYNC_BYTES + 4u;
    mp3 = (uint8_t *)calloc(1u, size);
    assert(mp3 != NULL);
    assert(open_bytes(mp3, size, 13u) == MP3_E_SYNC);
    free(mp3);
}

static void test_later_format_change_is_rejected(void)
{
    int16_t pcm[MP3_SOURCE_FRAME_SAMPLES * 2u];
    size_t cbr_size;
    size_t mono_size;
    uint8_t *cbr = read_fixture("test_44100_stereo_cbr.mp3", &cbr_size);
    uint8_t *mono = read_vector("invalid_44100_mono.mp3", &mono_size);
    uint8_t *joined = (uint8_t *)malloc(cbr_size + mono_size);
    memory_stream_t stream;
    mp3_io_t io;
    mp3_source_t source;
    mp3_result_t result;

    assert(joined != NULL);
    memcpy(joined, cbr, cbr_size);
    memcpy(joined + cbr_size, mono, mono_size);
    stream = (memory_stream_t){joined, cbr_size + mono_size, 0u, 17u, 0};
    io = (mp3_io_t){memory_read, &stream};
    assert(mp3_source_open(&source, &io) == MP3_OK);
    do {
        size_t produced = 0u;
        result = mp3_source_decode(
            &source, pcm, MP3_SOURCE_FRAME_SAMPLES, &produced);
    } while (result == MP3_OK);
    assert(result == MP3_E_CHANNELS);
    free(joined);
    free(mono);
    free(cbr);
}

int main(void)
{
    test_cbr_vbr_and_split_sync();
    test_id3_prefix();
    test_format_rejection_and_io_error();
    test_strict_format_rejection();
    test_corruption_recovers_and_sync_is_bounded();
    test_later_format_change_is_rejected();
    puts("mp3 source: PASS");
    return 0;
}
