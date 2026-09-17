/* SPDX-License-Identifier: MIT */
#include "pcm_ring.h"

static size_t minimum_size(size_t left, size_t right)
{
    return left < right ? left : right;
}

pcm_ring_result_t pcm_ring_init(pcm_ring_t *ring,
                                pcm_stereo_frame_t *storage,
                                uint32_t capacity_frames)
{
    if (ring == NULL || storage == NULL || capacity_frames < 2u
        || (capacity_frames & (capacity_frames - 1u)) != 0u
        || capacity_frames > (UINT32_MAX / 2u + 1u))
        return PCM_RING_E_INVALID;
    ring->storage = storage;
    ring->capacity = capacity_frames;
    ring->mask = capacity_frames - 1u;
    ring->producer = 0u;
    ring->consumer = 0u;
    return PCM_RING_OK;
}

size_t pcm_ring_size(const pcm_ring_t *ring)
{
    if (ring == NULL || ring->storage == NULL)
        return 0u;
    return (size_t)(ring->producer - ring->consumer);
}

size_t pcm_ring_space(const pcm_ring_t *ring)
{
    size_t size;

    if (ring == NULL || ring->storage == NULL)
        return 0u;
    size = pcm_ring_size(ring);
    return size < ring->capacity ? (size_t)ring->capacity - size : 0u;
}

size_t pcm_ring_push(pcm_ring_t *ring, const pcm_stereo_frame_t *frames,
                     size_t frame_count)
{
    size_t index;
    size_t count;

    if (ring == NULL || frames == NULL)
        return 0u;
    count = minimum_size(frame_count, pcm_ring_space(ring));
    for (index = 0u; index < count; ++index)
        ring->storage[(ring->producer + (uint32_t)index) & ring->mask]
            = frames[index];
    ring->producer += (uint32_t)count;
    return count;
}

size_t pcm_ring_peek(const pcm_ring_t *ring, pcm_stereo_frame_t *frames,
                     size_t frame_count)
{
    size_t index;
    size_t count;

    if (ring == NULL || frames == NULL)
        return 0u;
    count = minimum_size(frame_count, pcm_ring_size(ring));
    for (index = 0u; index < count; ++index)
        frames[index] = ring->storage[
            (ring->consumer + (uint32_t)index) & ring->mask];
    return count;
}

size_t pcm_ring_drop(pcm_ring_t *ring, size_t frame_count)
{
    size_t count;

    if (ring == NULL)
        return 0u;
    count = minimum_size(frame_count, pcm_ring_size(ring));
    ring->consumer += (uint32_t)count;
    return count;
}
