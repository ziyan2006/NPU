/* SPDX-License-Identifier: MIT */
#ifndef STEM_PCM_RING_H
#define STEM_PCM_RING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PCM_RING_OK = 0,
    PCM_RING_E_INVALID = -1
} pcm_ring_result_t;

typedef struct {
    int16_t left;
    int16_t right;
} pcm_stereo_frame_t;

typedef struct {
    pcm_stereo_frame_t *storage;
    uint32_t capacity;
    uint32_t mask;
    uint32_t producer;
    uint32_t consumer;
} pcm_ring_t;

pcm_ring_result_t pcm_ring_init(pcm_ring_t *ring,
                                pcm_stereo_frame_t *storage,
                                uint32_t capacity_frames);
size_t pcm_ring_size(const pcm_ring_t *ring);
size_t pcm_ring_space(const pcm_ring_t *ring);
size_t pcm_ring_push(pcm_ring_t *ring, const pcm_stereo_frame_t *frames,
                     size_t frame_count);
size_t pcm_ring_peek(const pcm_ring_t *ring, pcm_stereo_frame_t *frames,
                     size_t frame_count);
size_t pcm_ring_drop(pcm_ring_t *ring, size_t frame_count);

#ifdef __cplusplus
}
#endif

#endif /* STEM_PCM_RING_H */
