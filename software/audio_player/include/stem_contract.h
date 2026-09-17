/* SPDX-License-Identifier: MIT */
#ifndef STEM_CONTRACT_H
#define STEM_CONTRACT_H

#define STEM_SAMPLE_RATE_HZ       44100u
#define STEM_FFT_SIZE             1024u
#define STEM_HOP_SIZE             256u
#define STEM_FFT_BINS             513u
#define STEM_BAND_COUNT           128u
#define STEM_BLOCK_FRAMES         16u
#define STEM_BLOCK_SAMPLES        4096u
#define STEM_INPUT_CHANNELS       2u
#define STEM_OUTPUT_CHANNELS      4u
#define STEM_NHWC8_LANES          8u
#define STEM_STORAGE_BYTES        2u

#define STEM_INPUT_STORAGE_BYTES \
    (STEM_BAND_COUNT * STEM_BLOCK_FRAMES * STEM_NHWC8_LANES * \
     STEM_STORAGE_BYTES)
#define STEM_OUTPUT_STORAGE_BYTES STEM_INPUT_STORAGE_BYTES

_Static_assert(STEM_BLOCK_SAMPLES == STEM_BLOCK_FRAMES * STEM_HOP_SIZE,
               "4096-sample blocks must contain 16 256-sample hops");
_Static_assert(STEM_FFT_SIZE == 1024u, "STEM frontend requires a 1024-point FFT");
_Static_assert(STEM_HOP_SIZE == 256u, "STEM frontend requires a 256-sample hop");
_Static_assert(STEM_BLOCK_SAMPLES == 4096u,
               "STEM runtime requires 4096 samples per block");
_Static_assert(STEM_INPUT_CHANNELS <= STEM_NHWC8_LANES &&
               STEM_OUTPUT_CHANNELS <= STEM_NHWC8_LANES,
               "STEM tensors must fit one NHWC8 channel group");
_Static_assert(STEM_INPUT_STORAGE_BYTES == 32768u,
               "STEM input storage contract changed");
_Static_assert(STEM_OUTPUT_STORAGE_BYTES == 32768u,
               "STEM output storage contract changed");

#endif /* STEM_CONTRACT_H */
