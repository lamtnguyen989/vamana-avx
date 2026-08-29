#ifndef QUANTIZER_H
#define QUANTIZER_H

#include <cstdint>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Product Quantization implementaqtion
#if defined(PQ_IMPLEMENTATION)

typedef struct {
    float* centroids;
    uint32_t dim;
    uint32_t M;
    uint32_t K;
} PQCodebook;




#endif /* PQ_IMPLEMENTATION */

#endif /* QUANTIZER_H */