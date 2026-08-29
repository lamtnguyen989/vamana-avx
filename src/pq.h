// Product Quantization implementation

#ifndef PQ_H
#define PQ_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    float* centroids;
    uint32_t dim;
    uint32_t M;
    uint32_t K;
} PQCodebook;

static inline void pq_codebook_free(PQCodebook* pq){
    free(pq-> centroids);
    free(pq);
}

static inline int pq_codebook_save(const char* path, PQCodebook *pq) {
    // TODO
    return 0;
}

static inline int pq_codebook_load(const char* path, PQCodebook *pq) {
    // TODO
    return 0;
}

#endif /* PQ_H */