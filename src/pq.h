// Product Quantization implementation

#ifndef PQ_H
#define PQ_H

#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "distance.h"

#define PQ_MAGIC 0x50513031u /* ASCII code for "PQ01" */

typedef struct {
    float* centroids;   // Centroid data
    uint64_t hash;      // Centroid signature for versioning
    uint32_t dim;       // Dimension of search space
    uint32_t M;         // Number of sub-vectors
    uint32_t sub_dim;   // Sub-dimension of the quantization encoding
    uint32_t K;         // Centroids per subspace
} PQCodebook;


// Centroid getter
static inline float* pq_get_centroid(const PQCodebook* pq, uint32_t m, uint32_t k) {
    return pq->centroids + ((size_t) m * pq->K + k) * pq->sub_dim;
}

// Perform the product quantization encoding
static inline void pq_encode(const PQCodebook* pq, dist_fn_t distance, const float* vec, uint8_t* out) {
    for (uint32_t m = 0; m < pq->M; m++) {
        // Pointer to the m-th sub-vector slice
        const float* sub_vec = vec + (size_t)m * pq->sub_dim;

        // Find the closest centroid to the representative subspace vector
        float closest_dist = FLT_MAX;
        uint32_t closest_idx = 0;
        for (uint32_t k = 0; k < pq->K; k++) {
            float d = distance(sub_vec, pq_get_centroid(pq, m, k), pq->sub_dim);
            if (d < closest_dist) {
                closest_dist = d;
                closest_idx = k;
            }
        }
        out[m] = (uint8_t)closest_idx;
    }
}

// Building (ADC) distance look up table for the query
static inline void pq_build_distance_table(const PQCodebook* pq, dist_fn_t distance, const float* query, float* table)
{
    for (uint32_t m = 0; m < pq->M; m++) {
        const float* query_sub_vec = query + (size_t)m * pq->sub_dim;
        for (uint32_t k = 0; k < pq->K; k++) {
            table[(size_t)m * pq->K + k] = distance(query_sub_vec, pq_get_centroid(pq, m, k), pq->sub_dim);
        }
    }
}

// ADC distances
static inline void pq_adc_distances(const PQCodebook* pq, float* adc_table, uint8_t* codes) 
{
    float result = 0.0f;
    for (uint32_t m = 0; m < pq->M; m++) {
        result += adc_table[(size_t)m * pq->K + codes[m]];
    }
}

// Cleanups
static inline void pq_codebook_free(PQCodebook* pq)
{
    free(pq-> centroids);
    free(pq);
}


// Fingerprinting hash for the codebook data
static inline uint64_t fnv1a_hash64(const void* data, size_t len, uint64_t seed) 
{
    // Hashing based on FNV-1a scheme: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function 
    uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    uint64_t FNV_PRIME = 0x00000100000001b3ULL;

    uint64_t hash = seed ^= FNV_OFFSET;
    const uint8_t* p = (uint8_t*) data;
    for (size_t j = 0; j < len; j++) {
        hash ^= p[j];
        hash *= FNV_PRIME;
    }

    return hash;
}

// Hashing the codebook via hashing all the data (iteratively)
static inline uint64_t pq_codebook_hash(PQCodebook* pq)
{
    uint64_t hash = fnv1a_hash64(&pq->dim, sizeof(uint32_t), 0);
    hash = fnv1a_hash64(&pq->M, sizeof(uint32_t), hash);
    hash = fnv1a_hash64(&pq->K, sizeof(uint32_t), hash);
    hash = fnv1a_hash64(pq->centroids, (size_t)pq->M * pq->K * pq->sub_dim * sizeof(float), hash);
    return hash;
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