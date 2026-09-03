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

/* Product Quantization codebook */
// This will be serialized to `.pqbook` for storing data
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
    return &pq->centroids[((size_t) m * pq->K + k) * pq->sub_dim];
}

// Perform the product quantization encoding
static inline void pq_encode(const PQCodebook* pq, dist_fn_t distance, const float* vec, uint8_t* out) {
    for (uint32_t m = 0; m < pq->M; m++) {
        // Pointer to the m-th sub-vector slice
        const float* sub_vec = &vec[(size_t)m * pq->sub_dim];

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
        const float* query_sub_vec = &query[(size_t)m * pq->sub_dim];
        for (uint32_t k = 0; k < pq->K; k++) {
            table[(size_t)m * pq->K + k] = distance(query_sub_vec, pq_get_centroid(pq, m, k), pq->sub_dim);
        }
    }
}

// ADC distances
static inline float pq_adc_distances(const PQCodebook* pq, float* adc_table, uint8_t* codes) 
{
    float result = 0.0f;
    for (size_t m = 0; m < pq->M; m++) {
        result += adc_table[m * pq->K + codes[m]];
    }

    return result;
}

// Cleanups
static inline void pq_codebook_free(PQCodebook* pq)
{
    // Note we assume codebook struct is stack-allocated except for the data
    free(pq-> centroids);
    pq->centroids = NULL;
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


// Serializing PQ codebook
static inline int pq_codebook_save(const char* path, PQCodebook *pq) 
{
    // Opening file for serializing codebook data
    FILE* codebook = fopen(path, "wb");
    if (codebook == NULL) {
        perror("pq_codebook_save: fopen\n");
        return -1;
    }

    // Writing codebook metadata to disk
    uint32_t pq_magic = PQ_MAGIC;
    pq->hash = pq_codebook_hash(pq);
    fwrite(&pq_magic, sizeof(uint32_t), 1, codebook);
    fwrite(&pq->dim, sizeof(uint32_t), 1, codebook);
    fwrite(&pq->M, sizeof(uint32_t), 1, codebook);
    fwrite(&pq->K, sizeof(uint32_t), 1, codebook);
    fwrite(&pq->hash, sizeof(uint64_t), 1, codebook);

    // Writing actual centroid data to disk
    size_t n = (size_t)pq->M * pq->K * pq->sub_dim;
    fwrite(pq->centroids, sizeof(float), n, codebook);
    fclose(codebook);

    return 0;
}

// Loading codebook data
static inline int pq_codebook_load(const char* path, PQCodebook *pq) 
{
    // Opening codebook data file
    FILE* codebook = fopen(path, "rb");
    if (codebook == NULL) {
        perror("pq_codebook_load: fopen\n");
        return -1;
    }

    // Verifying magic bytes
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, codebook) != 1 || magic != PQ_MAGIC) {
        fprintf(stderr, "pq_codebook_load: bad magic in %s\n", path);
        fclose(codebook);
        return -1;
    }

    // Populating codebook metadata
    if (fread(&pq->dim, sizeof(uint32_t), 1, codebook) != 1 ||
        fread(&pq->M, sizeof(uint32_t), 1, codebook) != 1 ||
        fread(&pq->K, sizeof(uint32_t), 1, codebook) != 1 ||
        fread(&pq->hash, sizeof(uint64_t), 1, codebook) != 1) 
    {
        fprintf(stderr, "pq_codebook_load: header corrupted or not match expected format in %s\n", path);
        fclose(codebook); 
        return -1;
    }

    // Populating codebook data in memory
    pq->sub_dim = pq->dim / pq->M;

    size_t n = (size_t)pq->M * pq->K * pq->sub_dim;
    pq->centroids = (float*) malloc(n * sizeof(float));
    if (fread(pq->centroids, sizeof(float), n, codebook) != n) {
        fprintf(stderr, "pq_codebook_load: centroid data loading problem in %s\n", path);
        fclose(codebook); 
        free(pq->centroids); 
        return -1;
    }
        fclose(codebook);

    return 0;
}


/* Product quantization encodings of the codebook serialization (used for encoding data shard) */
// This will be serialized to `.pqbin` for storing data
typedef struct {
    uint32_t n_points;
    uint32_t M;
    uint64_t codebook_hash;
    uint8_t* data;
} PQCodes;

static inline int pq_codes_load(const char* path, PQCodes* pq_codes) 
{
    // Setting up encoding codes file stram
    FILE* encodings_file = fopen(path, "rb");
    if (encodings_file == NULL) {
        perror("pq_codes_load: fopen");
        return -1;
    }

    // Reading magic bytes
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, encodings_file) != 1 || magic != PQ_MAGIC) {
        fprintf(stderr, "pq_codes_load: bad magic bytes in %s\n", path); 
        fclose(encodings_file); 
        return -1;
    }

    // Reading encodings metadata
    if (fread(&pq_codes->n_points, sizeof(uint32_t), 1, encodings_file) != 1 ||
        fread(&pq_codes->M, sizeof(uint32_t), 1, encodings_file) != 1 ||
        fread(&pq_codes->codebook_hash, sizeof(uint64_t), 1, encodings_file) != 1) 
    {
        fprintf(stderr, "pq_codes_load: header corrupted or not match expected format in %s\n", path);
        fclose(encodings_file);
        return -1;
    }

    // Reading actual data
    size_t data_size = (size_t)pq_codes->n_points * pq_codes->M;
    pq_codes->data = (uint8_t*) malloc(data_size);
    if (fread(pq_codes->data, 1, data_size, encodings_file) != data_size) {
        fprintf(stderr, "pq_codes_load: encoding data loading problem in %s\n", path);
        fclose(encodings_file); 
        free(pq_codes->data); 
        return -1;
    }
    fclose(encodings_file);

    return 0;
}

// Check the hash to make sure the we are using the correct encoding scheme
static inline int pq_codes_matches_codebook(const PQCodes* pq_codes, const PQCodebook* pq_codebook) 
{
    return (pq_codes->codebook_hash == pq_codebook->hash);
}

// Cleanup encoding instance
static inline void pq_codes_free(PQCodes* pq_codes) 
{
    free(pq_codes->data);
    pq_codes->data = NULL;
}

// PQ encoding data indexing
static inline uint8_t* pq_codes_at(PQCodes* pq_codes, uint32_t point_id) 
{
    return &pq_codes->data[(size_t)point_id*pq_codes->M];
}



#endif /* PQ_H */