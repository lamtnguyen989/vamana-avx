// A basic vector codebook serialization format for quantization

/*
* Layout (little-endian, no padding):
    [
        uint32_t num_vectors
        uint32_t dim
        float   data[num_vectors * dim] (contiguous, row-major following "CPU"-friendlyness)
    ]
*/

#ifndef VECFILE_H
#define VECFILE_H

#include <stdlib.h>
#include <stdint.h>

typedef struct {
    uint32_t num_vectors;
    uint32_t dim;
    float* data;
} VecFile;



static inline void vecfile_free(VecFile* vf)
{
    // Free the heap data
    free(vf->data);

    // Default stack data
    vf->data = NULL;
    vf->num_vectors = 0;
    vf->dim = 0;
}

static inline int vecfile_load(const char* path, VecFile* vf)
{
    
    return 0;
}

static inline int vecfile_save()
{
    return 0;
}

#endif