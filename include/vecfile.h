// A basic vectors serialization data format for quantization
// This is our own file format of `.vecf` for storing data

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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Helper wrappers for error checking
#define TRY_READ(dst, size, count, file_stream, err_msg)                \
    do {                                                                \
        if (fread((dst), (size), (count), (file_stream)) != (count)) {  \
            fprintf(stderr, "%s", (err_msg));                           \
            fclose((file_stream));                                      \
            return -1;                                                  \
        }                                                               \
    } while(0);


// Vecfile data definition
typedef struct {
    uint32_t num_vectors;
    uint32_t dim;
    float* data;
} VecFile;


// Loading serialization data
static inline int vecfile_load(const char* path, VecFile* vf)
{
    // Open the serialized data file
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        perror("vecfile_load: Fail to open file for read.");
        return -1;
    }

    // Reading metadata
    const char* load_fail_msg = "vecfile_load: malformed serialized VecFile.";
    TRY_READ(&vf->num_vectors, sizeof(uint32_t), 1, file, load_fail_msg);
    TRY_READ(&vf->dim, sizeof(uint32_t), 1, file, load_fail_msg);

    // Creating space for data and reads in
    size_t n = (size_t)vf->num_vectors * (size_t)vf->dim;
    vf->data = (float*) malloc(n * sizeof(float));
    TRY_READ(vf->data, sizeof(float), n, file, "vecfile_load: data loads.");

    return 0;
}

// Serialize data write (write the data instead of from struct version)
static inline int vecfile_save(const char *path, uint32_t num_vectors, uint32_t dim, const float *data)
{
    // Open file for serialized writes
    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        perror("vecfile_load: Fail to open file for write.");
        return -1;
    }

    // Due to partial writes possibility, holding off on error-checking for now
    size_t n = (size_t)num_vectors * (size_t)dim;

    fwrite(&num_vectors, sizeof(uint32_t), 1, file);
    fwrite(&dim, sizeof(uint32_t), 1, file);
    fwrite(data, sizeof(float), n, file);
    
    fclose(file);

    return 0;
}

// Indexing vecfile data
static inline float* vecfile_data_at(const VecFile *vf, uint32_t k) {
    return &vf->data[(size_t)k * vf->dim];
}


// Cleanups
static inline void vecfile_free(VecFile* vf)
{
    // Free the heap data
    free(vf->data);

    // Default stack data
    vf->data = NULL;
    vf->num_vectors = 0;
    vf->dim = 0;
}

#endif