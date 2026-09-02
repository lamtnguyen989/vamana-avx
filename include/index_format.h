/*
    Vector search index format:
        Header:
            uint32_t num_points
            uint32_t dim
            uint32_t R          (max out-degree)
            uint32_t medoid_id  (fixed search entry point)
        
        Record:
*/

#ifndef INDEX_FORMAT_H
#define INDEX_FORMAT_H

#include <stdint.h>

// Header
typedef struct {
    uint32_t n_points;
    uint32_t dim;
    uint32_t R;
    uint32_t medoid_id;
} IndexHeader;

// Record (data and searching context)
typedef struct {

} IndexRecord;

#endif /* INDEX_FORMAT_H */