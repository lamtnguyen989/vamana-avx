// Vector search indexing formats

#ifndef INDEX_FORMAT_H
#define INDEX_FORMAT_H

#include <stdint.h>

/* Index book keeping data structures */
// Header
typedef struct {
    uint32_t n_points;
    uint32_t dim;
    uint32_t R;
    uint32_t medoid_id;
} IndexHeader;

// Record for a single vector (query)
typedef struct {
    const float*    vector;
    uint32_t        degree;
    const uint32_t* neighbors;
} IndexRecord;


/* List (essentially a vec or array) of searching candidates */
// Searching candidates (with respect to a vector)
typedef struct {
    uint32_t id;
    float   dist;
    uint8_t visited;
} Candidate;


typedef struct {
    Candidate* items;
    uint32_t size;
    uint32_t capacity;
} CandidateList;



#endif /* INDEX_FORMAT_H */