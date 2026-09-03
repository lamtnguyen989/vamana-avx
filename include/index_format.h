// Vector search indexing formats

#ifndef INDEX_FORMAT_H
#define INDEX_FORMAT_H

#include <stdint.h>
#include <stdlib.h>

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

// Transforming byte buffer into an index record
static inline void index_record_decode(IndexHeader* header, const uint8_t* buffer, IndexRecord* out_record)
{
    out_record->vector = (const float*) buffer;
    const uint32_t *tail = (const uint32_t*) (buffer + (size_t)header->dim * sizeof(float));
    out_record->degree = tail[0];
    out_record->neighbors = tail + 1;
}


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

static inline void candidate_list_init(CandidateList* list, uint32_t capacity)
{
    list->items = (Candidate*) malloc(capacity*sizeof(Candidate));
    list->size = 0;
    list->capacity = capacity;
}

static inline void candidate_list_free(CandidateList *list)
{
    free(list->items);
    list->items = NULL;
}

// Insert candidate into the list, sorted by ascending distance
// Returns 1 for success (true) and 0 (false) for fail 
static inline int insert_candidate(CandidateList *list, uint32_t id, float distance)
{
    /* Rejection checks */
    // Check if the id is already present
    for (uint32_t i = 0; i < list->size; i++) {
        if (list->items[i].id == id) {
            return 0;
        }
    }
    // Reject inserting candidate if the list is full and the candidates is worse than the worst candidate
    if ((list->size == list->capacity) && (list->items[list->size - 1].dist <= distance)) {
        return 0;
    }

    /* Insertion */
    // Finding insertion point
    uint32_t last_pos = (list->size < list->capacity) ? (list->size) : (list->capacity - 1);
    uint32_t insert_pos = 0;
    while ((insert_pos < last_pos) && list->items[insert_pos].dist < distance) {
        insert_pos++;
    }
    
    // Insert to the end of the list 
    for (uint32_t i = last_pos; i > insert_pos; i--) {
        list->items[i] = list->items[i - 1];
    }
    list->items[insert_pos] = (Candidate){
        .id = id,
        .dist = distance,
        .visited = 0,
    };

    // Increase the size after successfully insert
    if (list->size < list->capacity) {
        list->size++;
    }

    return 1;
}

static inline uint32_t next_unvisted_candidates(CandidateList* list, uint32_t* out_indices, uint32_t max_out) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < list->size && count < max_out; i++) {
        if (!list->items[i].visited) {
            out_indices[count++] = i;
        }
    }
    return count;
}



#endif /* INDEX_FORMAT_H */