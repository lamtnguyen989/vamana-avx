#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SWAP(a, b)              \
    do {                        \
        typeof(a) _tmp = (a);   \
        (a) = (b);              \
        (b) = _tmp;             \
    } while (0);

/* Neighbors graph (represented as adjacency list) */
typedef struct {
    uint32_t* ids;
    uint32_t count;
    uint32_t cap;
} Neighbors;

static void neighbor_push(Neighbors* list, uint32_t id)
{
    // Rejecting duplicates
    for (uint32_t k = 0; k < list->count; k++) {
        if (id == list->ids[k]) 
            return;
    }

    // Expanding size if full
    if (list->count == list->cap) {
        list->cap *= 2; 
        list->ids = (uint32_t*) realloc(list->ids, list->cap*sizeof(uint32_t));
    }

    // Add the id of neighbor to the list
    list->ids[list->count++] = id;
}

static void neighbors_init(Neighbors* list, uint32_t cap) 
{
    uint32_t capacity = cap;
    if (cap == 0) {
        #if defined (DEBUG)
            fprintf(stderr, "Capacity can not be zero, default capacity to 32");
        #endif
        capacity = 32;
    }

    *list = (Neighbors) {
        .ids = (uint32_t*) malloc(capacity*sizeof(uint32_t)),
        .count = 0,
        .cap = capacity,
    };
}
