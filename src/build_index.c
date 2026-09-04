// Building index for a `vecfile` serialized dataset (shard)

#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "distance.h"
#include "index_format.h"
#include "vecfile.h"

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

// Medoid finder (technically not a true one but more like centroid-closest vector heuristic approximation)
static uint32_t find_medoid(VecFile* file, dist_fn_t dist_fn) {
    // Extracting file (dataset space) metadata
    uint32_t dim = file->dim;
    uint32_t n_vectors = file->num_vectors;

    // Calculate the centroid
    float* centroid = (float *)calloc(dim, sizeof(float));
    for (uint32_t k = 0; k < n_vectors; k++) {
        float* vector = vecfile_data_at(file, k);
        for (uint32_t d = 0; d < dim; d++) {
            centroid[d] += vector[d];
        }
    }
    for (uint32_t d = 0; d < dim; d++) {
        centroid[d] /= (float)n_vectors;
    }
    
    // Find the closest vector to the centroid
    float best_dist = FLT_MAX;
    uint32_t best_index = 0;
    for (uint32_t k = 0; k < n_vectors; k++) {
        float dist = dist_fn(vecfile_data_at(file, k), centroid, dim);
        if (dist < best_dist) {
            best_dist = dist;
            best_index = k;
        }
    }

    free(centroid);
    return best_index;
}

static void greedy_beam_search(VecFile* vf, Neighbors* graph, dist_fn_t dist_fn, 
                        uint32_t start_id, uint32_t beam_width, 
                        const float* query, CandidateList* candidates)
{
    /* Initialize the candidates list */
    insert_candidate(candidates, start_id, dist_fn(query, vecfile_data_at(vf, start_id), vf->dim));

    /* Process upto `beam_width` unvisited candidates per steps */
    uint32_t* batch_index = (uint32_t*) malloc(beam_width*sizeof(uint32_t));
    uint32_t batch_count = 0;

    while ((batch_count = next_unvisted_candidates(candidates, batch_index, beam_width)) > 0) {
        
        for (uint32_t b = 0; b < batch_count; b++) {
            // Mark candidates in current batch as visited
            candidates->items[batch_index[b]].visited = 1;
            
            // Collect, evaluate and try insert all neighbors across the batch
            uint32_t curr_id = candidates->items[batch_index[b]].id;
            Neighbors* neighbors = &graph[curr_id];
            for (uint32_t k = 0; k < neighbors->count; k++) {
                uint32_t nb_id = neighbors->ids[k];
                insert_candidate(candidates, nb_id, dist_fn(query, vecfile_data_at(vf, nb_id), vf->dim));
            }
        }
    }

    // Cleanups
    free(batch_index);
}

static void robust_prune()
{

}


int main(int argc, char** argv) 
{
    return 0;
}