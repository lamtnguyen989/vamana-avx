// Building index for a `vecfile` serialized dataset (shard)

#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "distance.h"
#include "index_format.h"
#include "option.h"
#include "vecfile.h"

DEFINE_OPTION(uint32_t);

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

static void greedy_search(VecFile* vf, Neighbors* graph, dist_fn_t dist_fn, 
                        uint32_t start_id, const float* query, CandidateList* candidates)
{
    /* Initialize the candidates list */
    insert_candidate(candidates, start_id, dist_fn(query, vecfile_data_at(vf, start_id), vf->dim));

    /* Fills the candidate list up to list count */
    int candidate_idx;
    while ((candidate_idx = closest_unvisited_candidate(candidates)) != -1) {
        // Mark the candidate as visited
        uint32_t cur_idx = candidates->items[candidate_idx].id;
        candidates->items[candidate_idx].visited = 1;

        // Try insert to the candidate list
        Neighbors* nb = &graph[cur_idx];
        for (uint32_t k = 0; k < nb->count; k++) {
            uint32_t nb_id = nb->ids[k];
            float d = dist_fn(vecfile_data_at(vf, nb_id), query, vf->dim);
            insert_candidate(candidates, nb_id, d);
        }
    }
}

// Alpha-pruning
static void robust_prune(VecFile* vf, dist_fn_t dist_fn, uint32_t base_id,
                        Candidate* candidates, uint32_t n, uint32_t R, float alpha,
                        uint32_t* out_ids, uint32_t* out_count)
{
    // Purely gredy when alpha==1.0
    if (alpha == 1.0) {
        return;
    }

    // Bookeeping for alive candidates
    uint8_t* alive = (uint8_t*) malloc(n*sizeof(uint8_t));
    memset(alive, 1, n);

    // Make sure we are not considering the base id due it haveing distance zero
    for (uint32_t k = 0; k < n; k++) {
        if (candidates[k].id == base_id) {
            alive[k] = 0;
        }
    }

    // Pruning
    uint32_t count = 0;
    for (uint32_t k = 0; (k < n) && (count < R); k++) {
        if (!alive[k])
            continue;

        out_ids[count++] = candidates[k].id;
        const float* vec = vecfile_data_at(vf, candidates[k].id);
        for (uint32_t j = 0; j < n; j++) {
            if (!alive[j])
                continue;
            
            float dist = dist_fn(vecfile_data_at(vf, candidates[k].id), 
                                vecfile_data_at(vf, candidates[j].id),
                                vf->dim);

            if (alpha*dist <= candidates[j].dist) {
                alive[j] = 0;
            }
        }
    }

    *out_count = count;
    free(alive);
}

static void shuffle(uint32_t* arr, uint32_t n, OPTION(uint32_t) seed_opt)
{
    if (OPTION_IS_SOME(seed_opt)) {srand(OPTION_UNWRAP(seed_opt));}

    for (uint32_t k = 0; k < n; k++) {
        uint32_t rand_idx = rand() % (k+1);
        SWAP(arr[k], arr[rand_idx]);
    }
}

int main(int argc, char** argv) 
{
    // CLI parsing
    if (argc < 3) {
        fprintf(stderr, "usage: %s <vectors.vecf> <index.vecfidx> [R=32] [L=64] [alpha=1.2] [seed]\n"
                        "\n"
                        "  vectors.vecf:        Indexing data (shard)\n"
                        "  index.vecfidx:       Output index file\n"
                        "  R:                   Max out degree\n"
                        "  L:                   Search list length\n"
                        "  alpha:               Pruning distance scailing parameter\n"
                        "  seed:                Optional seed\n"
                        , argv[0]);
        return 1;
    }
    const char* vec_path = argv[1];
    const char* index_path = argv[2];
    uint32_t R = argc > 3 ? (uint32_t) atoi(argv[3]) : 32;
    uint32_t L = argc > 4 ? (uint32_t) atoi(argv[4]) : 64;
    float alpha = argc > 5 ? (float) atof(argv[5]) : 1.20;
    OPTION(uint32_t) seed_opt = argc > 6
        ? OPTION_SOME(uint32_t, (uint32_t)strtoul(argv[8], NULL, 10))
        : OPTION_NONE(uint32_t);

    // Load vectors
    VecFile vf;
    if (vecfile_load(vec_path, &vf) != 0) {
        fprintf(stderr, "Error loading data at %s", vec_path);
        return -1;
    }

    // Initialize metric
    dist_fn_t dist_fn = metric();

    // Initialize index graph
    Neighbors* graph = (Neighbors*) malloc(vf.num_vectors * sizeof(Neighbors));
    for (uint32_t k = 0; k < vf.num_vectors; k++) {
        neighbors_init(&graph[k], R + 4);
    }

    // Notify initializations metadata
    printf("Data indexing with %u vectors, dim=%u, R=%u, L=%u, alpha=%.2f\n",
            vf.num_vectors, vf.dim, R, L, alpha);


    // Find medoid
    uint32_t medoid = find_medoid(&vf, dist_fn);
    printf("Medoid at: %u\n", medoid);

    // Shuffle data to avoid bias
    uint32_t* shuffle_order = (uint32_t*) malloc(vf.num_vectors*sizeof(uint32_t));
    for (size_t k = 0; k < vf.num_vectors; k++) {shuffle_order[k] = k;}
    shuffle(shuffle_order, vf.num_vectors, seed_opt);

    // Building index
    uint32_t* out_ids = (uint32_t*) malloc(R * sizeof(uint32_t));
    for (uint32_t idx = 0; idx < vf.num_vectors; idx++) {
        uint32_t base_id = shuffle_order[idx];

        // Building candidates list and greedy search
        CandidateList candidates;
        candidate_list_init(&candidates, L);
        greedy_search(&vf, graph, dist_fn, medoid, vecfile_data_at(&vf, base_id), &candidates);

        // Pruning
        uint32_t out_count = 0;
        robust_prune(&vf, dist_fn, base_id, candidates.items, candidates.size, R, alpha, out_ids, &out_count);
        candidate_list_free(&candidates);

        // Pushing neighbors
        graph[base_id].count = 0;
        for (uint32_t k = 0; k < out_count; k++) {
            neighbor_push(&graph[base_id], out_ids[k]);
        }
    }


    // Cleanups
    free(shuffle_order);
    free(out_ids);
    for (uint32_t k = 0; k < vf.num_vectors; k++) free(graph[k].ids);
    free(graph);
    vecfile_free(&vf);

    return 0;
}