// Building Vamana index and PQ encode shards all in a single MPI executable

#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#include "distance.h"
#include "index_format.h"
#include "option.h"
#include "pq.h"
#include "vecfile.h"
#include "vamana.h"

DEFINE_OPTION(uint32_t);

// Medoid finder (technically not a true one but more like centroid-closest vector heuristic approximation)
static uint32_t find_medoid(VecFile* file, dist_fn_t dist_fn, int n_threads)
{
    uint32_t dim = file->dim;
    uint32_t n_vectors = file->num_vectors;

    float* centroid = (float*) calloc(dim, sizeof(float));
    #pragma omp parallel for num_threads(n_threads) reduction(+:centroid[:dim])
    for (uint32_t k = 0; k < n_vectors; k++) {
        float* vector = vecfile_data_at(file, k);
        for (uint32_t d = 0; d < dim; d++) {
            centroid[d] += vector[d];
        }
    }
    for (uint32_t d = 0; d < dim; d++) {
        centroid[d] /= (float) n_vectors;
    }

    float best_dist = FLT_MAX;
    uint32_t best_index = 0;
    #pragma omp parallel num_threads(n_threads)
    {
        uint32_t local_best_idx = 0;
        float local_best_dist = FLT_MAX;

        #pragma omp for nowait
        for (uint32_t k = 0; k < n_vectors; k++) {
            float dist = dist_fn(vecfile_data_at(file, k), centroid, dim);
            if (dist < local_best_dist) {
                local_best_dist = dist;
                local_best_idx = k;
            }
        }
        #pragma omp critical
        {
            if (local_best_dist < best_dist) {
                best_dist = local_best_dist;
                best_index = local_best_idx;
            }
        }
    }

    free(centroid);
    return best_index;
}

static uint32_t snapshot_neighbors(Neighbors* graph, omp_lock_t* locks, uint32_t node_id, uint32_t* out_snapshot)
{
    omp_set_lock(&locks[node_id]);
    uint32_t count = graph[node_id].count;
    memcpy(out_snapshot, graph[node_id].ids, count*sizeof(uint32_t));
    omp_unset_lock(&locks[node_id]);
    
    return count;
}

// Greedy search
static void greedy_search(VecFile* vf, 
                        Neighbors* graph, 
                        dist_fn_t dist_fn, 
                        uint32_t start_id, 
                        const float* query,
                        omp_lock_t* locks, 
                        uint32_t* nbr_idx_snapshot,
                        CandidateList* candidates)
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
        uint32_t degree = snapshot_neighbors(graph, locks, cur_idx, nbr_idx_snapshot);
        for (uint32_t k = 0; k < degree; k++) {
            uint32_t nb_id = nbr_idx_snapshot[k];
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
    // Bookeeping for alive candidates
    uint8_t alive[n];
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
        float* vec1 = vecfile_data_at(vf, candidates[k].id);
        for (uint32_t j = 0; j < n; j++) {
            if (!alive[j])
                continue;
            
            float* vec2 = vecfile_data_at(vf, candidates[j].id);
            float dist = dist_fn(vec1, vec2, vf->dim);

            if (alpha*dist <= candidates[j].dist) {
                alive[j] = 0;
            }
        }
    }

    *out_count = count;
}

static void shuffle(uint32_t* arr, uint32_t n, OPTION(uint32_t) seed_opt)
{
    if (OPTION_IS_SOME(seed_opt)) {srand(OPTION_UNWRAP(seed_opt));}

    for (uint32_t k = 0; k < n; k++) {
        uint32_t rand_idx = rand() % (k+1);
        SWAP(arr[k], arr[rand_idx]);
    }
}

// Re-deriving candidates list from neighbors adjacency list
static void candidates_from_neighbors(const VecFile* vf, dist_fn_t dist_fn, 
                                    uint32_t pt_id, Neighbors* nb, Candidate* out)
{
    // Initialize the candidates
    const float* pt = vecfile_data_at(vf, pt_id);
    for (uint32_t k = 0; k < nb->count; k++) {
        out[k] = (Candidate) {
            .id = nb->ids[k],
            .dist = dist_fn(pt, vecfile_data_at(vf, nb->ids[k]), vf->dim),
            .visited = 0,
        };
    }

    // Insertion sort by dist (list is small, bounded by R)
    for (int k = 1; k < nb->count; k++) {
        Candidate key = out[k];
        int j = k - 1;
        while (j >= 0 && out[j].dist > key.dist) { 
            out[j + 1] = out[j];
            j--; 
        }
        out[j + 1] = key;
    }
}


int main(int argc, char** argv) 
{
    return 0;
}