// Building Vamana index and PQ encode shards all in a single MPI executable

#include <stdio.h>
#include <stdint.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>

#include "distance.h"
#include "index_format.h"
#include "option.h"
#include "pq.h"
#include "vecfile.h"
#include "vamana.h"

#define VECFILE_HEADER_BYTES (2*sizeof(uint32_t))

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

// Globbing (per-shard) pipeline data to a configuration
typedef struct {
    const char* index_dir;
    const char* encoding_dir;
    float alpha;
    uint32_t R;
    uint32_t L;
    int n_threads;
    OPTION(uint32_t) seed_opt;
    PQCodebook* pq_codebook;
    dist_fn_t dist_fn;
} ShardJobConfig;

// Vamana index building pipeline
static void build_shard_vamana_index(VecFile* vf, const ShardJobConfig* cfg, const char* base_filename, int rank)
{
    /* Getting data from config */
    dist_fn_t dist_fn = cfg->dist_fn;
    uint32_t R = cfg->R;
    uint32_t L = cfg->L;
    float alpha = cfg->alpha;

    /* Initialize Vamana index graph and also OpenMP locks */
    Neighbors* graph = (Neighbors*) malloc(vf->num_vectors * sizeof(Neighbors));
    omp_lock_t* locks = (omp_lock_t*) malloc(vf->num_vectors * sizeof(omp_lock_t));
    for (uint32_t k = 0; k < vf->num_vectors; k++) {
        neighbors_init(&graph[k], R + 4);
        omp_init_lock(&locks[k]);
    }

    /* Find medoid */
    uint32_t medoid = find_medoid(vf, dist_fn, cfg->n_threads);

    // Shuffle data to avoid bias
    uint32_t* shuffle_order = (uint32_t*) malloc(vf->num_vectors * sizeof(uint32_t));
    for (uint32_t k = 0; k < vf->num_vectors; k++) { shuffle_order[k] = k; }
    shuffle(shuffle_order, vf->num_vectors, cfg->seed_opt);

    /* Building index */
    time_t t0 = time(NULL);
    #pragma omp parallel num_threads(cfg->n_threads)
    {
        uint32_t* out_ids = (uint32_t*) malloc(R * sizeof(uint32_t));
        Candidate* scratch = (Candidate *)malloc((L + R + 1) * sizeof(Candidate));  // Scratch buffer for re-pruning overflowed nodes
        uint32_t* nbr_scratch = (uint32_t*) malloc(R*sizeof(uint32_t)); // Local neighbor id scratch

        #pragma omp for schedule(dynamic)
        for (uint32_t idx = 0; idx < vf->num_vectors; idx++) {
            uint32_t base_id = shuffle_order[idx];

            // Building candidates list and greedy search
            CandidateList candidates;
            candidate_list_init(&candidates, L);
            greedy_search(vf, graph, dist_fn, medoid, vecfile_data_at(vf, base_id), locks, nbr_scratch, &candidates);

            // Pruning
            uint32_t out_count = 0;
            robust_prune(vf, dist_fn, base_id, candidates.items, candidates.size, R, alpha, out_ids, &out_count);
            candidate_list_free(&candidates);

            // Pushing neighbors (note critical writes)
            omp_set_lock(&locks[base_id]);
            graph[base_id].count = 0;
            for (uint32_t k = 0; k < out_count; k++) {
                neighbor_push(&graph[base_id], out_ids[k]);
            }
            omp_unset_lock(&locks[base_id]);

            // Reverse edges and prune nodes that has more than R neighbors (every iteration is critical writes)
            for (uint32_t k =0; k< out_count; k++) {
                // Acquire lock
                omp_set_lock(&locks[out_ids[k]]);

                // Reversing edges
                neighbor_push(&graph[out_ids[k]], base_id);
                
                // Re-deriving candidate list and prune for nodes has more than R neighbors
                if (graph[out_ids[k]].count > R) {
                    candidates_from_neighbors(vf, dist_fn, out_ids[k], &graph[out_ids[k]], scratch);
                    uint32_t new_count = 0;
                    uint32_t* new_ids = (uint32_t*) malloc(R * sizeof(uint32_t));
                    robust_prune(vf, dist_fn, out_ids[k], scratch, graph[out_ids[k]].count, R, alpha, new_ids, &new_count);
                    graph[out_ids[k]].count = 0;
                    for (uint32_t j = 0; j < new_count; j++) {
                        neighbor_push(&graph[out_ids[k]], new_ids[j]);
                    }
                    free(new_ids);
                }
                // Release lock
                omp_unset_lock(&locks[out_ids[k]]);
            }
        }

        free(nbr_scratch);
        free(scratch);
        free(out_ids);
    }
    printf("Done with building index, took %lds. Now serializing...\n", time(NULL) - t0);

    // Cleanup locks before serializing since everything is serial from here
    for (uint32_t k = 0; k < vf->num_vectors; k++) { omp_destroy_lock(&locks[k]); }
    free(locks);

    /* Serialize index */
    char out_path[2048];
    snprintf(out_path, sizeof(out_path), "%s/%s.vamindx", cfg->index_dir, base_filename);
    FILE* vamana_out = fopen(out_path, "wb");
    if (vamana_out == NULL) {
        perror("Failed to open file to export the computed Vamana index\n");
        exit(1);
    }

    IndexHeader hdr;

    // Cleanups
    for (uint32_t k = 0; k < vf->num_vectors; k++) { free(graph[k].ids); }
    free(graph);
    free(shuffle_order);
}


// Shard encoding pipeline
static void encode_shard(VecFile* vf, const ShardJobConfig* cfg, const char* base_filename, int rank)
{
    PQCodebook* pq = cfg->pq_codebook;

    // Dimension checks
    if (vf->dim != pq->dim) {
        fprintf(stderr, "Dimensional mismatch: Shard has dim=%u, codebook trained on dim=%u\n", vf->dim, pq->dim);
        return;
    }

    // Make space to store the encodings
    uint8_t* encodings = (uint8_t*) malloc(vf->num_vectors * pq->M * sizeof(uint8_t));
    if (encodings == NULL) {
        fprintf(stderr, "Rank %d: Encodings buffer failed to allocate for shard %s", rank, base_filename);
    }

    // Encode (parallelized across vectors in dataset)
    #pragma omp parallel for num_threads(cfg->n_threads)
    for (uint32_t k = 0; k < vf->num_vectors; k++) {
        pq_encode(pq, cfg->dist_fn, vecfile_data_at(vf, k), &encodings[k * pq->M * sizeof(uint8_t)]);
    }

    // Serialize the encodings
    char out_path[2048];
    snprintf(out_path, sizeof(out_path), "%s/%s.pqbin", cfg->encoding_dir, base_filename);
    FILE* encodings_file = fopen(out_path, "wb");
    if (encodings_file == NULL) {
        perror("encoding_shard: fopen");
        return;
    }
    uint32_t magic = PQ_MAGIC;
    fwrite(&magic, sizeof(uint32_t), 1, encodings_file);
    fwrite(&vf->num_vectors, sizeof(uint32_t), 1, encodings_file);
    fwrite(&pq->M, sizeof(uint32_t), 1, encodings_file);
    fwrite(&pq->hash, sizeof(uint64_t), 1, encodings_file); /* Tie the codes file to this exact codebook */
    fwrite(encodings, 1, vf->num_vectors * pq->M * sizeof(uint8_t), encodings_file);

    // Notify
    printf("Rank %d: Wrote PQ encodings with hash=%016lx to %s\n", rank, pq->hash, out_path);

    // Cleanups
    free(encodings);
    fclose(encodings_file);
}


int main(int argc, char** argv) 
{
    /* Start MPI multi-threaded environment */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided != MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation doesn't support MPI_THREAD_FUNNELED (got %d)\n", provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Initializing config
    dist_fn_t dist_fn = metric();



    /* Cleanups */
    MPI_Finalize();
    return 0;
}