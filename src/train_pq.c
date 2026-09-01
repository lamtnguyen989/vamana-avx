#include <float.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "distance.h"
#include "vecfile.h"
#include "option.h"
#include "pq.h"

DEFINE_OPTION(uint32_t);

// Randomness helper
static inline float random_uniform() {return (float)rand() / (float)RAND_MAX;}

// KMeans++ for initalization
static void kmeans_init(
    const float* data, 
    uint32_t n_points,
    uint32_t dim, 
    uint32_t K, 
    dist_fn_t dist_fn, 
    float* centroids)
{
    // Initialize distance tally
    float* min_dist_sq = (float*) malloc(n_points * sizeof(float));
    for (uint32_t k = 0; k < n_points; k++) {
        min_dist_sq[k] = FLT_MAX;
    }

    // Append a randomly selected starting centroid data
    size_t first_centroid_idx = (size_t)(random_uniform() * n_points);
    memcpy(&centroids[0], &data[first_centroid_idx * dim], dim * sizeof(float));

    // Selecting the remainderaining centroids
    for (size_t c = 1; c < K; c++) {
        // For each point, compute squared distance to nearest selected centroid and parallel reduce the total nearest distance
        const float *latest_centroid = &centroids[(c-1) * dim];
        float total_dist = 0.0f;

        #pragma omp parallel for reduction(+:total_dist)
        for (size_t k = 0; k < n_points; k++) {
            float d = dist_fn(&data[k * dim], latest_centroid, dim);
            float d_squared = square(d);
            if (d < min_dist_sq[k]) {
                min_dist_sq[k] = d_squared;
            }

            total_dist += min_dist_sq[k];
        }

        // Choose next centroid with probability proportional to distance squared
        float cum_dist = 0.0f;
        float threshold_dist  = random_uniform() * total_dist;
        uint32_t next_centroid_idx = n_points - 1;

        for (uint32_t k = 0; k < n_points; k++) {
            cum_dist += min_dist_sq[k];
            if (cum_dist >= threshold_dist) {
                next_centroid_idx = k;
                break;
            }
        }
        
        memcpy(&centroids[c*dim], &data[next_centroid_idx*dim], dim * sizeof(float));
    }

    // Cleanups
    free(min_dist_sq);
}

// Lloyd KMeans algorithm for one subspace 
static inline void kmeans_lloyd(
    const float *data, 
    uint32_t n_points, 
    uint32_t dim,                      
    uint32_t K, 
    dist_fn_t dist_fn, 
    float* centroids, 
    uint32_t iterations,
    float tolerance)
{
    // Creating buffers
    uint32_t* assignments = (uint32_t*) malloc(n_points*sizeof(uint32_t));
    float* cluster_sums = (float*) malloc(K*dim*sizeof(float));
    uint32_t* cluster_size = (uint32_t*) malloc(K*sizeof(uint32_t));

    // Running through the iteration trainings
    for (uint32_t iter = 0; iter < iterations; iter++) {
        /* Cluster assignment (parallelized) */
        // Finding closest centroid for every point in the dataset
        #pragma omp parallel for 
        for (uint32_t i = 0; i < n_points; i++) {
            float closest_dist = FLT_MAX;
            uint32_t closest_idx = 0;

            const float* point = &data[(size_t)i * dim];
            for (uint32_t k = 0; k < K; k++) {
                const float* curr_centroid = &centroids[(size_t)k * dim];
                float d = dist_fn(point, curr_centroid, dim);

                if (d < closest_dist) {
                    closest_dist = d;
                    closest_idx = k;
                }
            }

            assignments[i] = closest_idx;
        }

        /* Update steps */
        // Reset tally buffers
        memset(cluster_sums, 0, K * dim * sizeof(float));
        memset(cluster_size, 0, K * sizeof(uint32_t));

        for (uint32_t point_idx = 0; point_idx < n_points; point_idx++) {
            // Get cluster and point data
            uint32_t cluster_id = assignments[point_idx];
            const float* point = &data[(size_t)point_idx * dim];

            // Get the sum accumulator for the cluster
            float *sum = &cluster_sums[(size_t)cluster_id * dim];

            for (uint32_t dim_idx = 0; dim_idx < dim; dim_idx++) {
                sum[dim_idx] += point[dim_idx];
            }

            // Record cluster size hit
            cluster_size[cluster_id]++;
        }

        // Update centroid and convergence check
        float max_shift_sq = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            // Leave empty centroid where it was
            if (cluster_size[k] == 0) { 
                continue;
            }

            float* centroid = &centroids[(size_t)k * dim];
            float* sum = &cluster_sums[(size_t)k * dim];
            float shift_sq = 0.0f;

            for (uint32_t dim_idx = 0; dim_idx < dim; dim_idx++) {
                // Calculate the shift of the new cluster means with respect to the old centroid
                float new_val = sum[dim_idx] / (float)cluster_size[k];
                shift_sq += square(new_val - centroid[dim_idx]);

                // Update the centroid 
                centroid[dim_idx] = new_val;

                // Update max global shift value
                if (shift_sq > max_shift_sq) {
                    max_shift_sq = shift_sq;
                }
            }

            // Early stopping if no centroid move more than tolerance
            if (tolerance > 0.0f && max_shift_sq < tolerance) {
                break;
            }
        }
    }

    // Cleanups
    free(assignments);
    free(cluster_sums);
    free(cluster_size);
}

int main(int argc, char** argv)
{
    /* Initalize MPI */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* CLI parsing (very basic) */
    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: mpirun -n <ranks> %s <sample.bin> <codebook.bin> "
                "[M=16] [K=256] [iters=30] [threads_per_rank=4] [epsilon=0.0] [seed]\n"
                "\n"
                "  sample.bin       Representative sample from the full dataset\n"
                "  codebook.bin     Output codebook file\n"
                "  M                Number of PQ subspaces (default: 16)\n"
                "  K                Centroids per subspace (default: 256), maximum allowed value: 256 \n"
                "  iters            K-means iterations (default: 30)\n"
                "  threads_per_rank CPU threads per MPI rank (default: 4)\n"
                "  epsilon          Tolerance for convergence consideration (default: 0.0)\n"
                "  seed             Optional reproduciblity initialization seed\n",
                argv[0]);

        }
        MPI_Finalize();
        return 1;
    }
    const char *vec_path = argv[1];
    const char *codebook_path = argv[2];
    uint32_t M = argc > 3 ? (uint32_t)atoi(argv[3]) : 16;
    uint32_t K = argc > 4 ? (uint32_t)atoi(argv[4]) : 256;
    if (K > 256) { 
        fprintf(stderr, "K must be <= 256 (codes are bytes)\n"); 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    uint32_t iters = argc > 5 ? (uint32_t)atoi(argv[5]) : 30;
    int threads_per_rank = argc > 6 ? atoi(argv[6]) : 4;
    if (threads_per_rank > 1) {
        if (rank == 0) {
            fprintf(stderr, "Since floating-point arithmetics are non-associative, running with more than 1 threads are non-determistic! "
                            "But overall values should be the similar.\n");
        }
    }
    float epsilon = argc > 7 ? atof(argv[7]) : 0.0f;

    OPTION(uint32_t) seed_opt = argc > 8
        ? OPTION_SOME(uint32_t, (uint32_t)strtoul(argv[8], NULL, 10))
        : OPTION_NONE(uint32_t);

    // Extra warnings
    if (M < world_size && rank == 0) {
        fprintf(stderr, "Since M=%u < world_size=%d, some ranks will get zero subspaces "
                        "and sit idle; consider running with fewer ranks or a larger M\n", M, world_size);
    }

    /* Reading the sample as rank 0 and broadcast (small dataset, relatively speaking) */
    uint32_t n_vectors = 0;
    uint32_t dim = 0;
    float* data = NULL;
    if (rank == 0) {
        // Loading sample from the vecfile serialization
        VecFile vf;
        if (vecfile_load(argv[1], &vf) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        n_vectors = vf.num_vectors;
        dim = vf.dim;
        data = vf.data;
    }
    // Broadcast data to other
    MPI_Bcast(&n_vectors, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        data = (float*) malloc(n_vectors*dim*sizeof(float));
    }
    MPI_Bcast(data, n_vectors * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Determining quantization sub-dimension */
    if (dim % M != 0) {
        if (rank == 0) {
            fprintf(stderr, "dim (%u) must be divisible by M (%u)\n", dim, M);
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    uint32_t subspace_dim = dim / M;

    /* Set the OpenMP threads per rank and notify training metadata */
    omp_set_num_threads(threads_per_rank);
    if (rank == 0) {
        printf("Training with %u sample points, dim=%u, M=%u subspaces, subspace_dim=%u, K=%u centroids/subspace, %d ranks and %d threads with %.8f tolerance.\n",
                n_vectors, dim, M, subspace_dim, K, world_size, threads_per_rank, epsilon);
    }

    /* Scatter subspaces across ranks and each rank train to completion */
    // Randomization initialization for each rank
    if (OPTION_IS_SOME(seed_opt)) {
        if (rank == 0) {
            printf("Training with random base seed: %u\n", seed_opt.value);
        }
        srand(seed_opt.value + rank);
    } else {
        if (rank == 0) {
            printf("No seed provided, this training might not be reproducible.");
        }
    }

    // Partition the work
    size_t base = (size_t)M / world_size;
    size_t remainder = (size_t)M % world_size;

    size_t start = rank*base + (rank < remainder? rank : remainder);
    size_t subspace_count = base + (rank < remainder ? 1 : 0);

    // Notifying the subspace training range
    printf("Rank %d: Training subspace %ld to subspace %ld\n", rank, start, (start + subspace_count));

    // Initializing training data
    float* local_centroids = (float*) malloc((size_t)subspace_count * K * subspace_dim * sizeof(float));
    float* subspace_data = (float*) malloc((size_t)n_vectors * subspace_dim * sizeof(float));
    
    dist_fn_t dist_fn = metric();

    // Actual training loop
    time_t t0 = time(NULL);
    for (size_t local_m = 0; local_m < subspace_count; local_m++) {
        size_t m = start + local_m;
        for (size_t i = 0; i < n_vectors; i++) {
            memcpy(&subspace_data[i*subspace_dim], &data[i*dim + m*subspace_dim], subspace_dim * sizeof(float));
        }
        float *centroids_m = &local_centroids[(size_t)local_m * K * subspace_dim];
        kmeans_init(subspace_data, n_vectors, subspace_dim, K, dist_fn, centroids_m);
        kmeans_lloyd(subspace_data, n_vectors, subspace_dim, K, dist_fn, centroids_m, iters, epsilon);
        printf("Rank %d: subspace %lu done (%lds elapsed)\n", rank, m, time(NULL) - t0);
    }
    
    /* Gather finished centroid back to rank 0 */
    // Setting up booking keeping at rank 0
    int* recvcounts = NULL;
    int* displs = NULL;
    float* recv_centroids = (float*) malloc(M*K*subspace_dim *sizeof(float));

    if (rank == 0) {
        recvcounts = (int*) malloc(world_size * sizeof(int));
        displs = (int*) malloc(world_size * sizeof(int));

        for (int r = 0; r < world_size; r++) {
            uint32_t cnt = base + ((uint32_t)r < remainder ? 1 : 0);
            uint32_t start = (uint32_t)r * base + ((uint32_t)r < remainder ? (uint32_t)r : remainder);
            recvcounts[r] = (int)(cnt * K * subspace_dim);
            displs[r] = (int)(start * K * subspace_dim);
        }
    }

    // Execute the gather
    MPI_Gatherv(local_centroids, (int)(subspace_count * K * subspace_dim), MPI_FLOAT,
                recv_centroids, recvcounts, displs,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Serialize the codebook (at rank 0)
    if (rank == 0) {
        // Initalize codebook data
        PQCodebook pq = {
            .centroids = recv_centroids,
            .hash = 0,
            .dim = dim,
            .M = M,
            .sub_dim = subspace_dim,
            .K = K,
        };

        // Serialize
        if (pq_codebook_save(codebook_path, &pq) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 4);
        }

        // Notify the trained codebook
        printf("Wrote the codebook with stamp: %016llx\n", (unsigned long long)pq.hash);

        // Cleanup the book keeping
        pq_codebook_free(&pq);
        free(recvcounts);
        free(displs);
    }


    /* Cleanups */
    free(local_centroids);
    free(subspace_data);
    MPI_Finalize();

    return 0;
}
