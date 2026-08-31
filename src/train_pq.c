#include <float.h>
#include <mpi.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "distance.h"
#include "vecfile.h"
#include "option.h"

DEFINE_OPTION(uint32_t);

// Randomness helper
static inline float random_uniform() {return (float)rand() / (float)RAND_MAX;}

// KMeans++ for initalization
static void kmeans_init(
    const float* sub_data, 
    uint32_t n_points,
    uint32_t sub_dim, 
    uint32_t K, 
    dist_fn_t dist_fn, 
    float *centroids 
){
    // Initialize distance tally
    float* min_dist_sq = (float*) malloc(n_points * sizeof(float));
    for (uint32_t k = 0; k < n_points; k++) {
        min_dist_sq[k] = FLT_MAX;
    }

    // Append a randomly selected starting centroid data
    size_t first_centroid_idx = (size_t)(random_uniform() * n_points);
    memcpy(&centroids[0], &sub_data[first_centroid_idx * sub_dim], sub_dim * sizeof(float));

    // Selecting the remaining centroids
    for (size_t c = 1; c < K; c++) {
        // For each point, compute squared distance to nearest selected centroid and parallel reduce the total nearest distance
        const float *latest_centroid = &centroids[(c-1) * sub_dim];
        float total_dist = 0.0f;

        #pragma omp parallel for reduction(+:total_dist)
        for (size_t k = 0; k < n_points; k++) {
            float d = dist_fn(&sub_data[k * sub_dim], latest_centroid, sub_dim);
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
        
        memcpy(&centroids[c*sub_dim], &sub_data[next_centroid_idx*sub_dim], sub_dim * sizeof(float));
    }

    // Cleanups
    free(min_dist_sq);
}

// Lloyd KMeans algorithm for one subspace 
static inline void kmeans_lloyd(
    const float *sub_data, 
    uint32_t n_points, 
    uint32_t sub_dim,                      
    uint32_t K, 
    dist_fn_t dist_fn, 
    float *centroids, 
    uint32_t iterations,
    float* tolerance)
{
    // Creating buffers
    uint32_t* assignments = (uint32_t*) malloc(n_points*sizeof(uint32_t));
    float* cluster_sums = (float*) malloc(K*sub_dim*sizeof(float));
    uint32_t* cluster_size = (uint32_t*) malloc(K*sizeof(uint32_t));

    // Running through the iteration trainings
    for (uint32_t iter = 0; iter < iterations; iter++) {
        /* Cluster assignment (parallelized) */
        // Finding closest centroid for every point in the dataset
        #pragma omp parallel for 
        for (uint32_t i = 0; i < n_points; i++) {
            float closest_dist = FLT_MAX;
            uint32_t closest_idx = 0;

            const float* point = &sub_data[(size_t)i * sub_dim];
            for (uint32_t k = 0; k < K; k++) {
                const float* curr_centroid = &centroids[(size_t)k * sub_dim];
                float d = dist_fn(point, curr_centroid, sub_dim);

                if (d < closest_dist) {
                    closest_dist = d;
                    closest_idx = k;
                }
            }

            assignments[i] = closest_idx;
        }

        /* Update steps */
        // Reset tally buffers
        memset(cluster_sums, 0, K*sub_dim*sizeof(float));
        memset(cluster_size, 0, K * sizeof(uint32_t));

        for (uint32_t point_idx = 0; point_idx < n_points; point_idx++) {
            // Get cluster and point data
            uint32_t cluster_id = assignments[point_idx];
            const float* point = &sub_data[(size_t)point_idx * sub_dim];

            // Get the sum accumulator for the cluster
            float *sum = &cluster_sums[(size_t)cluster_id * sub_dim];

            for (uint32_t dim_idx = 0; dim_idx < sub_dim; dim_idx++) {
                sum[dim_idx] += point[dim_idx];
            }

            // Record cluster size hit
            cluster_size[cluster_id]++;
        }

        for (uint32_t k = 0; k < K; k++) {
            // Leave empty centroid where it was
            if (cluster_size[k] == 0) { 
                continue;
            }

            // Update centroid
            float* centroid = &centroids[(size_t)k * sub_dim];
            float* sum = &cluster_sums[(size_t)k * sub_dim];
            
            for (uint32_t dim_idx = 0; dim_idx < sub_dim; dim_idx++) {
                centroid[dim_idx] = sum[dim_idx] / (float)cluster_size[k];
            }
        }

        // TODO: Convergence check
    }

    // Cleanups
    free(assignments);
    free(cluster_sums);
    free(cluster_size);
}

int main(int argc, char** argv)
{
    return 0;
}
