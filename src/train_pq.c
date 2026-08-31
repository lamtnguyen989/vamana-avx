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
    const float* data, 
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
    memcpy(&centroids[0], &data[first_centroid_idx * sub_dim], sub_dim * sizeof(float));

    // Selecting the remaining centroids
    for (size_t c = 1; c < K; c++) {
        // For each point, compute squared distance to nearest selected centroid (note parallelizable)
        const float *latest_centroid = &centroids[(c-1) * sub_dim];
        float total_dist = 0.0f;

        #pragma omp parallel for reduction(+:total_dist)
        for (size_t k = 0; k < n_points; k++) {
            float d = dist_fn(&data[k * sub_dim], latest_centroid, sub_dim);
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
        
        memcpy(&centroids[c*sub_dim], &data[next_centroid_idx*sub_dim], sub_dim * sizeof(float));
    }

    // Cleanups
    free(min_dist_sq);
}

static inline float kmeans()
{
    return 0.0f;
}

int main(int argc, char** argv)
{
    return 0;
}
