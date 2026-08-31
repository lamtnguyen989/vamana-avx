#include <float.h>
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>

#include "distance.h"
#include "vecfile.h"

// Randomness helper
static inline float random_uniform() {return (float)rand() / (float)RAND_MAX;}

static void kmeans_init(
    const float* data, 
    uint32_t n_points,
    uint32_t sub_dim, 
    uint32_t K, 
    dist_fn_t dist_fn, 
    float *centroids 
){
    // Initalize distance tally
    float* min_dist = (float*) malloc(n_points * sizeof(float));
    for (uint32_t k = 0; k < n_points; k++) {
        min_dist[k] = FLT_MAX;
    }

    // 


}

int main(int argc, char** argv)
{
    return 0;
}
