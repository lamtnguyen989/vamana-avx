// Encoding shard after training a codebook

#include <stdio.h>

#include "distance.h"
#include "vecfile.h"
#include "pq.h"

int main(int argc, char** argv) 
{
    // CLI parsing
    if (argc < 4) {
        fprintf(stderr, "usage: %s <vectors_shard_N.bin> <codebook.bin> <codes_shard_N.bin> [threads=4]\n", argv[0]);
        return 1;
    }
    const char *shard_path = argv[1];
    const char *codebook_path = argv[2];
    const char *codes_path = argv[3];
    int num_threads = argc > 4 ? atoi(argv[4]) : 4;

    // Instantiate the distance
    dist_fn_t dist_fn = metric();

    // Load the trained codebook
    PQCodebook pq;
    if (pq_codebook_load(codebook_path, &pq) != 0) {
        return 1;
    }
    printf("Loaded codebook '%s': dim=%u M=%u K=%u stamp=%016llx\n",
            codebook_path, pq.dim, pq.M, pq.K, (unsigned long long)pq.stamp);


    return 0;
}