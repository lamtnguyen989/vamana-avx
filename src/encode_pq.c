// Encoding shard after training a codebook

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "distance.h"
#include "vecfile.h"
#include "pq.h"

int main(int argc, char** argv) 
{
    // CLI parsing
    if (argc < 4) {
        fprintf(stderr, "usage: %s <vectors_shard_N.vecf> <codebook.pqbook> <codes_shard_N.pqbin> [threads=4]\n", argv[0]);
        return 1;
    }
    const char *shard_path = argv[1];
    const char *codebook_path = argv[2];
    const char *codes_path = argv[3];
    int num_threads = argc > 4 ? atoi(argv[4]) : 4;

    // Instantiate the distance
    dist_fn_t dist_fn = metric();

    // Load the data shard
    VecFile vf;
    if (vecfile_load(shard_path, &vf) != 0) {
        printf("Fail to load shard: %s\n", shard_path);
        return 1;
    }
    
    // Load the trained codebook
    PQCodebook pq;
    if (pq_codebook_load(codebook_path, &pq) != 0) {
        return 2;
    }
    printf("Loaded codebook '%s': dim=%u M=%u K=%u hash=%016llx\n",
            codebook_path, pq.dim, pq.M, pq.K, (unsigned long long)pq.hash);

    // Dimension checks
    if (vf.dim != pq.dim) {
        printf("Dimensional mismatch: Shard has dim=%u, codebook trained on dim=%u\n", vf.dim, pq.dim);
        return 1;
    }

    // Encode (note encodings are parallelizable between points)
    uint8_t *codes = (uint8_t *)malloc((size_t)vf.num_vectors * pq.M);

    omp_set_num_threads(num_threads);
    #pragma parallel for schedule(dynamic)
    for (size_t k = 0; k < vf.num_vectors; k++) {
        pq_encode(&pq, dist_fn, vecfile_data_at(&vf, k), &codes[k * pq.M]);
    }

    // Write the code serialization
    FILE *codes_file = fopen(codes_path, "wb");
    if (codes_file == NULL) { 
        perror("fopen codes file"); 
        return 1;
    }
    uint32_t magic = PQ_MAGIC;
    fwrite(&magic, sizeof(uint32_t), 1, codes_file);
    fwrite(&vf.num_vectors, sizeof(uint32_t), 1, codes_file);
    fwrite(&pq.M, sizeof(uint32_t), 1, codes_file);
    fwrite(&pq.hash, sizeof(uint64_t), 1, codes_file); /* Tie the codes file to this exact codebook */
    fwrite(codes, 1, (size_t)vf.num_vectors * pq.M, codes_file);
    fclose(codes_file);

    // Notify the writes quantities
    size_t codes_bytes = (size_t)vf.num_vectors * pq.M;
    size_t full_bytes = (size_t)vf.num_vectors * vf.dim * sizeof(float);
    printf("Wrote %s: %u points x %u bytes = %zu bytes (vs %zu bytes full-precision, %.1fx smaller), hash=%016llx\n",
            codes_path, vf.num_vectors, pq.M, codes_bytes, full_bytes, (double)full_bytes / codes_bytes,
            (unsigned long long)pq.hash);

    // Cleanups
    free(codes);
    pq_codebook_free(&pq);
    vecfile_free(&vf);

    return 0;
}