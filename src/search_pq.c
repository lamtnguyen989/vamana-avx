#include <stdlib.h>
#include <stdio.h>
#include <liburing.h>
#include <mpi.h>
#include <omp.h>

#include "index_format.h"
#include "vecfile.h"
#include "pq.h"

#define URING_QUEUE_DEPTH 128
#define RERANK_POOL 128

/* Singleton uring context for each thread (could have used "option.h" here but black-magic is not worth it) */
typedef struct {
    struct io_uring ring;
    int ready;
} UringContext;

static __thread UringContext thread_uring_context = {0};

// Get and initialize thread-local uring
static struct io_uring* get_thread_uring()
{
    if (!thread_uring_context.ready) {
        if (io_uring_queue_init(URING_QUEUE_DEPTH, &thread_uring_context.ring, 0) < 0) {
            perror("io_uring_queue_init");
            exit(1);
        }
        thread_uring_context.ready = 1;
    }

    return &thread_uring_context.ring;
}

// Doing 1 query beam search
static void beam_search()
{

}


int main(int argc, char** argv)
{
    /* Start MPI multi-threaded environment */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation doesn't support MPI_THREAD_FUNNELED (got %d)\n", provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Usual MPI setup */
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* CLI parsing */
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: mpirun -n <ranks> %s <queries.vecf> <K> <index_dir/> <codebook.pqbook> <pq_encoding_dir/> "
                "[L=64] [beam_width=8] [threads_per_rank=4] \n"
                "\n"
                "  queries.vecf         Queries for search in the dataset.\n"
                "  K                    Number of top ranked choices.\n"
                "  index_dir/           Directory of Vamana graph index.\n"
                "  codebook.pqbook      Product quantization codebook file.\n"
                "  pq_encoding_dir/     Product quantization encodings directory (of data shards).\n"
                "  L                    Search candidate list size (default: 64)\n"
                "  beam_width           Batch-size for beam search (default: 8)\n"
                "  threads_per_rank     CPU threads per MPI rank (default: 4)\n"
                "\n"
                , argv[0]);
            MPI_Finalize();
        }
        return 1;
    }
    const char* query_path = argv[1];
    uint32_t K = (uint32_t)atoi(argv[2]);
    const char* index_dir = argv[3];
    const char* codebook_path = argv[4];
    const char* pq_dir = argv[5];
    int n_threads = argc > 6 ? atoi(argv[6]) : 4;
    uint32_t L = argc > 7 ? (uint32_t)atoi(argv[7]) : 64;
    uint32_t beam_width = argc > 8 ? (uint32_t)atoi(argv[8]) : 8; 

    /* Rank 0 load queries and broadcast to other ranks */
    uint32_t n_queries = 0;
    uint32_t dim = 0;
    float* queries = NULL;
    if (rank == 0) {
        // Load query file
        VecFile qvf;
        if (vecfile_load(query_path, &qvf) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        // Extracting query information
        n_queries = qvf.num_vectors;
        dim = qvf.dim;
        queries = qvf.data;
    }
    // Broadcast
    MPI_Bcast(&n_queries, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        // Make space in other ranks for broadcasting the data
        queries = (float*) malloc(n_queries*dim*sizeof(float));
    } 
    MPI_Bcast(&queries, (int)(n_queries*dim), MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Every rank load a global pq codebook (better ways possible, but simplicity for now) */
    PQCodebook codebook;
    if (pq_codebook_load(codebook_path, &codebook) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    /* Partition the shard indexing work */
    

    /* Cleanups */
    free(queries); // Techically a potential memory hazard for rank 0 queries but all vecfile except for data is stack-allocated.

    MPI_Finalize();

    return 0;
}   