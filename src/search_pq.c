#include <stdlib.h>
#include <stdio.h>
#include <liburing.h>
#include <mpi.h>
#include <omp.h>

#include "index_format.h"

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
                "[threads_per_rank=4] \n"
                "\n"
                "  queries.vecf         Queries to be given to the dataset.\n"
                "  K                    Number of top ranked choices.\n"
                "  index_dir/           Directory of Vamana graph index.\n"
                "  codebook.pqbook      Product quantization codebook file.\n"
                "  pq_encoding_dir/     Product quantization encodings directory (of data shards).\n"
                "  threads_per_rank     CPU threads per MPI rank (default: 4)\n"
                "\n"
                , argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    /* Rank 0 load queries and broadcast to other ranks */

    /* Cleanups */
    MPI_Finalize();

    return 0;
}   