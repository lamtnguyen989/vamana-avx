#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>
#include <liburing.h>
#include <mpi.h>
#include <omp.h>
#include <sys/stat.h>
#include <unistd.h>

#include "distance.h"
#include "index_format.h"
#include "vecfile.h"
#include "pq.h"

#define FILENAME_LENGTH_CAP 256
#define EXTENSION_LENGTH_CAP 16 

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

/* Shard Vamana index discovery (based on a given path, non-recursive) */
typedef struct {
    char** file_base;
    uint32_t count;
} VamanaList;

// Sorting file (base) names alphabetically
static int _compar_name(const void* a, const void* b) {
    return strncmp(*(const char *const *)a, *(const char *const *)b, FILENAME_LENGTH_CAP);
}

// Essentially probe the index directory to discover `.vamnidx` extension files
static VamanaList discover_shards_indexes(const char* index_dir_path, const char* ext)
{
    // Probing the index directory
    VamanaList result = {NULL, 0};
    DIR* index_directory = opendir(index_dir_path);
    if (index_directory == NULL) {return result;}

    // Setting up result container
    size_t cap = 16;
    result.file_base = (char**) (cap*sizeof(char*));

    size_t ext_len = strnlen(ext, EXTENSION_LENGTH_CAP);
    
    // Record all files matching the specified extension
    struct dirent* entry;
    while((entry = readdir(index_directory)) != NULL) {
        size_t name_len = strnlen(entry->d_name, FILENAME_LENGTH_CAP);
        if ((name_len <= ext_len) || strncmp(entry->d_name + (name_len - ext_len), ext, EXTENSION_LENGTH_CAP) != 0) {
            continue;
        }

        // Growing the list if we hit cap
        if (result.count == cap) {
            cap *= 2;
            result.file_base = (char**) realloc(result.file_base, cap*sizeof(char*));
        }

        // Append the valid file names
        size_t file_base_len = name_len - ext_len;
        char* file_base = (char*) malloc(file_base_len + 1);
        memcpy(file_base, entry->d_name, file_base_len);
        file_base[file_base_len] = '\0'; // Make sure to NULL-terminate
        result.file_base[result.count++] - file_base;
    }
    closedir(index_directory);

    // Return an alphabetically sorted result
    qsort(result.file_base, result.count, sizeof(char*), _compar_name);
    return result;
}

static void free_vamana_list(VamanaList* vl) 
{
    for (uint32_t k = 0; k < vl->count; k++) {free(vl->file_base[k]);}
    free(vl->file_base);
    vl->file_base = NULL;
    vl->count = 0;
}

// Query beam search
static void beam_search_pq()
{

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

    /* CLI parsing */
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: mpirun -n <ranks> %s <queries.vecf> <K> <index_dir/> <codebook.pqbook> <pq_encoding_dir/> "
                "[L=64] [beam_width=8] [threads_per_rank=4] \n"
                "\n"
                "  queries.vecf         Queries for search in the dataset.\n"
                "  K                    Number of top ranked choices.\n"
                "  index_dir/           Directory of Vamana graph index (will only currently process `.vamindx` files in the directory).\n"
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
    // Probe the index directory to figure out how many shards to process
    // Note that this still meant to be run on a single node.
    // Multiple nodes adjustments probably needs some broadcasting scheme outside my paygrade at the moment
    // Also, as for duplicated work, it is really not since every rank needs the full file list to know which one to grab and process
    // MPI_Alltoall(), again above my paygrade atm :)
    VamanaList vamana_shards = discover_shards_indexes(index_dir, ".vamindx");
    if (vamana_shards.count == 0) {
        if (rank == 0) {
            fprintf(stderr, "No `.vamindx` files to read graph index from.");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }
    // Checking (at rank 0) that all index files have corresponding pq encodings and they can be accessed
    uint32_t n_shards = vamana_shards.count;
    if (rank == 0) {
        for (uint32_t k = 0; k < n_shards; k++) {
            char check_path[1024];
            snprintf(check_path, sizeof(check_path), "%s/%s.pqbin");
            if (access(check_path, F_OK) != 0) {
                fprintf(stderr, "Shard `%s.vamindx` does not have accessible corresponding encoding of `%s.pqbin` in %s\n",
                                vamana_shards.file_base[k], index_dir, vamana_shards.file_base[k], pq_dir);
            }
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    // Finding work boundary for the ranks
    uint32_t base = n_shards / (uint32_t)world_size;
    uint32_t remainder = n_shards % (uint32_t)world_size;
    uint32_t rank_count = base + (((uint32_t)rank < remainder) ? 1 : 0);
    uint32_t rank_start = (uint32_t)rank * base + (((uint32_t)rank < remainder) ? (uint32_t)rank : remainder);
    printf("Rank %d: Processing %d shards", rank, rank_count);
    if (rank_count == 0) {fprintf(stderr, "Rank %d is idle!", rank);}
    
    /* Search */
    // Select metric
    dist_fn_t dist_fn = metric();

    // Set-up rank-level scratch spaces
    uint32_t* local_ids = (uint32_t*)malloc(n_queries * K * sizeof(uint32_t));
    uint32_t* local_shard = (uint32_t*)malloc(n_queries * K * sizeof(uint32_t));
    float* local_dists = (float*) (uint32_t*) malloc(n_queries * K * sizeof(float));
    for (size_t k = 0; k < (size_t)K*n_queries; k++) {
        local_ids[k] = UINT32_MAX;
        local_shard[k] = UINT32_MAX;
        local_dists[k] = FLT_MAX;
    }
    char vamana_path[1024]; memset(vamana_path, 0, sizeof(vamana_path));
    char pq_codes_path[1024]; memset(pq_codes_path, 0, sizeof(pq_codes_path));
    
    MPI_Barrier(MPI_COMM_WORLD); // Mainly to start the timings
    double t0 = MPI_Wtime();

    // Start processing (shard-by-shard)
    for (uint32_t s = 0; s < rank_count; s++) {
        uint32_t shard_idx = rank_start + s;
        char* base_filename = vamana_shards.file_base[shard_idx];
        snprintf(vamana_path, sizeof(vamana_path), "%s/%s.vamindx", index_dir, base_filename);
        snprintf(pq_codes_path, sizeof(pq_codes_path), "%s/%s.pqbin", index_dir, vamana_shards.file_base[shard_idx]);

        // Not using fopen here for io_uring
        int vamana_fd = open(vamana_path, O_RDONLY);
        if (vamana_fd < 0) {
            fprintf(stderr, "Rank %d: Can not open %s", rank, vamana_path);
            MPI_Abort(MPI_COMM_WORLD, 5);
        }

        // Read the IndexHeader
        IndexHeader hdr;
        if (pread(vamana_fd, &hdr, sizeof(IndexHeader), 0) != sizeof(IndexHeader)) {
            fprintf(stderr, "Rank %d: Failed to read Vamana Index header from %s", rank, vamana_path);
            MPI_Abort(MPI_COMM_WORLD, 6);
        }

        // Read in the PQ encodings
        PQCodes encodings;
        if (pq_codes_load(pq_codes_path, &encodings) != 0) {
            fprintf(stderr, "Rank %d: Failed to read the encodings at %s", rank, pq_codes_path);
            MPI_Abort(MPI_COMM_WORLD, 7);
        }

        // Checking hashes
        if (pq_codes_matches_codebook(&encodings, &codebook) != 0) {
            fprintf(stderr, "Rank %d: Encodings at %s was not encoded with the provide codebook at %s!\n"
                            "Encoding hash: %016llx which not matches codebook hash %016llx.\n",
                            rank, pq_codes_path, codebook_path, encodings.codebook_hash, codebook.hash);
            MPI_Abort(MPI_COMM_WORLD, 8);
        }

        // Check points metadata against the header
        if (encodings.n_points != hdr.n_points) {
            fprintf(stderr, "Rank %d: PQ encoding point count %u does not match Vamana graph point count %u! "
                            "Base file name of %s"
                        ,rank, encodings.n_points, hdr.n_points, base_filename);
        }

        // Shard cleanups
        close(vamana_fd);
        pq_codes_free(&encodings);
    }

    /* Cleanups */
    free(queries); // Techically a potential memory hazard for rank 0 queries but all vecfile except for data is stack-allocated.
    pq_codebook_free(&codebook);
    free_vamana_list(&vamana_shards);
    free(local_ids);
    free(local_shard);
    free(local_dists);

    MPI_Finalize();

    return 0;
}   