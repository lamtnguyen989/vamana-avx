#include <dirent.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
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

static void uring_read_vamana_index(struct io_uring* uring,
                                    int vamana_fd,
                                    const IndexHeader* hdr,
                                    uint32_t* vamana_ids,
                                    uint32_t count,
                                    size_t read_len,
                                    void* buffer,
                                    uint64_t stride)
{
    // Prep the submission queue
    for (uint64_t k = 0; k < count; k++) {
        // Get the submission queue
        struct io_uring_sqe* submission_queue = io_uring_get_sqe(uring);
        
        // Offset off the header and stride through flat index
        uint64_t offset = (uint64_t)index_offset_of(vamana_ids[k], hdr);
        
        // Prep read and tag with the index
        io_uring_prep_read(submission_queue, vamana_fd, buffer + k*stride, read_len, offset);
        io_uring_sqe_set_data64(submission_queue, k);
    }

    // Submit request
    io_uring_submit(uring);

    // Completion queue processing
    for (uint32_t k = 0; k < count; k++) {
        // Wait for the completion queue
        struct io_uring_cqe* completion_queue;
        io_uring_wait_cqe(uring, &completion_queue);
        
        // Check result code
        if (completion_queue->res < 0) {
            fprintf(stderr, "io_uring read failed. Error: ", strerror(completion_queue->res));
        }

        // Marking completion
        io_uring_cqe_seen(uring, completion_queue);
    }

}

// Query beam search to be run for each thread
static void beam_search_pq(int vamana_fd, 
                        const IndexHeader* hdr, 
                        const PQCodebook* codebook,
                        const PQCodes* encodings,
                        dist_fn_t dist_fn,
                        const float* query,
                        uint32_t L, 
                        uint32_t K, 
                        uint32_t beam_width,
                        uint32_t* out_ids,
                        float* out_dists)
{
    // Grab the thread-local io_uring
    struct io_uring* uring = get_thread_uring();

    // ADC table (1 instance per query)
    float* table = (float*) malloc(codebook->M * codebook->K * sizeof(float));
    pq_build_distance_table(codebook, dist_fn, query, table);

    // Initialize candidates
    CandidateList cand_list;
    candidate_list_init(&cand_list, L);

    // Initialize base distance (to medoid)
    float d0 = pq_adc_distance(codebook, table, pq_codes_at(encodings, hdr->medoid_id));
    insert_candidate(&cand_list, hdr->medoid_id, d0);

    // Batch scratch space initialization
    uint32_t record_size = index_record_size_from_header(hdr);
    uint8_t* batch_buffer = (uint8_t*) malloc(beam_width * record_size * sizeof(uint8_t));
    uint32_t* batch_vamana_ids = (uint32_t*) malloc(beam_width*sizeof(uint32_t));
    uint32_t* unvisited_ids = (uint32_t*) malloc(beam_width*sizeof(uint32_t));

    // Preparing reading from Vamana index in batch of `beam_width` (maximum)
    uint32_t batch_count = 0;
    while((batch_count = next_unvisted_candidates_batch(&cand_list, unvisited_ids, beam_width)) > 0) {
        for (uint32_t k = 0; k < batch_count; k++) {
            cand_list.items[unvisited_ids[k]].visited = 1;
            batch_vamana_ids[k] = cand_list.items[unvisited_ids[k]].id;
        }

        // Read from Vamana index in batch of `batch_count` <= `beam_width`
        uring_read_vamana_index(uring, vamana_fd, hdr, batch_vamana_ids, batch_count, record_size, batch_buffer, record_size);

        // Decode all buffer to retrieve records and insert to the list
        IndexRecord record;
        for (uint32_t k = 0; k < batch_count; k++) {
            index_record_decode(hdr, batch_buffer + k*record_size, &record);
            // Walk ALL neighbors of this record, not just index [k]
            for (uint32_t j = 0; j < record.degree; j++) {
                uint32_t nb = record.neighbors[j];
                float d = pq_adc_distance(codebook, table, pq_codes_at(encodings, nb));
                insert_candidate(&cand_list, nb, d);
            }
        }
    }

    // Copy the top-K results out, padding any remainder with sentinels
    uint32_t top = (cand_list.size < K) ? cand_list.size : K;
    for (uint32_t k = 0; k < top; k++) {
        out_ids[k]   = cand_list.items[k].id;
        out_dists[k] = cand_list.items[k].dist;
    }
    for (uint32_t k = top; k < K; k++) {
        out_ids[k]   = UINT32_MAX;
        out_dists[k] = FLT_MAX;
    }

    // Cleanups
    candidate_list_free(&cand_list);
    free(table);
    free(batch_buffer);
    free(batch_vamana_ids);
    free(unvisited_ids);
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
    result.file_base = (char**) malloc(cap*sizeof(char*));

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
        result.file_base[result.count++] = file_base;
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

typedef struct {
    uint32_t id;        // Neighbor id 
    float dist;         // Distance
    uint32_t shard_id;  // Data shard that this neighbor came from
} ShardCandidate;

static void top_k_reduction(
    uint32_t* accumulator_ids, float* accumulator_dists, uint32_t* accumulator_shard_id,
    uint32_t* new_ids, float* new_dists, uint32_t new_shard_id, 
    uint32_t K, ShardCandidate* scratch_space)
{
    // Populate the scratch space
    uint32_t m = 0;
    for (uint32_t i = 0; (i < K) && accumulator_ids[i] != UINT32_MAX; i++) {
        scratch_space[m++] = (ShardCandidate) {
            .id = accumulator_ids[i],
            .dist = accumulator_dists[i],
            .shard_id = accumulator_shard_id[i],
        };
    }
    for (uint32_t i = 0; (i < K) && new_ids[i] != UINT32_MAX; i++) {
        scratch_space[m++] = (ShardCandidate) {
            .id = new_ids[i],
            .dist = new_dists[i],
            .shard_id = new_shard_id,
        };
    }

    // Insertion sort on ascending distance (reasonable since K is small relatively speaking)
    for (uint32_t i = 1; i < m; i++) {
        ShardCandidate key = scratch_space[i];
        int j = (int)i - 1;
        while (j >= 0 && scratch_space[j].dist > key.dist) {
            scratch_space[j + 1] = scratch_space[j]; 
            j--;
        }
    }

    // Record top-K
    uint32_t top = (m < K) ? m : K;
    for (uint32_t i = 0; i < top; i++) {
        accumulator_ids[i] = scratch_space[i].id; 
        accumulator_dists[i] = scratch_space[i].dist; 
        accumulator_shard_id[i] = scratch_space[i].shard_id; 
    }
    for (uint32_t i = top; i < K; i++) { 
        accumulator_ids[i] = UINT32_MAX; 
        accumulator_dists[i] = FLT_MAX; 
        accumulator_shard_id[i] = UINT32_MAX; 
    }
    
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
    if (argc < 6) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: mpirun -n <ranks> %s <queries.vecf> <K> <index_dir/> <codebook.pqbook> <pq_encoding_dir/> <result_file>"
                "[L=64] [beam_width=8] [threads_per_rank=4] \n"
                "\n"
                "  queries.vecf         Queries for search in the dataset.\n"
                "  K                    Number of top ranked choices.\n"
                "  index_dir/           Directory of Vamana graph index (will only currently process `.vamindx` files in the directory).\n"
                "  codebook.pqbook      Product quantization codebook file.\n"
                "  pq_encoding_dir/     Product quantization encodings directory (of data shards).\n"
                "  result_file          CSV File holding the results\n"
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
    const char* result_file = argv[6];
    uint32_t L = argc > 7 ? (uint32_t)atoi(argv[7]) : 64;
    uint32_t beam_width = argc > 8 ? (uint32_t)atoi(argv[8]) : 8;
    int n_threads = argc > 9 ? atoi(argv[9]) : 4;

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
    MPI_Bcast(queries, (int)(n_queries*dim), MPI_FLOAT, 0, MPI_COMM_WORLD);

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
            snprintf(check_path, sizeof(check_path), "%s/%s.pqbin", pq_dir, vamana_shards.file_base[k]);
            if (access(check_path, F_OK) != 0) {
                fprintf(stderr, "Shard `%s.vamindx` does not have accessible corresponding encoding of `%s.pqbin` in %s\n",
                                vamana_shards.file_base[k], index_dir, vamana_shards.file_base[k], pq_dir);
                MPI_Abort(MPI_COMM_WORLD, 4);
            }
        }
    }

    // Finding work boundary for the ranks
    uint32_t base = n_shards / (uint32_t)world_size;
    uint32_t remainder = n_shards % (uint32_t)world_size;
    uint32_t rank_count = base + (((uint32_t)rank < remainder) ? 1 : 0);
    uint32_t rank_start = (uint32_t)rank * base + (((uint32_t)rank < remainder) ? (uint32_t)rank : remainder);
    printf("Rank %d: Processing %d shards\n", rank, rank_count);
    if (rank_count == 0) {fprintf(stderr, "Rank %d is idle!", rank);}
    
    /* Search */
    // Select metric
    dist_fn_t dist_fn = metric();

    // Set-up rank-level scratch spaces
    uint32_t* local_ids = (uint32_t*)malloc(n_queries * K * sizeof(uint32_t));
    uint32_t* local_shard = (uint32_t*)malloc(n_queries * K * sizeof(uint32_t));
    float* local_dists = (float*) malloc(n_queries * K * sizeof(float));
    for (size_t k = 0; k < (size_t)K*n_queries; k++) {
        local_ids[k] = UINT32_MAX;
        local_shard[k] = UINT32_MAX;
        local_dists[k] = FLT_MAX;
    }
    char vamana_path[1024]; memset(vamana_path, 0, sizeof(vamana_path));
    char pq_codes_path[1024]; memset(pq_codes_path, 0, sizeof(pq_codes_path));

    uint32_t* shard_ids = (uint32_t*) malloc(n_queries*K*sizeof(uint32_t));
    float* shard_dists = (float*) malloc(n_queries*K*sizeof(float));

    ShardCandidate* reduction_scratch = (ShardCandidate*) malloc(2*K*sizeof(ShardCandidate));
    
    MPI_Barrier(MPI_COMM_WORLD); // Mainly to start the timings
    double t0 = MPI_Wtime();

    // Start processing (shard-by-shard)
    for (uint32_t s = 0; s < rank_count; s++) {
        uint32_t shard_idx = rank_start + s;
        char* base_filename = vamana_shards.file_base[shard_idx];
        snprintf(vamana_path, sizeof(vamana_path), "%s/%s.vamindx", index_dir, base_filename);
        snprintf(pq_codes_path, sizeof(pq_codes_path), "%s/%s.pqbin", pq_dir, vamana_shards.file_base[shard_idx]);

        // Not using fopen here for io_uring
        int vamana_fd = open(vamana_path, O_RDONLY);
        if (vamana_fd < 0) {
            fprintf(stderr, "Rank %d: Can not open %s\n", rank, vamana_path);
            MPI_Abort(MPI_COMM_WORLD, 5);
            return 5;
        }

        // Read the IndexHeader
        IndexHeader hdr;
        if (pread(vamana_fd, &hdr, sizeof(IndexHeader), 0) != sizeof(IndexHeader)) {
            fprintf(stderr, "Rank %d: Failed to read Vamana Index header from %s\n", rank, vamana_path);
            MPI_Abort(MPI_COMM_WORLD, 6);
            return 6;
        }

        // Read in the PQ encodings
        PQCodes encodings;
        if (pq_codes_load(pq_codes_path, &encodings) != 0) {
            fprintf(stderr, "Rank %d: Failed to read the encodings at %s\n", rank, pq_codes_path);
            MPI_Abort(MPI_COMM_WORLD, 7);
            return 7;
        }

        // Checking hashes
        if (!pq_codes_matches_codebook(&encodings, &codebook)) {
            fprintf(stderr, "Rank %d: Encodings at %s was not encoded with the provide codebook at %s!\n"
                            "Encoding hash: %016llx which not matches codebook hash %016llx.\n",
                            rank, pq_codes_path, codebook_path, encodings.codebook_hash, codebook.hash);
            MPI_Abort(MPI_COMM_WORLD, 8);
            return 8;
        }

        // Check points metadata against the header
        if (encodings.n_points != hdr.n_points) {
            fprintf(stderr, "Rank %d: PQ encoding point count %u does not match Vamana graph point count %u! "
                            "Base file name of %s\n"
                            ,rank, encodings.n_points, hdr.n_points, base_filename);
            MPI_Abort(MPI_COMM_WORLD, 9);
            return 9;
        }

        // Clearing out previous shard's data
        memset(shard_ids, 0, n_queries * K * sizeof(uint32_t));
        memset(shard_dists, 0, n_queries * K * sizeof(float));

        // Execute parallel search (across queries)
        #pragma omp parallel for num_threads(n_threads)
        for (uint32_t q = 0; q < n_queries; q++) {
            beam_search_pq(vamana_fd, &hdr, &codebook, &encodings, dist_fn, 
                            &queries[(size_t)q*dim], L, K, beam_width,
                            &shard_ids[(size_t)q*K], &shard_dists[(size_t)q*K]);
        }

        // Reduce to a top K results
        for (uint32_t q = 0; q < n_queries; q++) {
            top_k_reduction(&local_ids[(size_t)q*K], &local_dists[(size_t)q*K], &local_shard[(size_t)q*K], 
                            &shard_ids[(size_t)q*K], &shard_dists[(size_t)q*K], shard_idx, 
                            K, reduction_scratch
            );
        }

        // Shard cleanups
        close(vamana_fd);
        pq_codes_free(&encodings);
    }
    free(reduction_scratch);

    /* Gather all local top K results and to reduce to a global top K results at rank 0 */
    // Making space at rank 0 to store all results 
    uint32_t* all_ids = NULL;
    uint32_t* all_shards = NULL;
    float* all_dists = NULL;
    if (rank == 0) {
        all_ids = (uint32_t*) malloc(n_queries * K * world_size * sizeof(uint32_t));
        all_shards = (uint32_t*) malloc(n_queries * K * world_size * sizeof(uint32_t));
        all_dists = (float*) malloc(n_queries * K * world_size * sizeof(float));
    }

    // Gather all ranks result to rank 0
    MPI_Gather(local_shard, (int)(n_queries * K), MPI_UINT32_T,
                all_shards, (int)(n_queries * K), MPI_UINT32_T,
                0, MPI_COMM_WORLD);

    MPI_Gather(local_ids, (int)(n_queries * K), MPI_UINT32_T,
                all_ids, (int)(n_queries * K), MPI_UINT32_T,
                0, MPI_COMM_WORLD);

    MPI_Gather(local_dists, (int)(n_queries * K), MPI_FLOAT,
                all_dists, (int)(n_queries * K), MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    /* Merge to a global top-K at rank 0 */
    if (rank == 0) {
        // Open up the result file
        FILE* results = fopen(result_file, "w");
        if (!results) {
            perror("Failed to open file for writing results!\n");
            return -1;
        }
        fprintf(results, "query_id,neighbor_rank,neighbor_id,distance\n");

        // Insertion sort merging again
        ShardCandidate* merge_buf = (ShardCandidate*) malloc(world_size*K*sizeof(ShardCandidate));
        for (uint32_t q = 0; q < n_queries; q++) {
            uint32_t m = 0;

            // Collect top-K from all ranks for query q
            for (int r = 0; r < world_size; r++) {
                size_t base_offset = ((size_t)r * n_queries + q) * K;
                for (uint32_t k = 0; k < K; k++) { 
                    uint32_t candidate_id = all_ids[base_offset + k];
                    if (candidate_id != UINT32_MAX) {
                        merge_buf[m++] = (ShardCandidate) {
                            .id = candidate_id,
                            .dist = all_dists[base_offset + k],
                            .shard_id = all_shards[base_offset + k],
                        };
                    }
                }
            }

            // Insertion sort across all collected rank candidates
            for (uint32_t i = 1; i < m; i++) {
                ShardCandidate key = merge_buf[i];
                int j = (int)i - 1;
                while (j >= 0 && merge_buf[j].dist > key.dist) {
                    merge_buf[j + 1] = merge_buf[j];
                    j--;
                }
                merge_buf[j + 1] = key;
            }

            // Writing results to CSV
            uint32_t top = (m < K) ? m : K;
            for (uint32_t k = 0; k < top; k++) {
                fprintf(results, "%u,%u,%u,%.6f\n", q, k + 1, merge_buf[k].id, merge_buf[k].dist);
            }
        }

        // Cleanups
        fclose(results);
        free(merge_buf);
        free(all_dists);
        free(all_shards);
        free(all_ids);
    }

    /* Cleanups */
    free(shard_ids);
    free(shard_dists);
    free(queries); // Techically a potential memory hazard for rank 0 queries but all vecfile except for data is stack-allocated.
    pq_codebook_free(&codebook);
    free_vamana_list(&vamana_shards);
    free(local_ids);
    free(local_shard);
    free(local_dists);

    MPI_Finalize();

    return 0;
}   