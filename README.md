# vamana-avx
From-scratch implementation Vamana vector search algorithm (more specifically [DiskANN](https://milvus.io/blog/diskann-explained.md)) for learning [AVX-intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) programming (and a bit of under the hood ML algorithms) alongside some other cool high-performance software technologies!


## Checklist
- [x] Doing metric computations with SIMD extensions.
- [x] Creating a custom (and simple) data serialization of `.vecf` (again, this is not an established format, but just our own one)
    - [x] Make our own data generator and serialize it to `.vecf`
    - [x] Make a `.vecf` data sharding for encoding purposes.
- [x] Creating our own custom [product quantization](https://towardsdatascience.com/similarity-search-product-quantization-b2a1a6397701/) (of FP32s) scheme and codebooks (serialized format extensions of `.pqbook` for codebooks and `.pqbin` for encodings)
- [x] Make our own PQ codebook training algorithm.
- [x] Encode data shards using trained codebooks.
- [x] Create indexing formats to be used for searching.
- [ ] Building index for Vamana graph search
- [ ] Writing a (distributed) Vamana graph search algorithm on our custom data (MPI, OpenMP alongside `io_uring` for beam width batch and reranking).
- [ ] (Optional) Observe the searching program with eBPF through Rust Aya.

## Current pipeline 
```bash
# Create high dimensional dataset and serialize to `.vecf` format (along with the shardings)
./scripts/generate.sh

# Compile the codebook training binary and train 
./scripts/run_training.sh

# Encode the data shards using the trained codebook (note the hard-codings in the scripts)
./scripts/encode_shards.sh data/shards/ data/shards/pq_codebook.pqbook

# Build index and search (TODO)
```