# Bootstraping script for generating queries and generating ground truths

import argparse
import struct
import numpy as np
import pandas as pd

def read_vecf(file_path, header_bytes=8):
    """
    Read vecfile serialization via mmap
    """
    # Reading headers
    with open(file_path, "rb") as vf:
        n_vectors, dim = struct.unpack("<II", vf.read(header_bytes))
    # Extract data via mmap
    data = np.memmap(file_path, dtype=np.float32, offset=header_bytes, shape=(n_vectors, dim))
    # Return data alongside metadata
    return data, n_vectors, dim

def write_vecf(out_path, query_vectors):
    """
    Write to vecfile serialization (mainly for writing queries)
    """
    n_vectors, dim = query_vectors.shape
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", n_vectors, dim))
        f.write(query_vectors.astype(np.float32).tobytes())

def compute_ground_truths_L2(data, queries, K, batch_size=1000):
    """
    Compute exact top-K nearest neighbors using L2 distance.
    """
    # Pre-calculate squared norm of dataset vectors
    data_sq = np.sum(data**2, axis=1)

    # Sizes
    n_queries = queries.shape[0]
    gt_indices = np.zeros((n_queries, K), dtype=np.uint32)
    gt_distances = np.zeros((n_queries, K), dtype=np.float32)

    # Process query in batches
    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        q_batch = queries[start_idx:end_idx]

        # ||q - d||^2 = ||q||^2 + ||d||^2 - 2 * <q, d>
        q_sq = np.sum(q_batch**2, axis=1, keepdims=True)
        dists_sq = q_sq + data_sq - 2 * (q_batch @ data.T)
        dists = np.sqrt(np.maximum(dists_sq, 0.0))

        # Partition top-K elements
        partition_idx = np.argpartition(dists, K, axis=1)[:, :K]
        
        # Sort top-K candidates
        row_idx = np.arange(q_batch.shape[0])[:, None]
        topk_dists = dists[row_idx, partition_idx]
        sort_order = np.argsort(topk_dists, axis=1, kind="stable")

        gt_indices[start_idx:end_idx] = partition_idx[row_idx, sort_order]
        gt_distances[start_idx:end_idx] = topk_dists[row_idx, sort_order]
    return gt_indices, gt_distances

def save_ground_truth_csv(out_path, gt_indices, gt_distances):
    """
    Export ground truth to CSV.
    """
    n_queries, K = gt_indices.shape

    query_ids = np.repeat(np.arange(n_queries), K)
    ranks = np.tile(np.arange(1, K + 1), n_queries)

    df = pd.DataFrame({
        "query_id": query_ids,
        "neighbor_rank": ranks,
        "neighbor_id": gt_indices.ravel(),
        "distance": gt_distances.ravel()
    })
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    # Parsing arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("query_out_path")
    ap.add_argument("ground_truth_csv")
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--std-dev", type=float, default=5.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", choices=["L2", "cosine"], default="L2")
    ap.add_argument("--batch_size", type=int, default=2000)

    args = ap.parse_args()

    # Reading data
    data, n_vectors, dim = read_vecf(args.data_path)
    print(f"Dataset has {n_vectors} points in {dim} dimensions")

    # Generate noisy queries sampled from dataset points
    print(f"Generating {args.n_queries} queries (Gaussian std={args.std_dev})...")
    rng = np.random.default_rng(args.seed)
    base_indices = rng.choice(n_vectors, size=args.n_queries, replace=True)
    queries = (data[base_indices] + rng.normal(0, args.std_dev, size=(args.n_queries, dim))).astype(np.float32)

    # Write queries to binary format
    print(f"Writing query binary to '{args.query_out_path}'...")
    write_vecf(args.query_out_path, queries)

    # Compute ground truths
    print(f"Computing top-{args.k} ground truth using L2 distance...")
    gt_indices, gt_distances = compute_ground_truths_L2(
        data, queries, args.k, batch_size=args.batch_size
    )

    # Save ground truths to CSV
    print(f"Saving ground truth CSV to '{args.ground_truth_csv}'...")
    save_ground_truth_csv(args.ground_truth_csv, gt_indices, gt_distances)
    