import argparse
import random
import struct

def gen_random_cluster_and_write_vecfile(out_path, total_vectors, dim, n_clusters, cluster_st_dev, seed):
    """
    Generate random vectors that is clustered and write via our vecfile serialization format
    """
    random.seed(seed)
    centers = [[random.uniform(-10, 10) for _ in range(dim)] for _ in range(n_clusters)]

    with open(out_path, "wb") as f:
        # Write binary header
        f.write(struct.pack("<II", total_vectors, dim))
        
        # Stream vectors one by one
        for _ in range(total_vectors):
            c = centers[random.randrange(n_clusters)]
            v = [random.gauss(c[d], cluster_st_dev) for d in range(dim)]
            f.write(struct.pack(f"<{dim}f", *v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_path")
    ap.add_argument("--num-vectors", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-clusters", type=int, default=200)
    ap.add_argument("--cluster-st-dev", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=69)
    args = ap.parse_args()

    gen_random_cluster_and_write_vecfile(
        args.out_path, 
        args.num_vectors, 
        args.dim, 
        args.num_clusters, 
        args.cluster_st_dev, 
        args.seed
    )

    print(f"Wrote data to {args.out_path}")