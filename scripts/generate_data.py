import argparse
import random
import struct

def gen_random_cluster(total_vectors, dim, n_clusters, cluster_st_dev, seed):
    """
    Generate random vectors that is clustered
    """
    random.seed(seed)
    vectors = []
    for _ in range(num_vectors):
        c = centers[random.randrange(num_clusters)]
        vectors.append([random.gauss(c[d], cluster_std) for d in range(dim)])
    return vectors

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-vectors", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-clusters", type=int, default=200)
    ap.add_argument("--cluster-st-dev", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=69)
    args = ap.parse_args()

    # TODO