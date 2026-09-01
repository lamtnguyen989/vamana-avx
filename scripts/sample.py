
# Reservoir sampling from the VecFile serialization
import argparse
import random
import struct


if __name__ = "__main__":
    # Setup argument parser
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--sample-size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    # Reservoir sampling
    random.seed(args.seed)
    with open(args.in_path, "rb") as f:
        num_vectors, dim = struct.unpack("<II", f.read(8))
            rec_bytes = dim * 4
            n = min(args.sample_size, num_vectors)

            reservoir = []
            for i in range(num_vectors):
                rec = f.read(rec_bytes)
                if i < n:
                    reservoir.append(rec)
                else:
                    j = random.randrange(0, i + 1)
                    if j < n:
                        reservoir[j] = rec
    
    # Writing the samples
    with open(args.out_path, "wb") as out:
        out.write(struct.pack("<II", len(reservoir), dim))
        for rec in reservoir:
            out.write(rec)

    print(f"sampled {len(reservoir)}/{num_vectors} vectors from {args.in_path} -> {args.out_path}")
