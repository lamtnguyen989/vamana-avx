"""
Sharding our `vecfile` serialization format file's data
"""
import argparse
import random
import struct
import os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_dir")
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.in_path, "rb") as f:
        num_vectors, dim = struct.unpack("<II", f.read(8))
        base = num_vectors // args.num_shards
        rem = num_vectors % args.num_shards
        for s in range(args.num_shards):
            count = base + (1 if s < rem else 0)
            out_path = os.path.join(args.out_dir, f"shard_{s}.vecf")
            with open(out_path, "wb") as out:
                out.write(struct.pack("<II", count, dim))
                out.write(f.read(count * dim * 4))
            print(f"shard {s}: {count} vectors -> {out_path}")