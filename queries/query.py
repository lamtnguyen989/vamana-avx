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

if __name__ == "__main__":
    # Parsing arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path")

    args = ap.parse_args()

    # Reading data
    data, n_vectors, dim = read_vecf(args.data_path)
    print(f"Dataset has {n_vectors} points in {dim} dimensions")
    