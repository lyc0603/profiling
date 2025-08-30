"""Script to check the random walk data"""

#!/usr/bin/env python3
import os, gzip
from tqdm import tqdm
from environ.constant import PROCESSED_DATA_PATH

WALK_PATH = f"{PROCESSED_DATA_PATH}/random_walk/weighted_walks.txt"


def open_any(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def main():
    num_walks = 0
    total_len = 0
    min_len = float("inf")
    max_len = 0
    first_nodes = set()

    with open_any(WALK_PATH) as f:
        for line in tqdm(f, desc="Scanning walks"):
            toks = line.strip().split()
            if not toks:
                continue
            L = len(toks)
            num_walks += 1
            total_len += L
            min_len = min(min_len, L)
            max_len = max(max_len, L)
            first_nodes.add(toks[0])

    avg_len = total_len / num_walks if num_walks > 0 else 0

    print(f"Total walks: {num_walks:,}")
    print(f"Length: min={min_len}, max={max_len}, avg={avg_len:.2f}")
    print(f"Unique first nodes: {len(first_nodes):,}")

    # Optionally dump first nodes to file
    out_path = os.path.splitext(WALK_PATH)[0] + "_firstnodes.txt"
    with open(out_path, "w") as f:
        for node in sorted(first_nodes, key=int):
            f.write(str(node) + "\n")
    print(f"Wrote first-node set to {out_path}")


if __name__ == "__main__":
    main()
