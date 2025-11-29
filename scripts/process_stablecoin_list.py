#!/usr/bin/env python3
"""
Process the USDC/USDT List (plus CEX address sets) with multiprocessing.

- Reads *.snappy.parquet token transfer files in parallel.
- For each file, filters rows for USDC and USDT (by contract address).
- Writes per-token CSV shards into PROCESSED_DATA_PATH/transfer_{token}/
- Collects unique sender/receiver addresses for each token across all files.
- Saves final deduplicated address lists:
    - PROCESSED_DATA_PATH/usdc_address.txt
    - PROCESSED_DATA_PATH/usdt_address.txt
"""

import os
import glob
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, get_context

from environ.constant import (
    token_transfer_schema,
    DATA_PATH,
    USDC,
    USDT,
    PROCESSED_DATA_PATH,
)

# ------------------------------ Config ------------------------------

NAME_LIST = ["usdc", "usdt"]
ADD_LIST = [USDC, USDT]

INPUT_GLOB = f"{DATA_PATH}/ethereum/token_transfer/*.snappy.parquet"
OUT_DIRS = {
    name: os.path.join(PROCESSED_DATA_PATH, f"transfer_{name}") for name in NAME_LIST
}

# Number of workers (tune if needed)
N_WORKERS = 10

# ------------------------------ Utils ------------------------------


def _ensure_dirs():
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    for p in OUT_DIRS.values():
        os.makedirs(p, exist_ok=True)


def _make_output_path(in_path: str, token_name: str) -> str:
    """Map an input parquet path to the token-specific CSV output path."""
    base = os.path.basename(in_path).replace(".snappy.parquet", ".csv")
    return os.path.join(OUT_DIRS[token_name], base)


# ------------------------------ Worker ------------------------------


def _process_one(path: str):
    """
    Worker function:
    - read parquet
    - filter for USDC/USDT
    - write CSV shards
    - return (usdc_set, usdt_set) of unique addresses found in this file
    """
    # Local sets to return
    usdc_addrs = set()
    usdt_addrs = set()

    # Read parquet
    df = pd.read_parquet(path, engine="pyarrow")
    # Ensure schema alignment if needed
    df.columns = token_transfer_schema

    # Process each token
    for name, address in zip(NAME_LIST, ADD_LIST):
        df_subset = df[df["contract_address"] == address]

        if df_subset.empty:
            # Still create an empty CSV for traceability, or skip if preferred
            out_path = _make_output_path(path, name)
            # Create an empty file only if you want markers; otherwise skip writing
            # Here we write only when there is data:
            # pd.DataFrame(columns=df_subset.columns).to_csv(out_path, index=False)
            continue

        # Update local sets
        src_unique = df_subset["from_address"].unique()
        dst_unique = df_subset["to_address"].unique()

        if name == "usdc":
            usdc_addrs.update(src_unique)
            usdc_addrs.update(dst_unique)
        else:
            usdt_addrs.update(src_unique)
            usdt_addrs.update(dst_unique)

        # Write CSV shard
        out_path = _make_output_path(path, name)
        df_subset.to_csv(out_path, index=False)

    # Return the address sets for merging in parent
    return usdc_addrs, usdt_addrs


# ------------------------------ Main ------------------------------


def main():
    _ensure_dirs()

    token_files = glob.glob(INPUT_GLOB)
    token_files.sort()

    if not token_files:
        print("No input files found. Check path:", INPUT_GLOB)
        return

    global_usdc = set()
    global_usdt = set()

    # Use spawn (more robust across platforms) and keep progress with tqdm
    with get_context("spawn").Pool(processes=N_WORKERS) as pool:
        for usdc_set, usdt_set in tqdm(
            pool.imap_unordered(_process_one, token_files, chunksize=1),
            total=len(token_files),
            desc="Processing parquet files",
        ):
            # Merge into global sets
            if usdc_set:
                global_usdc |= usdc_set
            if usdt_set:
                global_usdt |= usdt_set

    # Save final deduped address lists
    usdc_out = os.path.join(PROCESSED_DATA_PATH, "usdc_address.txt")
    usdt_out = os.path.join(PROCESSED_DATA_PATH, "usdt_address.txt")

    # Write one per line; cast to str in case
    with open(usdc_out, "w") as f:
        f.write("\n".join(map(str, sorted(global_usdc))))
    with open(usdt_out, "w") as f:
        f.write("\n".join(map(str, sorted(global_usdt))))

    print(f"Done. USDC addresses: {len(global_usdc):,} -> {usdc_out}")
    print(f"Done. USDT addresses: {len(global_usdt):,} -> {usdt_out}")


if __name__ == "__main__":
    main()
