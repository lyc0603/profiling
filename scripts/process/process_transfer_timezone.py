"""Script to aggregate timezone data."""

import os
from glob import glob
import multiprocessing as mp
from functools import partial

import pandas as pd
from tqdm import tqdm

from environ.constant import (
    DATA_PATH,
    PROCESSED_DATA_PATH,
    token_transfer_schema,
    native_transfer_schema,
)

N_WORKERS = 10


# Load stablecoin address
with open(f"{PROCESSED_DATA_PATH}/stablecoin_address.txt", "r") as f:
    stablecoin_addrs = set(line.strip() for line in f)

sc = pd.DataFrame({"sc_address": list(stablecoin_addrs)})


def _process_one(file: str, token_type: str):
    """Process a single token transfer file to filter out transfers."""

    idx = os.path.basename(file).replace(".snappy.parquet", "")

    # Load token transfer
    df = pd.read_parquet(file)
    df.columns = (
        token_transfer_schema if token_type == "token" else native_transfer_schema
    )

    # Merge on from_address
    df = df.merge(
        sc,
        how="left",
        left_on="from_address",
        right_on="sc_address",
        indicator="from_ind",
    )

    # Keep only rows where merge found a match
    mask = df["from_ind"] == "both"
    df = df.loc[mask]

    # Drop merge bookkeeping columns
    df = df.drop(columns=["sc_address", "from_ind"])

    df.to_parquet(
        PROCESSED_DATA_PATH
        / "ethereum"
        / f"{token_type}_transfer_timezone"
        / f"{idx}.parquet",
        index=False,
        compression="snappy",
    )


if __name__ == "__main__":
    with mp.Pool(processes=N_WORKERS) as pool:
        for token_type in ["token", "native"]:
            os.makedirs(
                PROCESSED_DATA_PATH / "ethereum" / f"{token_type}_transfer_timezone",
                exist_ok=True,
            )
            files = glob(f"{DATA_PATH}/ethereum/{token_type}_transfer/*.parquet")
            work = partial(_process_one, token_type=token_type)
            list(tqdm(pool.imap(work, files), total=len(files)))
