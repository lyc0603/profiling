"""Script to filter out transfer without the involvement of smart contracts."""

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

# Load smart contract table
sc = pd.read_parquet(f"{PROCESSED_DATA_PATH}/contract.parquet")[["address"]]
sc.columns = ["sc_address"]
sc = sc.drop_duplicates(subset=["sc_address"])


def _process_one(file: str, token_type: str):
    """Process a single token transfer file to filter out transfers involving smart contracts."""

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

    # Merge on to_address
    df = df.merge(
        sc, how="left", left_on="to_address", right_on="sc_address", indicator="to_ind"
    )

    # Keep only rows where both merges found *no* match
    mask = (df["from_ind"] == "left_only") & (df["to_ind"] == "left_only")
    df = df.loc[mask]

    # Drop merge bookkeeping columns
    df = df.drop(columns=["sc_address_x", "sc_address_y", "from_ind", "to_ind"])

    df.to_parquet(
        PROCESSED_DATA_PATH
        / "ethereum"
        / f"{token_type}_transfer_wo_sc"
        / f"{idx}.parquet",
        index=False,
        compression="snappy",
    )


if __name__ == "__main__":
    with mp.Pool(processes=N_WORKERS) as pool:
        for token_type in ["token", "native"]:
            os.makedirs(
                PROCESSED_DATA_PATH / "ethereum" / f"{token_type}_transfer_wo_sc",
                exist_ok=True,
            )
            files = glob(f"{DATA_PATH}/ethereum/{token_type}_transfer/*.parquet")
            work = partial(_process_one, token_type=token_type)
            list(tqdm(pool.imap(work, files), total=len(files)))
