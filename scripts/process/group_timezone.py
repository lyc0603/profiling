"""Script to merge timezone information"""

import os
from glob import glob
import multiprocessing as mp
from functools import partial

import pandas as pd
from tqdm import tqdm

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH

N_WORKERS = 10
DST_MONTHS = {3, 4, 5, 6, 7, 8, 9, 10}


def _process_one(file: str, token_type: str) -> None:
    """Process one file to calculate the timezone"""

    idx = os.path.basename(file)
    df = pd.read_parquet(file)

    # Select relevant columns and convert block_timestamp to datetime
    df = df[["from_address", "block_timestamp"]].copy()
    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], unit="s")

    # Extract hour from block_timestamp
    df["hour"] = df["block_timestamp"].dt.hour

    # Determine if the month is in DST months
    df["month"] = df["block_timestamp"].dt.month
    df["is_dst_month"] = df["month"].isin(DST_MONTHS).astype(int)

    # convert to 24 one-hot encoding
    hour_dummies = pd.get_dummies(df["hour"], dtype=int)
    hour_dummies = hour_dummies.reindex(columns=range(24), fill_value=0)
    hour_dummies = hour_dummies.add_prefix("all_")
    df = pd.concat([df, hour_dummies], axis=1)

    # check whether it is DST month 24 one-hot encoding or non-DST month 24 one-hot encoding
    for hour in range(24):
        df[f"dst_{hour}"] = ((df["is_dst_month"] == 1) & (df["hour"] == hour)).astype(
            int
        )
        df[f"non_dst_{hour}"] = (
            (df["is_dst_month"] == 0) & (df["hour"] == hour)
        ).astype(int)

    df.drop(columns=["block_timestamp", "hour", "month", "is_dst_month"], inplace=True)
    df = df.groupby(["from_address"]).sum().reset_index()

    df.to_parquet(
        PROCESSED_DATA_PATH / "timezone" / f"{token_type}_transfer" / f"{idx}",
        index=False,
        compression="snappy",
    )


if __name__ == "__main__":
    with mp.Pool(processes=N_WORKERS) as pool:
        for token_type in ["token", "native"]:
            os.makedirs(
                PROCESSED_DATA_PATH / "timezone" / f"{token_type}_transfer",
                exist_ok=True,
            )
            files = glob(
                f"{PROCESSED_DATA_PATH}/ethereum/{token_type}_transfer_timezone/*.parquet"
            )
            work = partial(_process_one, token_type=token_type)
            list(tqdm(pool.imap(work, files), total=len(files)))
