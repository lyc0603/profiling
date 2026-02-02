"""Script to merge the timezone files in batches of 100."""

from glob import glob

import math
import pandas as pd
from tqdm import tqdm

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH

BATCH_SIZE = 100

for token_type in ["token", "native"]:
    df_agg = None
    files = sorted(
        glob(f"{PROCESSED_DATA_PATH}/timezone/{token_type}_transfer/*.parquet")
    )
    n_files = len(files)
    n_batches = math.ceil(n_files / BATCH_SIZE)

    for start in tqdm(range(0, n_files, BATCH_SIZE), desc=f"{token_type} batches"):
        batch_files = files[start : start + BATCH_SIZE]

        # Load + concat this batch
        batch_list = [pd.read_parquet(f) for f in batch_files]
        batch_df = pd.concat(batch_list, axis=0, ignore_index=True)

        # Group within this batch
        batch_grouped = batch_df.groupby("from_address", as_index=False).sum(
            numeric_only=True
        )

        # Merge with global aggregated df
        if df_agg is None:
            df_agg = batch_grouped
        else:
            df_agg = (
                pd.concat([df_agg, batch_grouped], axis=0, ignore_index=True)
                .groupby("from_address", as_index=False)
                .sum(numeric_only=True)
            )

    df_agg.to_parquet(
        PROCESSED_DATA_PATH / f"{token_type}_timezone.parquet", index=False
    )
