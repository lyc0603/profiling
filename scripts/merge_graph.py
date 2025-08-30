"""
Merge Graph
"""

import gc
import os
import glob
import pandas as pd
from tqdm import tqdm
from environ.constant import PROCESSED_DATA_PATH


def aggregate_temp_files(csv_paths):
    """Aggregate a list of temp CSV files into one edge DataFrame."""
    agg_dict = {}
    for path in tqdm(csv_paths, desc=f"Merging {len(csv_paths)} files"):
        df = pd.read_csv(path)
        for row in df.itertuples(index=False):
            key = (row.from_address, row.to_address)
            agg_dict[key] = agg_dict.get(key, 0) + row.amount_usd

    return pd.DataFrame(
        [(k[0], k[1], v) for k, v in agg_dict.items()],
        columns=["from_address", "to_address", "amount_usd"],
    )


if __name__ == "__main__":
    for label in ["token_transfer", "native_transfer"]:
        print(f"Processing {label}...")
        csv_paths = glob.glob(os.path.join(PROCESSED_DATA_PATH, label, "*.csv"))
        df = aggregate_temp_files(csv_paths)

        out_path = os.path.join(PROCESSED_DATA_PATH, f"{label}_graph.parquet")
        df.to_parquet(out_path, index=False)
        del df
        gc.collect()
