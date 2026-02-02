"""Script to process the timezone data"""

import numpy as np
import pandas as pd
import json

from environ.constant import PROCESSED_DATA_PATH
from tqdm import tqdm

df = pd.concat(
    [
        pd.read_parquet(f"{PROCESSED_DATA_PATH}/{token_type}_timezone.parquet")
        for token_type in ["token", "native"]
    ],
    axis=0,
    ignore_index=True,
)

df = df.groupby("from_address", as_index=False).sum(numeric_only=True)

# calculate the percentage of transfers in each timezone bucket for all, dst, and non-dst
type_col = ["all", "dst", "non_dst"]

for t in type_col:
    cols = [f"{t}_{h}" for h in range(24)]

    denom = df[cols].sum(axis=1)

    df[cols] = df[cols].div(denom.replace(0, pd.NA), axis=0).fillna(0.0)

df.to_parquet(PROCESSED_DATA_PATH / "timezone.parquet", index=False)

df = pd.read_parquet(PROCESSED_DATA_PATH / "timezone.parquet")
df.reset_index(drop=True, inplace=True)
cols = [type_col_item + f"_{h}" for type_col_item in type_col for h in range(24)]

# get the seed2id mapping
seed2id = {
    address: idx
    for idx, address in tqdm(
        enumerate(df["from_address"].tolist()), desc="Building seed2id mapping"
    )
}
with open(PROCESSED_DATA_PATH / "timezone" / "seed2id.json", "w") as f:
    json.dump(seed2id, f)

vector = df[cols].to_numpy()
np.save(PROCESSED_DATA_PATH / "timezone" / "timezone.npy", vector)
