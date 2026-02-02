"""Script to process the Flipside wallet"""

import os
from glob import glob

import pandas as pd
from tqdm import tqdm

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH, label_schema

os.makedirs(f"{PROCESSED_DATA_PATH}/flipside", exist_ok=True)

label_files = glob(f"{DATA_PATH}/ethereum/label/*.snappy.parquet")
dfs = []

for file in tqdm(label_files):
    df = pd.read_parquet(file)
    dfs.append(df)

df_all = pd.concat(dfs, axis=0, ignore_index=True)
df_all.columns = label_schema

df_hot_wallet = df_all.loc[
    (df_all["label_type"] == "cex") & (df_all["label_subtype"] == "hot_wallet"),
    ["address_name", "label", "address"],
]

# compress and save
df_hot_wallet.reset_index(drop=True, inplace=True)
# df_hot_wallet.to_parquet(f"{PROCESSED_DATA_PATH}/flipside/hot_wallet.parquet")
