"""Script to process smart contract."""

import os
import pandas as pd
from glob import glob
from tqdm import tqdm

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH, contract_schema

sc = []
for file in tqdm(glob(f"{DATA_PATH}/ethereum/contract/*.snappy.parquet")):
    df = pd.read_parquet(file)
    sc.append(df)

sc = pd.concat(sc, ignore_index=True)
sc.columns = contract_schema
sc = sc[["address"]]

sc.to_parquet(f"{PROCESSED_DATA_PATH}/contract.parquet", index=False)
