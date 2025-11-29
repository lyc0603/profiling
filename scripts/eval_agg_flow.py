"""Script to evaluate country-level flow aggregation."""

import os
import glob
import json
import multiprocessing as mp
from functools import partial

import pandas as pd
from tqdm import tqdm
from environ.constant import PROCESSED_DATA_PATH

for stable in ["usdc", "usdt"]:
    flow_files = glob.glob(f"{PROCESSED_DATA_PATH}/transfer_{stable}_country/*.csv")

    flow_agg = []
    for file_path in tqdm(flow_files):
        idx = os.path.basename(file_path).split(".")[0]
        df_flow = pd.read_csv(file_path)

        df_flow.dropna(subset=["from_country", "to_country"], how="any", inplace=True)
        df_flow = (
            df_flow.groupby(["from_country", "to_country"], dropna=False)
            .agg({"amount": "sum", "amount_usd": "sum"})
            .reset_index()
        )
        flow_agg.append(df_flow)

    df_flow_agg = pd.concat(flow_agg, ignore_index=True)
    df_flow_agg = (
        df_flow_agg.groupby(["from_country", "to_country"], dropna=False)
        .agg({"amount": "sum", "amount_usd": "sum"})
        .reset_index()
    )
    df_flow_agg.to_csv(
        f"{PROCESSED_DATA_PATH}/transfer_{stable}_country_agg.csv", index=False
    )
