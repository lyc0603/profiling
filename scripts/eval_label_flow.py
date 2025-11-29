"""Label country information for stablecoin transfer flows."""

import os
import glob
import json
import multiprocessing as mp
from functools import partial

import pandas as pd
from tqdm import tqdm
from environ.constant import PROCESSED_DATA_PATH

N_WORKERS = 20

# Create output directories
for stable in ["usdc", "usdt"]:
    os.makedirs(f"{PROCESSED_DATA_PATH}/transfer_{stable}_country", exist_ok=True)

# Load mapping once
with open(PROCESSED_DATA_PATH / "address_wallet_country.json", "r") as f:
    mapping = json.load(f)

df_map = pd.DataFrame(list(mapping.items()), columns=["address", "country"])
df_map["address"] = df_map["address"].astype("string")
mapping = None


def identify_country(file_path: str, stable: str) -> None:
    """Save country-labeled flow data for one file."""

    idx = os.path.basename(file_path).split(".")[0]
    df_flow = pd.read_csv(file_path)

    # from country
    df_flow = df_flow.merge(
        df_map, left_on="from_address", right_on="address", how="left"
    )
    df_flow.rename(columns={"country": "from_country"}, inplace=True)
    df_flow.drop(columns=["address"], inplace=True)

    # to country
    df_flow = df_flow.merge(
        df_map, left_on="to_address", right_on="address", how="left"
    )
    df_flow.rename(columns={"country": "to_country"}, inplace=True)
    df_flow.drop(columns=["address"], inplace=True)

    df_flow.to_csv(
        f"{PROCESSED_DATA_PATH}/transfer_{stable}_country/{idx}.csv",
        index=False,
    )
    return None


if __name__ == "__main__":
    with mp.Pool(processes=N_WORKERS) as pool:
        for stable in ["usdc", "usdt"]:
            flow_files = glob.glob(f"{PROCESSED_DATA_PATH}/transfer_{stable}/*.csv")
            work = partial(identify_country, stable=stable)

            list(tqdm(pool.imap(work, flow_files), total=len(flow_files)))
