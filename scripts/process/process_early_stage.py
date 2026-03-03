"""Script to process early stage CEX country data."""

import os
from glob import glob
import multiprocessing as mp
from functools import partial

import pandas as pd
from environ.constant import (
    PROCESSED_DATA_PATH,
    DATA_PATH,
    token_transfer_schema,
    native_transfer_schema,
    MIN_DATE,
)
from tqdm import tqdm


N_WORKERS = 15
MIN_YEAR = pd.to_datetime(MIN_DATE).year

# Load the early stage CEX country data
df = []
early_stage_files = glob(f"{DATA_PATH}/early_stage/*.xlsx")

for file in early_stage_files:
    df_early_stage = pd.read_excel(file, sheet_name="GeminiResults")
    df_early_stage["exchange"] = df_early_stage["exchange_name"].str.lower()
    df_early_stage.dropna(subset=["exchange"], inplace=True)
    df.append(df_early_stage)

df_early_stage = pd.concat(df, ignore_index=True)

df = []
for idx, row in df_early_stage.iterrows():
    exchange = row["exchange"]
    latest_year = row["latest_year"]
    if pd.isna(latest_year):
        continue
    latest_year = int(latest_year)
    years = list(range(MIN_YEAR, latest_year))
    for year in years:
        df.append(
            {
                "exchange": exchange,
                "country": row["country"],
                "year": int(year),
            }
        )

df = pd.DataFrame(df)

# Load the exchange data
cex = pd.read_excel(f"{DATA_PATH}/cex_wallet_260127.xlsx")
cex["exchange"] = cex["exchange"].str.lower()
cex = cex.loc[cex["network"] == "ETH"]

# remove the duplicates
cex.sort_values(by=["exchange", "address"], ascending=True, inplace=True)
cex["address"] = cex["address"].str.lower()
cex.drop_duplicates(subset=["exchange", "address"], inplace=True)

# Merge with the early stage CEX country data
df_early_stage = cex.merge(
    df, left_on="exchange", right_on="exchange", how="left"
).dropna(subset=["year"])
df_early_stage = df_early_stage[["address", "country", "year"]]

# Create from_address and to_address versions
df_early_stage_from = df_early_stage.rename(columns={"address": "from_address"})
df_early_stage_to = df_early_stage.rename(columns={"address": "to_address"})


def _process_one(file: str, token_type: str):
    """Process a single token transfer file to filter out transfers involving smart contracts."""

    idx = os.path.basename(file).replace(".snappy.parquet", "")

    # Load token transfer
    df = pd.read_parquet(file)
    df.columns = (
        token_transfer_schema if token_type == "token" else native_transfer_schema
    )
    df = df[["from_address", "to_address", "block_timestamp"]]
    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], unit="s")
    df["year"] = df["block_timestamp"].dt.year

    # Merge with early stage CEX country data
    df_from = (
        df.merge(
            df_early_stage_from,
            on=["from_address", "year"],
            how="left",
        )
        .dropna(subset=["country"])[
            [
                "to_address",
                "year",
                "country",
            ]
        ]
        .rename(columns={"to_address": "address"})
    )

    df_to = (
        df.merge(
            df_early_stage_to,
            on=["to_address", "year"],
            how="left",
        )
        .dropna(subset=["country"])[
            [
                "from_address",
                "year",
                "country",
            ]
        ]
        .rename(columns={"from_address": "address"})
    )

    pd.concat([df_from, df_to]).to_parquet(
        PROCESSED_DATA_PATH
        / "ethereum"
        / f"{token_type}_early_stage"
        / f"{idx}.parquet",
        index=False,
        compression="snappy",
    )


if __name__ == "__main__":
    with mp.Pool(processes=N_WORKERS) as pool:
        for token_type in ["token", "native"]:
            os.makedirs(
                PROCESSED_DATA_PATH / "ethereum" / f"{token_type}_early_stage",
                exist_ok=True,
            )
            files = glob(f"{DATA_PATH}/ethereum/{token_type}_transfer/*.parquet")
            work = partial(_process_one, token_type=token_type)
            list(tqdm(pool.imap(work, files), total=len(files)))
