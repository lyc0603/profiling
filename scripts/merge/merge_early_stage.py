"""Script to merge early stage data"""

import json
from collections import defaultdict
from glob import glob
import pandas as pd
from environ.constant import (
    DATA_PATH,
    MIN_DATE,
    PROCESSED_DATA_PATH,
)
from tqdm import tqdm

MIN_YEAR = pd.to_datetime(MIN_DATE).year

# Load the early stage CEX country data
df_early_stage = pd.read_excel(
    DATA_PATH / "early_stage_cex_country.xlsx",
    sheet_name="GeminiResults",
)
df_early_stage["exchange"] = df_early_stage["exchange_name"].str.lower()
df_early_stage.dropna(subset=["exchange"], inplace=True)

# Load the exchange data
cex = pd.read_csv(f"{DATA_PATH}/cex.csv")
cex["exchange"] = cex["exchange"].str.lower()
cex = cex.loc[cex["network"] == "ETH"]

# remove the duplicates
cex.sort_values(by=["exchange", "address"], ascending=True, inplace=True)
cex["address"] = cex["address"].str.lower()
cex.drop_duplicates(subset=["exchange", "address"], inplace=True)

# Merge with the early stage CEX country data
df_early_stage = cex.merge(
    df_early_stage, left_on="exchange", right_on="exchange", how="left"
).dropna(subset=["exchange_name"])
df_early_stage = df_early_stage[["address", "country"]]
address_set = set(df_early_stage["address"].unique())


early_stage_country_address = defaultdict(set)

for token_type in ["token", "native"]:
    files = glob(f"{PROCESSED_DATA_PATH}/ethereum/{token_type}_early_stage/*.parquet")
    for file in tqdm(files):
        df = pd.read_parquet(file)
        df = df.loc[~df["address"].isin(address_set)]
        for _, row in df.iterrows():
            address = row["address"]
            country = row["country"]
            early_stage_country_address[country].add(address)

early_stage_address_freq_count = defaultdict(int)
for country, addresses in early_stage_country_address.items():
    for address in addresses:
        early_stage_address_freq_count[address] += 1

# Keep only addresses that appear in only one country
early_stage_address_country = {}
for country, addresses in early_stage_country_address.items():
    for address in addresses:
        if early_stage_address_freq_count[address] == 1:
            early_stage_address_country[address] = country


# Load the seed2id mapping
with open(PROCESSED_DATA_PATH / "seed2id.json", "r") as f:
    seed2id = json.load(f)

seed_set = set(seed2id.keys())
wallet_set = set(early_stage_address_country.keys())

# Find the intersection
intersection = seed_set.intersection(wallet_set)

final_early_stage_address_country = {}
for wallet in intersection:
    final_early_stage_address_country[wallet] = early_stage_address_country[wallet]

with open(PROCESSED_DATA_PATH / "early_stage_gt.json", "w") as f:
    json.dump(final_early_stage_address_country, f)
