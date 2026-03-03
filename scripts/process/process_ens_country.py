"""Script to process the ens country data"""

import json
from collections import defaultdict

import pandas as pd

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH


# Load the anchor country name ISO mapping
with open(PROCESSED_DATA_PATH / "anchor_country_iso.json", "r") as f:
    anchor_country_iso = json.load(f)

anchor_country_iso["Taiwan"] = "TWN"
anchor_country_iso["Isle of Man"] = "IMN"
anchor_country_iso["China"] = "CHN"
iso_anchor_country = {v: k for k, v in anchor_country_iso.items()}

df = pd.read_parquet(
    DATA_PATH / "ens" / "3301_ens_language_country_with_weights.parquet"
)

# Keep only high confidence records and unique country
df = df.loc[df["avg_probs"] >= 0.99]
df = df.loc[
    ~df["owner"].isin(
        df.loc[df[["ens_domain", "owner"]].duplicated(), "owner"].unique()
    )
]

country_address = defaultdict(set)

for _, row in df.iterrows():
    country_iso = row["ctry_iso"]
    address = row["owner"]
    if country_iso in iso_anchor_country:
        country_address[iso_anchor_country[country_iso]].add(address)

# Convert sets to lists
country_address = {k: list(v) for k, v in country_address.items()}

with open(PROCESSED_DATA_PATH / "gt" / "ens_gt.json", "w") as f:
    json.dump(country_address, f)
