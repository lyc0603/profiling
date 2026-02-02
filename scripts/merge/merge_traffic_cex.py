"""Script to process the cex embedding data"""

import json

import pandas as pd

from environ.constant import DATA_PATH, PROCESSED_DATA_PATH
from environ.utils import clean_name, match_top

CEX_TRAFFIC_MAPPING = {
    "btcturk": "btcTurk kripto",
    "gate.io": "gate",
    "crypto.com exchange": "crypto.comexchange",
    "xt": "xt.com",
}

# Load the traffic data
df_traffic = pd.read_csv(PROCESSED_DATA_PATH / "traffic.csv")
df_traffic["c_name"] = df_traffic["cex"].apply(clean_name)
cands = df_traffic["c_name"].unique().tolist()

# Load the exchange data
cex = pd.read_csv(f"{DATA_PATH}/cex.csv")
cex["exchange"] = cex["exchange"].str.lower()
cex["exchange"] = cex["exchange"].replace(CEX_TRAFFIC_MAPPING)
cex = cex.loc[cex["network"] == "ETH"]

# remove the duplicates
cex.sort_values(by=["exchange", "address"], ascending=True, inplace=True)
cex["address"] = cex["address"].str.lower()
cex.drop_duplicates(subset=["exchange", "address"], inplace=True)

# prepare the matching
cex_traffic = cex.copy()
cex = cex.drop_duplicates(subset=["exchange"])
cex["c_name"] = cex["exchange"].apply(clean_name)

# Match the cex names
cex["score"], cex["match"] = zip(*cex["c_name"].apply(lambda x: match_top(x, cands)[0]))
cex = cex.loc[cex["score"] == 1, ["exchange", "c_name"]]

cex = cex.merge(df_traffic, left_on="c_name", right_on="c_name", how="left")
cex_traffic = (
    cex_traffic.merge(cex, on="exchange", how="left")
    .dropna(subset=["c_name"])
    .drop(columns=["network", "website", "c_name", "cex"])
)
cex_traffic.drop_duplicates(subset=["address"], inplace=True)

# Load the China dataset
with open(f"{PROCESSED_DATA_PATH}/cex_china.json", "r") as f:
    cex_china = json.load(f)

cex_traffic["is_china"] = cex_traffic["exchange"].map(cex_china)


# Transform the matrix to panel data *
value_cols = [
    c for c in cex_traffic.columns if c not in ["exchange", "address", "is_china"]
]

panel = cex_traffic.melt(
    id_vars=["exchange", "address", "is_china"],
    value_vars=value_cols,
    var_name="country",
    value_name="value",
)

panel = panel.sort_values(by=["exchange", "country"])

# Create datasets with 0.5 China traffic
CHINA_PCT = 0.5
panel_china_5 = panel.copy()
panel_china_5["value"] = panel_china_5["value"] * panel_china_5["is_china"].apply(
    lambda x: 1 - CHINA_PCT if x else 1.0
)

# Add China rows using original matrix *
china_rows_5 = cex_traffic[["exchange", "address", "is_china"]].assign(
    country="China",
    value=cex_traffic["is_china"].apply(lambda x: CHINA_PCT if x else 0.0),
)

panel_china_5 = pd.concat([panel_china_5, china_rows_5], ignore_index=True)
panel_china_5 = panel_china_5.sort_values(by=["exchange", "country"])
cex_traffic_china_5 = panel_china_5.drop(columns=["is_china"])

# Create datasets with 0.9 China traffic
CHINA_PCT = 0.9
panel_china_9 = panel.copy()
panel_china_9["value"] = panel_china_9["value"] * panel_china_9["is_china"].apply(
    lambda x: 1 - CHINA_PCT if x else 1.0
)

# Add China rows using original matrix *
china_rows_9 = cex_traffic[["exchange", "address", "is_china"]].assign(
    country="China",
    value=cex_traffic["is_china"].apply(lambda x: CHINA_PCT if x else 0.0),
)

panel_china_9 = pd.concat([panel_china_9, china_rows_9], ignore_index=True)
panel_china_9 = panel_china_9.sort_values(by=["exchange", "country"])
cex_traffic_china_9 = panel_china_9.drop(columns=["is_china"])

cex_traffic = panel.drop(columns=["is_china"])
