"""Script to process the cex embedding data"""

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

# convert to panel
cex_traffic_panel = []
country = [_ for _ in cex_traffic.columns if _ not in ["address", "exchange"]]
for idx, row in cex_traffic.iterrows():
    for col in country:
        cex_traffic_panel.append(
            {
                "exchange": row["exchange"],
                "address": row["address"],
                "country": col,
                "value": row[col],
            }
        )

cex_traffic = pd.DataFrame(cex_traffic_panel)
cex_traffic.sort_values(by=["exchange", "country"], inplace=True)
