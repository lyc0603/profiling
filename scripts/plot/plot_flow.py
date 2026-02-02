"""Script to plot flow between countries."""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from mpl_chord_diagram import chord_diagram

from environ.constant import PROCESSED_DATA_PATH


# Load data
with open(PROCESSED_DATA_PATH / "embedding" / "anchor_country.json") as f:
    anchor_country = json.load(f)
country_dict = {i: c for i, c in enumerate(anchor_country)}
country_dict[77] = "South Korea"

df_usdc = pd.read_csv(PROCESSED_DATA_PATH / "transfer_usdc_country_agg.csv")
df_usdt = pd.read_csv(PROCESSED_DATA_PATH / "transfer_usdt_country_agg.csv")
df = pd.concat([df_usdc, df_usdt], ignore_index=True)

df["source"] = df["from_country"].map(country_dict)
df["target"] = df["to_country"].map(country_dict)

df = df.groupby(["source", "target"], dropna=False).amount_usd.sum().reset_index()
# Calculate percentage and keep two decimal places
total_amount = df["amount_usd"].sum()
df["amount_usd_pct"] = (df["amount_usd"] / total_amount * 100).round(2)
df.sort_values("amount_usd_pct", ascending=False, inplace=True)

# Calculate the reginonal net flow
outflow = df.groupby("source")["amount_usd"].sum()
inflow = df.groupby("target")["amount_usd"].sum()
net_flow = outflow - inflow
net_flow = net_flow.reset_index()
net_flow.columns = ["country", "net_outflow_usd"]
net_flow.sort_values("net_outflow_usd", ascending=True, inplace=True)

# Calculate the bilateral net flow
pivot = df.pivot_table(
    index="source", columns="target", values="amount_usd", fill_value=0
)

# Bilateral net flow: A → B minus B → A
net_matrix = pivot - pivot.T

net_flow_table = net_matrix.stack().reset_index().rename(columns={0: "amount_usd"})
net_flow_table = net_flow_table[net_flow_table["source"] != net_flow_table["target"]]
net_flow_table = net_flow_table[net_flow_table["amount_usd"] != 0]
net_flow_table = net_flow_table.loc[net_flow_table["amount_usd"] > 0]
for col in ["source", "target"]:
    net_flow_table[col] = net_flow_table[col].apply(lambda x: x.replace(" ", "\n"))

# Convert to matrix remember the source and target are not same
countries = sorted(
    list(set(net_flow_table["source"]).union(set(net_flow_table["target"])))
)
country_index = {c: i for i, c in enumerate(countries)}
matrix = np.zeros((len(countries), len(countries)))
for _, row in net_flow_table.iterrows():
    i = country_index[row["source"]]
    j = country_index[row["target"]]
    matrix[i, j] = row["amount_usd"]

# Plot chord diagram
plt.figure(figsize=(30, 30))
chord_diagram(
    matrix,
    names=countries,
    directed=True,
    fontsize=6,
)
plt.show()
