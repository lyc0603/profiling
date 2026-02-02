"""Script to plot the wallet classification data."""

import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from environ.constant import PROCESSED_DATA_PATH

# Load the country list
with open(PROCESSED_DATA_PATH / "embedding" / "anchor_country_multiple.json") as f:
    anchor_country = json.load(f)

country_dict = {}
for idx, country in enumerate(anchor_country):
    country_dict[idx] = country

# Load the address to wallet country mapping
with open(PROCESSED_DATA_PATH / "address_wallet_country.json") as f:
    address_wallet_country = json.load(f)

# Convert mapping to DataFrame
df_country = pd.DataFrame(
    [
        {"address": addr, "country_id": cid}
        for addr, cid in address_wallet_country.items()
    ]
)

# Map country_id to country name
df_country["country"] = df_country["country_id"].map(country_dict)

# Count wallets per country
country_counts = df_country["country"].value_counts().sort_values(ascending=False)

# Convert it into percentage
country_counts = country_counts / country_counts.sum() * 100

# Pllot
plt.figure(figsize=(14, 8))
plt.bar(country_counts.index, country_counts.values)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Wallet Percentage (%)", fontsize=20)
plt.tight_layout()
plt.show()
