"""Script to plot the average wallet country embedding (2x2 figure)."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environ.constant import PROCESSED_DATA_PATH

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

row_map = {"single": 0, "multiple": 1}
col_map = {"avg": 0, "one_hot": 1}

for kind in ["single", "multiple"]:
    for vector_kind in ["avg", "one_hot"]:

        # Load the average wallet country embedding
        avg_country_emb = np.load(
            PROCESSED_DATA_PATH
            / f"{vector_kind}_wallet_country_embedding"
            / f"no_china_{kind}.npy"
        )

        # Load the country list
        with open(
            PROCESSED_DATA_PATH / "embedding" / f"anchor_country_{kind}.json"
        ) as f:
            anchor_country = json.load(f)

        df_country = pd.DataFrame(
            {"country": anchor_country, "value": list(avg_country_emb)}
        ).sort_values(by="value", ascending=False)

        # Keep top 20 countries
        df_country = df_country.head(20)

        # Select subplot position
        ax = axes[row_map[kind], col_map[vector_kind]]

        # Plot
        ax.bar(df_country["country"], df_country["value"])
        ax.set_xticklabels(df_country["country"], rotation=90, fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("country share", fontsize=16)
        ax.set_title(f"{vector_kind} wallet country embedding ({kind})", fontsize=18)

plt.tight_layout()
plt.show()
