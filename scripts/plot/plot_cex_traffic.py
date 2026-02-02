"""Script to plot CEX traffic data."""

import pandas as pd
import matplotlib.pyplot as plt
from scripts.merge_traffic_cex import cex_traffic, cex_traffic_china


fig, axes = plt.subplots(2, 2, figsize=(20, 17))
plt.subplots_adjust(hspace=0.4)

row_map = {"no_china": 0, "with_china": 1}
col_map = {"multiple": 0, "pooling": 1}

for china_kind in ["no_china", "with_china"]:
    for traffic_kind in ["multiple", "pooling"]:

        if china_kind == "no_china":
            df_plot = cex_traffic
        else:
            df_plot = cex_traffic_china

        if traffic_kind == "multiple":
            df_plot = (
                df_plot.groupby("country")["value"]
                .mean()
                .reset_index()
                .sort_values(by="value", ascending=False)
            )
        else:
            # deduplicate address-country pairs
            df_plot = (
                df_plot.drop_duplicates(subset=["exchange", "country"])
                .groupby("country")["value"]
                .sum()
                .reset_index()
                .sort_values(by="value", ascending=False)
            )

        # Keep top 20 countries
        df_plot = df_plot.head(20)

        # Select subplot position
        ax = axes[row_map[china_kind], col_map[traffic_kind]]

        # Plot
        ax.bar(df_plot["country"], df_plot["value"])
        ax.set_xticklabels(df_plot["country"], rotation=90, fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel(
            "average traffic" if traffic_kind == "multiple" else "total traffic",
            fontsize=16,
        )
        ax.set_title(f"CEX traffic ({traffic_kind}, {china_kind})", fontsize=18)
