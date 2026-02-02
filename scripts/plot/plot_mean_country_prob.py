"""Script to plot the mean country probability across all wallet embeddings."""

import os

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from environ.constant import PROCESSED_DATA_PATH, CHUNK

os.makedirs(
    f"{PROCESSED_DATA_PATH}/mean_country_prob",
    exist_ok=True,
)

for have_china in ["with_china_9"]:
    for kind in ["multiple"]:
        mean_emb = []
        length = 0
        for chunk in tqdm(range(CHUNK)):
            # Load country probability data for the chunk
            country_prob = np.load(
                PROCESSED_DATA_PATH / f"country/{kind}_{have_china}/{chunk}.npy"
            )

            # Calculate the sum of country probabilities
            mean_emb.append(np.sum(country_prob, axis=0))

            # Keep track of total length
            length += country_prob.shape[0]

        # Calculate the mean country probability
        mean_emb = np.sum(mean_emb, axis=0) / length
        np.save(
            PROCESSED_DATA_PATH / "mean_country_prob" / f"{kind}_{have_china}.npy",
            mean_emb,
        )

        # Load the country list
        anchor_country = np.load(
            PROCESSED_DATA_PATH / f"anchor/country_asc_{have_china}.npy",
        )

        # Create a DataFrame for plotting
        df_country = pd.DataFrame(
            {"country": anchor_country, "value": list(mean_emb)}
        ).sort_values(by="value", ascending=False)

        # Keep top 20 countries
        df_country = df_country.head(20)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(df_country["country"], df_country["value"])
        plt.xticks(rotation=90, fontsize=12)
        plt.ylabel("Mean Country Probability", fontsize=14)
        plt.title(f"Mean Country Probability ({kind}, {have_china})", fontsize=16)
        plt.tight_layout()
        plt.show()
