"""Script to generate the anchor embedding data"""

import numpy as np
import json
import os
from gensim.models import KeyedVectors

from environ.constant import PROCESSED_DATA_PATH
from scripts.merge_traffic_cex import (
    cex_traffic,
    cex_traffic_china_5,
    cex_traffic_china_9,
)

OUTPUT_DIR = f"{PROCESSED_DATA_PATH}/anchor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the full embedding model
kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv", mmap="r")

# Load the address -> string id mapping
with open(f"{PROCESSED_DATA_PATH}/seed2id.json", "r") as f:
    seed2id = json.load(f)

for name, df in {
    "without_china": cex_traffic,
    "with_china_9": cex_traffic_china_9,
    "with_china_5": cex_traffic_china_5,
}.items():
    # Get the ascending lists of address, country, and CEX *
    address_asc = sorted(df["address"].unique())
    address_asc = [addr for addr in address_asc if addr in seed2id]
    df = df[df["address"].isin(address_asc)]
    country_asc = sorted(df["country"].unique())
    exchange_asc = sorted(df["exchange"].unique())

    country_asc = np.array(country_asc)
    np.save(f"{OUTPUT_DIR}/country_asc_{name}.npy", country_asc)

    # Generate multiple anchor embeddings *
    anchor_emb = np.stack([kv[str(seed2id[addr])] for addr in address_asc], axis=0)

    anchor_country = np.stack(
        [
            df.loc[df["address"] == addr]
            .set_index("country")
            .reindex(country_asc)["value"]
            .values
            for addr in address_asc
        ],
        axis=0,
    )

    anchor_emb = np.save(
        f"{OUTPUT_DIR}/anchor_embeddings_multiple_{name}.npy", anchor_emb
    )
    anchor_country = np.save(
        f"{OUTPUT_DIR}/anchor_country_multiple_{name}.npy", anchor_country
    )

    # Generate mean pooled anchor embeddings *
    anchor_emb = []
    anchor_country = []
    for exch in exchange_asc:
        exch_df = df.loc[df["exchange"] == exch]
        exch_addrs = [addr for addr in exch_df["address"].unique() if addr in seed2id]
        # mean pooled embedding
        exchange_embs = np.stack(
            [kv[str(seed2id[addr])] for addr in exch_addrs], axis=0
        )
        exchange_emb = np.mean(exchange_embs, axis=0)
        anchor_emb.append(exchange_emb)

        # address-country matrix
        exch_country_vec = (
            exch_df.groupby("country")["value"].mean().reindex(country_asc).values
        )
        anchor_country.append(exch_country_vec)

    anchor_emb = np.stack(anchor_emb, axis=0)
    anchor_country = np.stack(anchor_country, axis=0)

    np.save(f"{OUTPUT_DIR}/anchor_embeddings_mean_{name}.npy", anchor_emb)
    np.save(f"{OUTPUT_DIR}/anchor_country_mean_{name}.npy", anchor_country)
