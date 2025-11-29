"""Script to process the trend data"""

from collections import defaultdict
import json
from gensim.models import Word2Vec, KeyedVectors
import glob
import pandas as pd
import numpy as np
from environ.constant import DATA_PATH, PROCESSED_DATA_PATH
from scripts.merge_traffic_cex import cex_traffic
from environ.utils import (
    cosine_similarity,
    euclidean_distance,
    quadratic_distance,
    softmax,
    standardize,
)

# convert address to exchange mapping
cex_addr2name = {}
for _, row in cex_traffic[["address", "exchange"]].drop_duplicates().iterrows():
    cex_addr2name[row["address"].lower()] = row["exchange"]

country_data = cex_traffic[["exchange", "country", "value"]].copy()

# Load the address -> token mapping
with open(f"{PROCESSED_DATA_PATH}/seed2id.json", "r") as f:
    seed2id = json.load(f)

cex_addr2id = {}
for addr in cex_addr2name:
    if addr in seed2id:
        cex_addr2id[addr] = str(seed2id[addr])

# token -> index mapping
kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv", mmap="r")
vocab_size = len(kv.key_to_index)


DAVE_WALLET = "0x13B42449b151aF160794c0ccbe6038b1E55877F4".lower()
# DAVE_WALLET = "0x3ddfa8ec3052539b6c9549f12cea2c295cff5296".lower()
# DAVE_WALLET = "0x466d7361E33b7d616461f4f0722BcDF59bf6e7E0".lower()

vectors = {k: kv[v] for k, v in cex_addr2id.items() if v in kv}
vectors[DAVE_WALLET] = kv[str(seed2id[DAVE_WALLET])]


# Mutiple embedding
print("Multiple embedding approach")
for func in [cosine_similarity, euclidean_distance, quadratic_distance]:
    addr_sim = {}
    for addr, vec in vectors.items():
        if addr != DAVE_WALLET:
            addr_sim[addr] = func(vectors[DAVE_WALLET], vec)

    addr_sim = pd.DataFrame(
        [(addr, sim) for addr, sim in addr_sim.items()],
        columns=["address", "similarity"],
    )
    # addr_sim["exchange"] = (
    #     addr_sim["address"].map(cex_addr2name).apply(lambda x: x["exchange"])
    # )
    addr_sim["exchange"] = addr_sim["address"].map(cex_addr2name)
    addr_sim = addr_sim.groupby("exchange")["similarity"].mean().reset_index()

    addr_sim = country_data.merge(addr_sim, on="exchange", how="left")
    addr_sim.dropna(inplace=True)

    # sum the adjusted trend by country
    addr_sim["adjusted_value"] = addr_sim["value"] * addr_sim["similarity"]
    addr_sim = addr_sim.groupby("country")["adjusted_value"].mean().reset_index()

    for norm_fun in [softmax, standardize]:
        addr_sim["std_adjusted_value"] = norm_fun(addr_sim["adjusted_value"])

        addr_sim.sort_values("std_adjusted_value", ascending=False, inplace=True)
        print(f"Top 10 countries by {func.__name__} and {norm_fun.__name__}:")
        print(addr_sim.head(5))

# Average embedding
print("Average embedding approach")
avg_vector = defaultdict(list)

for add, idx in cex_addr2id.items():
    exchange = cex_addr2name[add]
    avg_vector[exchange].append(kv[idx])

# average the vectors
for k in avg_vector:
    avg_vector[k] = np.mean(avg_vector[k], axis=0)


for func in [cosine_similarity, euclidean_distance, quadratic_distance]:
    addr_sim = {}
    for exchangeg, vec in avg_vector.items():
        addr_sim[exchangeg] = func(vectors[DAVE_WALLET], vec)

    addr_sim = pd.DataFrame(
        [(exchangeg, sim) for exchangeg, sim in addr_sim.items()],
        columns=["exchange", "similarity"],
    )

    # merge trend data
    addr_sim = country_data.merge(addr_sim, on="exchange", how="left")
    addr_sim.dropna(inplace=True)

    # sum the adjusted trend by country
    addr_sim["adjusted_value"] = addr_sim["value"] * addr_sim["similarity"]
    addr_sim = addr_sim.groupby("country")["adjusted_value"].mean().reset_index()

    for norm_fun in [softmax, standardize]:
        addr_sim["std_adjusted_value"] = norm_fun(addr_sim["adjusted_value"])

        addr_sim.sort_values("std_adjusted_value", ascending=False, inplace=True)
        print(f"Top 10 countries by {func.__name__} and {norm_fun.__name__}:")
        print(addr_sim.head(5))
