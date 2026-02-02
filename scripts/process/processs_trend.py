"""Script to process the trend data"""

from collections import defaultdict
import json
from gensim.models import Word2Vec, KeyedVectors
import glob
import pandas as pd
import numpy as np
from environ.constant import DATA_PATH, PROCESSED_DATA_PATH
from environ.utils import (
    cosine_similarity,
    euclidean_distance,
    quadratic_distance,
    softmax,
    standardize,
    CEX_TREND_MAPPING,
)


# Load the exchange data

cex = pd.read_csv(f"{DATA_PATH}/cex.csv")
cex = cex.loc[cex["network"] == "ETH"]
cex["exchange"] = cex["exchange"].replace(CEX_TREND_MAPPING)
cex_addresses_set = set(cex.loc[cex["network"] == "ETH", "exchange"])

# Load the trend data
trend_list = glob.glob(f"{DATA_PATH}/Trends_Population/Trends/*.csv")
trend_list = set([x.split("/")[-1].split(".")[0] for x in trend_list])

trend_data = []
for trend_name in trend_list.intersection(cex_addresses_set):
    trend = pd.read_csv(
        f"{DATA_PATH}/Trends_Population/Trends/{trend_name}.csv", skiprows=2
    )
    trend.columns = ["country", "trend"]
    trend.replace({"trend": {"<1": 0}}, inplace=True)
    trend.fillna(0, inplace=True)
    trend["trend"] = trend["trend"].astype(int)
    trend["trend"] = trend["trend"] / trend["trend"].sum()
    trend["exchange"] = trend_name
    trend_data.append(trend)

trend_data = pd.concat(trend_data, ignore_index=True)
trend_cex_set = set(trend_data["exchange"])
cex = cex.loc[cex["exchange"].isin(trend_cex_set)]
cex_addr2name = {
    addr: {"exchange": name} for addr, name in zip(cex["address"], cex["exchange"])
}

# Load the address -> token mapping
with open(f"{PROCESSED_DATA_PATH}/seed2id.json", "r") as f:
    seed2id = json.load(f)

cex_addr2id = {}
for addr in cex_addr2name:
    if addr in seed2id:
        cex_addr2id[addr] = str(seed2id[addr])

# token -> index mapping
kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv")
vocab_size = len(kv.key_to_index)


# DAVE_WALLET = "0x13B42449b151aF160794c0ccbe6038b1E55877F4".lower()
# DAVE_WALLET = "0xD00F24c1d859edBc23311E802C32B76274367e8D".lower()
# DAVE_WALLET = "0x466d7361E33b7d616461f4f0722BcDF59bf6e7E0".lower()

DAVE_WALLET = "0x0BED830b5f4528F58B1a6D61763432B91E8Ad18A".lower()

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
    addr_sim["exchange"] = (
        addr_sim["address"].map(cex_addr2name).apply(lambda x: x["exchange"])
    )
    addr_sim = addr_sim.groupby("exchange")["similarity"].mean().reset_index()

    addr_sim = trend_data.merge(addr_sim, on="exchange", how="left")
    addr_sim.dropna(inplace=True)

    # sum the adjusted trend by country
    addr_sim["adjusted_trend"] = addr_sim["trend"] * addr_sim["similarity"]
    addr_sim = addr_sim.groupby("country")["adjusted_trend"].mean().reset_index()

    for norm_fun in [softmax, standardize]:
        addr_sim["std_adjusted_trend"] = norm_fun(addr_sim["adjusted_trend"])

        addr_sim.sort_values("std_adjusted_trend", ascending=False, inplace=True)
        print(f"Top 10 countries by {func.__name__} and {norm_fun.__name__}:")
        print(addr_sim.head(5))

# Average embedding
print("Average embedding approach")
avg_vector = defaultdict(list)

for add, idx in cex_addr2id.items():
    exchange = cex_addr2name[add]["exchange"]
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
    addr_sim = trend_data.merge(addr_sim, on="exchange", how="left")
    addr_sim.dropna(inplace=True)

    # sum the adjusted trend by country
    addr_sim["adjusted_trend"] = addr_sim["trend"] * addr_sim["similarity"]
    addr_sim = addr_sim.groupby("country")["adjusted_trend"].mean().reset_index()

    for norm_fun in [softmax, standardize]:
        addr_sim["std_adjusted_trend"] = norm_fun(addr_sim["adjusted_trend"])

        addr_sim.sort_values("std_adjusted_trend", ascending=False, inplace=True)
        print(f"Top 10 countries by {func.__name__} and {norm_fun.__name__}:")
        print(addr_sim.head(5))
