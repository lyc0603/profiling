"""Script to preprocess ground truth data for training."""

import os
import numpy as np
import pandas as pd
import json
from gensim.models import KeyedVectors
from environ.constant import PROCESSED_DATA_PATH
from tqdm import tqdm

# Load the ENS ground truth data
with open(PROCESSED_DATA_PATH / "ground_truth.json", "r") as f:
    ens_ground_truth = json.load(f)


# Load the early stage ground truth data
with open(PROCESSED_DATA_PATH / "early_stage_gt.json", "r") as f:
    early_stage_ground_truth = json.load(f)

# get the set of duplicated wallets in both ground truth datasets
ens_wallets = set()
for wallet_list in ens_ground_truth.values():
    ens_wallets.update(wallet_list)

early_stage_wallets = set(early_stage_ground_truth.keys())
duplicated_wallets = ens_wallets.intersection(early_stage_wallets)

# Load the graph embedding model
with open(PROCESSED_DATA_PATH / "seed2id.json", "r") as f:
    seed2id_graph = json.load(f)
kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv", mmap="r")

# Load the timezone model
with open(PROCESSED_DATA_PATH / "timezone" / "seed2id.json", "r") as f:
    seed2id_timezone = json.load(f)
timezone_vectors = np.load(PROCESSED_DATA_PATH / "timezone" / "timezone.npy")

# Load the country to region mapping
with open(PROCESSED_DATA_PATH / "anchor_country_region.json", "r") as f:
    country_region_mapping = json.load(f)

features = []
gt = []

# Process ENS ground truth wallet embedding data
for country, wallet_list in tqdm(ens_ground_truth.items()):
    for wallet in wallet_list:
        feature = []
        if wallet in seed2id_graph:
            feature.extend(kv[str(seed2id_graph[wallet])].tolist())
            feature.extend(timezone_vectors[seed2id_timezone[wallet]].tolist())
            features.append(feature)
            gt.append(country_region_mapping[country])
        else:
            continue


# Process early stage ground truth wallet embedding data
for wallet, country in tqdm(early_stage_ground_truth.items()):
    feature = []
    feature.extend(kv[str(seed2id_graph[wallet])].tolist())
    feature.extend(timezone_vectors[seed2id_timezone[wallet]].tolist())
    features.append(feature)
    gt.append(country_region_mapping[country])

os.makedirs(PROCESSED_DATA_PATH / "train", exist_ok=True)
np.save(PROCESSED_DATA_PATH / "train" / "features.npy", np.array(features))
np.save(PROCESSED_DATA_PATH / "train" / "gt.npy", np.array(gt))
