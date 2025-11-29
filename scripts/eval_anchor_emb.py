"""Script to process the anchor embedding data"""

import numpy as np
import json
import os
from gensim.models import KeyedVectors

from environ.constant import PROCESSED_DATA_PATH
from scripts.merge_traffic_cex import cex_traffic

CHUNK = 15

OUTPUT_DIR = f"{PROCESSED_DATA_PATH}/embedding"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the address -> string id mapping
with open(f"{PROCESSED_DATA_PATH}/seed2id.json", "r") as f:
    seed2id = json.load(f)

# Load the full embedding model
kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv", mmap="r")

# Multiple anchor embedding
print("Processing anchor embeddings")

# Get the list of CEX address and country in asending order
address_asc = sorted(cex_traffic["address"].unique())
address_asc = [addr for addr in address_asc if addr in seed2id]
country_asc = sorted(cex_traffic["country"].unique())

anchor_emb = []
for addr in address_asc:
    anchor_emb.append(kv[str(seed2id[addr])])

anchor_emb = np.stack(anchor_emb, axis=0)

anchor_country = []
for addr in address_asc:
    addr_data = cex_traffic.loc[cex_traffic["address"] == addr]
    addr_country_emb = []
    for country in country_asc:
        val = addr_data.loc[addr_data["country"] == country, "value"]
        addr_country_emb.append(val.values[0])
    anchor_country.append(addr_country_emb)

anchor_country = np.array(anchor_country)

# save the anchor embeddings
np.save(f"{OUTPUT_DIR}/anchor_embeddings.npy", anchor_emb)

# save the address_asc
with open(f"{OUTPUT_DIR}/anchor_address.json", "w") as f:
    json.dump(address_asc, f, indent=2)

# save the country embeddings
np.save(f"{OUTPUT_DIR}/anchor_country_embeddings.npy", anchor_country)

# save the country_asc
with open(f"{OUTPUT_DIR}/anchor_country.json", "w") as f:
    json.dump(country_asc, f, indent=2)
