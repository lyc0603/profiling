"""Script to evaluate the wallet embedding — save chunks as KV"""

import numpy as np
import json
import os
from gensim.models import KeyedVectors
from tqdm import tqdm

from environ.constant import PROCESSED_DATA_PATH
from scripts.merge_traffic_cex import cex_traffic
from environ.utils import cosine_similarity

CHUNK = 15

for dir in ["embedding", "similarity", "country"]:
    os.makedirs(f"{PROCESSED_DATA_PATH}/{dir}", exist_ok=True)


# Load the full embedding model
kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv", mmap="r")
vocab_keys = list(kv.key_to_index.keys())
vocab_size = len(vocab_keys)
chunk_size = vocab_size // CHUNK

# for i in tqdm(range(CHUNK), desc="Saving KV chunks"):
#     start_idx = i * chunk_size
#     end_idx = (i + 1) * chunk_size if i < CHUNK - 1 else vocab_size

#     sub_keys = vocab_keys[start_idx:end_idx]
#     sub_vectors = [kv[key] for key in sub_keys]

#     # Create a new KeyedVectors object for the chunk
#     kv_chunk = KeyedVectors(vector_size=kv.vector_size)
#     kv_chunk.add_vectors(sub_keys, sub_vectors)

#     # Save as a .kv file
#     out_path = f"{PROCESSED_DATA_PATH}/embedding/emb_chunk_{i}.kv"
#     kv_chunk.save(out_path)

#     print(f"Chunk {i}: saved {len(sub_keys)} vectors → {out_path}")

# # Load anchor embeddings
# anchor_mat = np.load(f"{PROCESSED_DATA_PATH}/embedding/anchor_embeddings.npy")

# # Calculate cosine similarity for each chunk
# for i in tqdm(range(CHUNK), desc="Calculating Cosine Similarity"):
#     kv = KeyedVectors.load(
#         f"{PROCESSED_DATA_PATH}/embedding/emb_chunk_{i}.kv", mmap="r"
#     )
#     wallet_keys = list(kv.key_to_index.keys())
#     wallet_mat = kv[wallet_keys]  # shape: (N_wallets, D)

#     # Normalize (vectorized)
#     wallet_norm = wallet_mat / np.linalg.norm(wallet_mat, axis=1, keepdims=True)
#     anchor_norm = anchor_mat / np.linalg.norm(anchor_mat, axis=1, keepdims=True)

#     # Cosine similarity matrix (N_wallets × A)
#     sim = wallet_norm @ anchor_norm.T

#     # Save the similarity matrix
#     out_path = f"{PROCESSED_DATA_PATH}/similarity/cosine_similarity_{i}.npy"
#     np.save(out_path, sim)


# # Load the country embeddings
# country_emb = np.load(f"{PROCESSED_DATA_PATH}/embedding/anchor_country_embeddings.npy")

# for i in tqdm(range(CHUNK), desc="Calculating Wallet Country Embeddings"):
#     sim = np.load(f"{PROCESSED_DATA_PATH}/similarity/cosine_similarity_{i}.npy")

#     # Normalized each row
#     row_sums = sim.sum(axis=1, keepdims=True)
#     sim_normalized = sim / row_sums

#     # Weighted country embeddings
#     wallet_country_emb = sim_normalized @ country_emb

#     # Save the wallet country embeddings
#     out_path = f"{PROCESSED_DATA_PATH}/country/wallet_country_embedding_{i}.npy"
#     np.save(out_path, wallet_country_emb)


# Load the wallet country embeddings
wallet_country_list = []

for i in tqdm(range(CHUNK), desc="Merging Wallet Country Embeddings"):
    wallet_country_emb = np.load(
        f"{PROCESSED_DATA_PATH}/country/wallet_country_embedding_{i}.npy"
    )
    # get the largest country index
    wallet_country_emb = np.argmax(wallet_country_emb, axis=1)

    for wallet_country in wallet_country_emb:
        wallet_country_list.append(int(wallet_country))

token_wallet_country = {}
for token, country in zip(kv.index_to_key, wallet_country_list):
    token_wallet_country[token] = country

# Load address to token mapping
with open(f"{PROCESSED_DATA_PATH}/seed2id.json", "r") as f:
    seed2id = json.load(f)

address_wallet_country = {}
for addr, token_id in tqdm(seed2id.items()):
    token_id_str = str(token_id)
    address_wallet_country[addr] = token_wallet_country[token_id_str]

# Save the address to wallet country mapping
with open(f"{PROCESSED_DATA_PATH}/address_wallet_country.json", "w") as f:
    json.dump(address_wallet_country, f)
