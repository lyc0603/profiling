"""Script to produce wallet country embedding data"""

import os

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

from environ.constant import PROCESSED_DATA_PATH, CHUNK


def emb_cos_sim(wallet_emb: np.ndarray, anchor_emb: np.ndarray) -> np.ndarray:
    """Calculate the cosine similarity between wallet embedding matrix and anchor embedding matrix."""
    wallet_norm = wallet_emb / np.linalg.norm(wallet_emb, axis=1, keepdims=True)
    anchor_norm = anchor_emb / np.linalg.norm(anchor_emb, axis=1, keepdims=True)

    return wallet_norm @ anchor_norm.T


def country_prob(sim_emb: np.ndarray, country_emb: np.ndarray) -> np.ndarray:
    """Calculate the country probability given the similarity embedding and country embedding."""
    return sim_emb @ country_emb


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


kind = "multiple"
have_china = "with_china_9"

os.makedirs(f"{PROCESSED_DATA_PATH}/country/{kind}_{have_china}", exist_ok=True)

# Load anchor embeddings
anchor_mat = np.load(
    f"{PROCESSED_DATA_PATH}/anchor/anchor_embeddings_{kind}_{have_china}.npy"
)

# Calculate cosine similarity for each chunk
for i in tqdm(range(CHUNK), desc="Calculating Country Probability"):
    kv = KeyedVectors.load(
        f"{PROCESSED_DATA_PATH}/embedding/emb_chunk_{i}.kv", mmap="r"
    )
    wallet_keys = list(kv.key_to_index.keys())  # token name / idx *
    wallet_mat = kv[wallet_keys]  # query using token name *

    emb = emb_cos_sim(wallet_mat, anchor_mat)
    wallet_mat = None

    country_emb = np.load(
        f"{PROCESSED_DATA_PATH}/anchor/anchor_country_{kind}_{have_china}.npy"
    )
    emb = softmax(country_prob(emb, country_emb))

    emb = np.save(f"{PROCESSED_DATA_PATH}/country/{kind}_{have_china}/{i}.npy", emb)
