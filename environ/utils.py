"""Utility functions"""

import heapq
import re

import numpy as np
import jellyfish

# Trend
CEX_TREND_MAPPING = {
    "Biconomy": "Biconomy_com",
    "Coinbase": "Coinbase Exchange",
    "Crypto.com Exchange": "Crypto_com",
    "Gate.io": "Gate",
    "XT.COM": "XT_com",
}


# Embedding
# function to calculate the cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def euclidean_distance(vec1, vec2):
    return 1 / np.linalg.norm(vec1 - vec2)


def quadratic_distance(vec1, vec2):
    return euclidean_distance(vec1, vec2) ** 2


# softmax function
def softmax(x, tau=0.1):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()


def standardize(x):
    demin = x - x.min()
    return demin / demin.sum()


# Fuzzy Matching
PUNC = """ |,|\.|;|:|\||\(|\)|&|'|"|\+|-|/"""

REMOVE = [
    "",
]


def clean_name(name: str) -> str:
    """Clean the name by removing special characters and lowercasing."""

    # lowercase
    name = name.lower()
    # replace special characters with space
    name = name.replace("(", "").replace(")", "")
    # split by punctuation
    wname = re.split(PUNC, name)

    return " ".join([w for w in wname if w]).strip()


def match_top(s, candidates, n=1, method="lev"):
    """Match the top n candidates from the list."""
    heap = [(-np.inf, "") for _ in range(n)]
    heapq.heapify(heap)

    for t in candidates:
        l = len(t)
        if method == "lev":
            score = jellyfish.levenshtein_distance(s, t) / max(l, len(s)) - 1
        elif method == "dam_lev":
            score = jellyfish.damerau_levenshtein_distance(s, t) / max(l, len(s)) - 1
        elif method == "jaro":
            score = -jellyfish.jaro_distance(s, t)
        elif method == "jaro_win":
            score = -jellyfish.jaro_winkler_similarity(s, t)

        heapq.heappushpop(heap, (-score, t))

    heap.sort(reverse=True)
    return heap
