"""Script to evaluate the accuracy of the embedding"""

import json
import os

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

from environ.constant import PROCESSED_DATA_PATH


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


# Load the address -> string id mapping
with open(f"{PROCESSED_DATA_PATH}/seed2id.json", "r") as f:
    seed2id = json.load(f)

# Load the full embedding model
kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv", mmap="r")

# Load ground truth country data
with open(PROCESSED_DATA_PATH / "ground_truth.json", "r") as f:
    ground_truth = json.load(f)

df_accs = []
df_preds = []

for have_china in ["without_china", "with_china_5", "with_china_9"]:
    for kind in ["multiple", "mean"]:

        gt_country_vector = []
        gt_wallet_embedding = []
        for country, wallet_list in ground_truth.items():
            if (country == "China") & (have_china == "without_china"):
                continue
            for wallet in wallet_list:
                if wallet in seed2id:
                    gt_country_vector.append(country)
                    gt_wallet_embedding.append(kv[str(seed2id[wallet])])

        gt_wallet_embedding = np.stack(gt_wallet_embedding, axis=0)

        # Calculate the cosine similarity
        anchor_mat = np.load(
            f"{PROCESSED_DATA_PATH}/anchor/anchor_embeddings_{kind}_{have_china}.npy"
        )
        gt_similarity = emb_cos_sim(gt_wallet_embedding, anchor_mat)

        # Calculate the country probabilities
        country_emb = np.load(
            f"{PROCESSED_DATA_PATH}/anchor/anchor_country_{kind}_{have_china}.npy"
        )
        gt_wallet_country_emb = softmax(country_prob(gt_similarity, country_emb))

        # Get the country confidence
        country_list = np.load(
            f"{PROCESSED_DATA_PATH}/anchor/country_asc_{have_china}.npy"
        ).tolist()

        df = {
            "label": [],
            "prediction": [],
            "predicted_confidence": [],
            "correct_confidence": [],
        }
        for country, country_vec in zip(gt_country_vector, gt_wallet_country_emb):
            predicted_idx = np.argmax(country_vec)
            predicted_country = country_list[predicted_idx]
            correct_country_idx = country_list.index(country)

            correct_confidence = country_vec[correct_country_idx]
            predicted_confidence = country_vec[predicted_idx]

            df["label"].append(country)
            df["prediction"].append(predicted_country)
            df["predicted_confidence"].append(predicted_confidence)
            df["correct_confidence"].append(correct_confidence)

        df = pd.DataFrame(df)
        df["correct"] = df["label"] == df["prediction"]

        acc_country = ["United States", "South Korea", "China"]
        if have_china == "without_china":
            acc_country.remove("China")

        df_pred = df["prediction"].value_counts().reset_index().copy()
        df_pred = df_pred.loc[df_pred["prediction"].isin(acc_country)]

        df_acc = df.loc[df["label"].isin(acc_country)].copy()
        df_acc = (
            df_acc.groupby(["label"])["correct"]
            .mean()
            .reset_index()
            .rename({"label": "country", "correct": "accuracy"}, axis=1)
        )
        df_acc["sample"] = kind + "_" + have_china
        df_pred["sample"] = kind + "_" + have_china

        df_accs.append(df_acc)
        df_preds.append(df_pred)

df_accs = pd.concat(df_accs, axis=0)
df_accs.to_csv(
    os.path.join(PROCESSED_DATA_PATH, "acc.csv"),
    index=False,
)

df_preds = pd.concat(df_preds, axis=0)
df_preds.to_csv(
    os.path.join(PROCESSED_DATA_PATH, "preds.csv"),
    index=False,
)
