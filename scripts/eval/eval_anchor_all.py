"""
Final script for wallet country inference
with Inverse-Frequency Reweighting (IFR)

Features:
- Full 100+ country evaluation
- Core-6 focused evaluation
- Top-1 / Top-3 / Brier / Cross-Entropy
- Random baseline comparison
- Inverse-frequency reweighting to correct anchor-country imbalance
"""

import json
import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report

from environ.constant import PROCESSED_DATA_PATH


# =========================
# CONFIG
# =========================
CORE_COUNTRIES = ["China", "United States", "South Korea", "Turkey", "India", "Japan"]
HAVE_CHINA_LIST = ["without_china", "with_china_5", "with_china_9"]
KIND_LIST = ["multiple", "mean"]

IFR_ALPHA = 1.0  # 控制逆频率重加权强度（推荐 0.5–1.0）
EPS = 1e-12  # 防止除零


# =========================
# UTILS
# =========================


def emb_cos_sim(wallet_emb: np.ndarray, anchor_emb: np.ndarray) -> np.ndarray:
    wallet_norm = wallet_emb / np.linalg.norm(wallet_emb, axis=1, keepdims=True)
    anchor_norm = anchor_emb / np.linalg.norm(anchor_emb, axis=1, keepdims=True)
    return wallet_norm @ anchor_norm.T


def country_prob(sim_emb: np.ndarray, country_emb: np.ndarray) -> np.ndarray:
    return sim_emb @ country_emb


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)


### [NEW] ———————— Inverse-Frequency Reweighting ————————
def reweight_country_emb(
    country_emb: np.ndarray, alpha: float = IFR_ALPHA
) -> np.ndarray:
    """
    country_emb shape: [A, C]
    Apply inverse-frequency reweighting to balance countries.
    """
    # 每国在 anchor 中的平均概率 f_c
    freq = country_emb.mean(axis=0)  # shape: [C]

    # 构造逆频率权重
    inv_weight = 1 / (np.power(freq + EPS, alpha))

    # 对每一行应用重加权
    reweighted = country_emb * inv_weight[None, :]

    # 行归一化保持概率结构
    reweighted /= reweighted.sum(axis=1, keepdims=True)

    return reweighted


### [NEW END] ————————————————————————————————


# =========================
# LOAD MODEL / DATA
# =========================
with open(PROCESSED_DATA_PATH / "seed2id.json", "r") as f:
    seed2id = json.load(f)

kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv", mmap="r")

with open(PROCESSED_DATA_PATH / "ground_truth.json", "r") as f:
    ground_truth = json.load(f)


# =========================
# STORAGE
# =========================
all_metrics = []
all_core_acc = []
all_pred_dist = []


# =========================
# MAIN LOOP
# =========================

for have_china in HAVE_CHINA_LIST:
    for kind in KIND_LIST:

        print(f"\n===== Evaluating: {kind} | {have_china} =====")

        # ------------------------------------------------
        # 1. Load ground truth embeddings
        # ------------------------------------------------
        gt_labels = []
        gt_wallet_emb = []

        for country, wallet_list in ground_truth.items():
            if (country == "China") and (have_china == "without_china"):
                continue
            for wallet in wallet_list:
                if wallet in seed2id:
                    gt_labels.append(country)
                    gt_wallet_emb.append(kv[str(seed2id[wallet])])

        gt_wallet_emb = np.stack(gt_wallet_emb, axis=0)
        gt_labels = np.array(gt_labels)

        # ------------------------------------------------
        # 2. Load anchors
        # ------------------------------------------------
        anchor_mat = np.load(
            f"{PROCESSED_DATA_PATH}/anchor/anchor_embeddings_{kind}_{have_china}.npy"
        )

        country_emb = np.load(
            f"{PROCESSED_DATA_PATH}/anchor/anchor_country_{kind}_{have_china}.npy"
        )

        country_list = np.load(
            f"{PROCESSED_DATA_PATH}/anchor/country_asc_{have_china}.npy"
        ).tolist()

        # ------------------------------------------------
        # 3. [NEW] Apply inverse-frequency reweighting
        # ------------------------------------------------
        country_emb = reweight_country_emb(country_emb, alpha=IFR_ALPHA)

        # ------------------------------------------------
        # 4. Predict wallet → country
        # ------------------------------------------------
        gt_similarity = emb_cos_sim(gt_wallet_emb, anchor_mat)
        gt_wallet_country_prob = softmax(country_prob(gt_similarity, country_emb))

        # ------------------------------------------------
        # 5. Build evaluation DF
        # ------------------------------------------------
        df = {
            "label": [],
            "prediction": [],
            "predicted_confidence": [],
            "correct_confidence": [],
            "top1_correct": [],
            "top3_correct": [],
            "brier": [],
            "cross_entropy": [],
        }

        for true_country, prob_vec in zip(gt_labels, gt_wallet_country_prob):
            pred_idx = int(np.argmax(prob_vec))
            pred_country = country_list[pred_idx]
            true_idx = country_list.index(true_country)

            top3_idx = np.argsort(prob_vec)[-3:][::-1]
            top3_countries = [country_list[i] for i in top3_idx]

            df["label"].append(true_country)
            df["prediction"].append(pred_country)
            df["predicted_confidence"].append(prob_vec[pred_idx])
            df["correct_confidence"].append(prob_vec[true_idx])

            df["top1_correct"].append(pred_country == true_country)
            df["top3_correct"].append(true_country in top3_countries)

            true_onehot = np.zeros(len(country_list))
            true_onehot[true_idx] = 1.0

            df["brier"].append(np.mean((prob_vec - true_onehot) ** 2))
            df["cross_entropy"].append(-np.log(prob_vec[true_idx] + 1e-12))

        df = pd.DataFrame(df)

        # ------------------------------------------------
        # 6. Metrics for ALL countries
        # ------------------------------------------------
        metrics_all = {
            "sample": f"{kind}_{have_china}",
            "scope": "ALL",
            "n_class": len(country_list),
            "top1_acc": df["top1_correct"].mean(),
            "top3_acc": df["top3_correct"].mean(),
            "mean_brier": df["brier"].mean(),
            "mean_cross_entropy": df["cross_entropy"].mean(),
            "random_baseline": 1 / len(country_list),
        }
        all_metrics.append(metrics_all)

        # ------------------------------------------------
        # 7. CORE-6 metrics
        # ------------------------------------------------
        df_core = df[df["label"].isin(CORE_COUNTRIES)].copy()

        if len(df_core) > 0:
            metrics_core = {
                "sample": f"{kind}_{have_china}",
                "scope": "CORE_6",
                "n_class": len(CORE_COUNTRIES),
                "top1_acc": df_core["top1_correct"].mean(),
                "top3_acc": df_core["top3_correct"].mean(),
                "mean_brier": df_core["brier"].mean(),
                "mean_cross_entropy": df_core["cross_entropy"].mean(),
                "random_baseline": 1 / len(CORE_COUNTRIES),
            }
            all_metrics.append(metrics_core)

            df_acc_core = (
                df_core.groupby("label")["top1_correct"]
                .mean()
                .reset_index()
                .rename(columns={"top1_correct": "top1_accuracy"})
            )
            df_acc_core["sample"] = f"{kind}_{have_china}"
            all_core_acc.append(df_acc_core)

        # ------------------------------------------------
        # 8. Prediction distribution (for collapse check)
        # ------------------------------------------------
        df_pred_core = (
            df[df["prediction"].isin(CORE_COUNTRIES)]["prediction"]
            .value_counts(normalize=True)
            .reset_index()
            .rename(columns={"index": "predicted_country", "prediction": "share"})
        )
        df_pred_core["sample"] = f"{kind}_{have_china}"
        all_pred_dist.append(df_pred_core)

        if len(df_core) > 0:
            print("\n----- Classification Report (CORE-6) -----")
            print(
                classification_report(df_core["label"], df_core["prediction"], digits=4)
            )


# =========================
# SAVE RESULTS
# =========================
df_metrics = pd.DataFrame(all_metrics)
df_metrics.to_csv(
    os.path.join(PROCESSED_DATA_PATH, "eval/metrics_summary_IFR.csv"), index=False
)

if len(all_core_acc) > 0:
    df_core_acc = pd.concat(all_core_acc, axis=0)
    df_core_acc.to_csv(
        os.path.join(PROCESSED_DATA_PATH, "eval/acc_core6_IFR.csv"), index=False
    )

if len(all_pred_dist) > 0:
    df_pred_dist = pd.concat(all_pred_dist, axis=0)
    df_pred_dist.to_csv(
        os.path.join(PROCESSED_DATA_PATH, "eval/pred_dist_core6_IFR.csv"), index=False
    )

print("\nSaved:")
print("- metrics_summary_IFR.csv")
print("- acc_core6_IFR.csv")
print("- pred_dist_core6_IFR.csv")
