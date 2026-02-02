"""Script to plot 2D PCA of ground truth wallet embeddings"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm

from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import lightgbm as lgb

from environ.constant import PROCESSED_DATA_PATH


# =========================
# Load mappings & model
# =========================

with open(PROCESSED_DATA_PATH / "seed2id.json", "r") as f:
    seed2id = json.load(f)

kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv", mmap="r")

with open(PROCESSED_DATA_PATH / "ground_truth.json", "r") as f:
    ground_truth = json.load(f)


# =========================
# Build GT embedding matrix
# =========================

labels = []
wallet_embs = []

for country, wallet_list in ground_truth.items():

    if country not in ["China", "United States", "South Korea", "Turkey", "India"]:
        continue
    for wallet in wallet_list:
        if wallet in seed2id:
            labels.append(country)
            wallet_embs.append(kv[str(seed2id[wallet])])

wallet_embs = np.stack(wallet_embs, axis=0)
labels = np.array(labels)

print("GT embedding shape:", wallet_embs.shape)


# =========================
# PCA (2D)
# =========================

pca = PCA(n_components=2, random_state=42)
pca_emb = pca.fit_transform(wallet_embs)

df_pca = pd.DataFrame(
    {
        "pc1": pca_emb[:, 0],
        "pc2": pca_emb[:, 1],
        "country": labels,
    }
)


# =========================
# Plot
# =========================

plt.figure(figsize=(12, 9))

for country in df_pca["country"].unique():
    sub = df_pca[df_pca["country"] == country]
    plt.scatter(
        sub["pc1"],
        sub["pc2"],
        s=20,
        alpha=0.7,
        label=country,
    )

plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.title("2D PCA of Ground Truth Wallet Embeddings")
plt.legend(markerscale=1.5)
plt.tight_layout()
plt.show()


# plot t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=2000,
    init="pca",
    random_state=42,
)


tsne_emb = tsne.fit_transform(wallet_embs)

df_tsne = pd.DataFrame(
    {
        "x": tsne_emb[:, 0],
        "y": tsne_emb[:, 1],
        "country": labels,
    }
)

# -------- Plot --------
plt.figure(figsize=(12, 9))

for country in df_tsne["country"].unique():
    sub = df_tsne[df_tsne["country"] == country]
    plt.scatter(
        sub["x"],
        sub["y"],
        s=20,
        alpha=0.7,
        label=country,
    )

plt.title("t-SNE of Ground Truth Wallet Embeddings")
plt.xlabel("t-SNE-1")
plt.ylabel("t-SNE-2")
plt.legend(markerscale=1.5)
plt.tight_layout()
plt.show()


# Cosine similarity matrix
countries = np.unique(labels)

country_mean_mat = np.stack(
    [wallet_embs[labels == c].mean(axis=0) for c in countries],
    axis=0,
)  # shape = (C, D)

country_mean_mat = country_mean_mat / norm(country_mean_mat, axis=1, keepdims=True)

country_cos_sim = country_mean_mat @ country_mean_mat.T  # (C, C)

df_country_cos = pd.DataFrame(
    country_cos_sim,
    index=countries,
    columns=countries,
)

print("Country Mean Cosine Similarity Matrix:")
print(df_country_cos.round(4))


# Classification experiments
X = wallet_embs
y = labels

# ------------------------------------
# 1. Train/Test split
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ------------------------------------
# 2. Logistic Regression
# ------------------------------------
log_clf = LogisticRegression(
    max_iter=2000,
    multi_class="multinomial",
    solver="lbfgs",
)
log_clf.fit(X_train, y_train)
log_pred = log_clf.predict(X_test)

print("\n===== Logistic Regression =====")
print(classification_report(y_test, log_pred))
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Macro-F1:", f1_score(y_test, log_pred, average="macro"))

# ------------------------------------
# 3. Linear SVM
# ------------------------------------
svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)

print("\n===== Linear SVM =====")
print(classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Macro-F1:", f1_score(y_test, svm_pred, average="macro"))

# ------------------------------------
# 4. LightGBM（非线性）
# ------------------------------------
lgb_clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31)
lgb_clf.fit(X_train, y_train)
lgb_pred = lgb_clf.predict(X_test)

print("\n===== LightGBM =====")
print(classification_report(y_test, lgb_pred))
print("Accuracy:", accuracy_score(y_test, lgb_pred))
print("Macro-F1:", f1_score(y_test, lgb_pred, average="macro"))

# ------------------------------------
# 5. Baseline（随机预测）
# ------------------------------------
n_classes = len(np.unique(y))
baseline = 1 / n_classes
print("\n===== Random baseline =====")
print("Chance accuracy:", baseline)
