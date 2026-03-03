"""Script to train XGBoost multi-classifier with hyperparameter tuning."""

import os
import json
import pickle
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from environ.constant import PROCESSED_DATA_PATH

np.random.seed(42)
os.makedirs(PROCESSED_DATA_PATH / "clf_model", exist_ok=True)

# -----------------------
# Load data
# -----------------------
features = np.load(PROCESSED_DATA_PATH / "train" / "features.npy")
gt = np.load(PROCESSED_DATA_PATH / "train" / "gt.npy")

class_counts = Counter(gt)
print("Class distribution in the dataset:")
for cls, count in class_counts.items():
    print(f"Class {cls}: {count} samples")

TRAINING_SET_SIZE_PER_CLASS = 10000
TEST_SET_SIZE_PER_CLASS = 10000

train_indices, test_indices = [], []
unique_classes = np.unique(gt)

for cls in unique_classes:
    cls_indices = np.where(gt == cls)[0]
    np.random.shuffle(cls_indices)

    train_cls = cls_indices[:TRAINING_SET_SIZE_PER_CLASS]
    test_cls = cls_indices[
        TRAINING_SET_SIZE_PER_CLASS : TRAINING_SET_SIZE_PER_CLASS
        + TEST_SET_SIZE_PER_CLASS
    ]

    train_indices.extend(train_cls)
    test_indices.extend(test_cls)

train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

X_train = features[train_indices]
y_train = gt[train_indices]
X_test = features[test_indices]
y_test = gt[test_indices]

print("Final dataset sizes:")
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

# -----------------------
# Encode labels
# -----------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
n_classes = len(le.classes_)
print("Num classes:", n_classes)

# -----------------------
# Base model
# -----------------------
base = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",  # change to "gpu_hist" if you have CUDA
    device="cuda",
    random_state=42,
    n_jobs=1,
)

# -----------------------
# Hyperparameter search space
# (Keep it modest first; expand later.)
# -----------------------
param_dist = {
    "n_estimators": [300, 500, 800, 1200],
    "max_depth": [4, 6, 7, 9, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 8, 12],
    "gamma": [0.0, 0.1, 0.2, 0.5, 1.0],
    "reg_alpha": [0.0, 1e-4, 1e-3, 1e-2, 0.1],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}

# -----------------------
# CV + Random Search
# -----------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_dist,
    n_iter=40,  # increase to 100+ if you can afford it
    scoring="f1_macro",  # key choice
    cv=cv,
    verbose=2,
    n_jobs=1,
    refit=True,  # refit best model on full training set
    random_state=42,
)

# IMPORTANT: for XGBoost multi-class, pass num_class via set_params
# (XGBClassifier usually infers, but being explicit avoids edge cases.)
search.estimator.set_params(num_class=n_classes)

print("\n--- Tuning (RandomizedSearchCV) ---")
search.fit(X_train, y_train_enc)

print("\nBest CV score (macro-F1):", search.best_score_)
print("Best params:", search.best_params_)

best_model = search.best_estimator_

# -----------------------
# Test evaluation
# -----------------------
y_pred_enc = best_model.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)

accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average="weighted")
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n--- Test ---")
print("Accuracy:", accuracy)
print("Weighted F1:", f1_weighted)
print("Macro F1:", f1_macro)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------
# Confusion matrix plot (normalized recall per class)
# -----------------------
labels = le.classes_
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm_norm, cmap="Blues")

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Normalized Confusion Matrix (Recall per Class)")

for i in range(len(labels)):
    for j in range(len(labels)):
        val = cm_norm[i, j]
        if not np.isnan(val):
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
            )

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# -----------------------
# Save artifacts
# -----------------------
out_dir = PROCESSED_DATA_PATH / "clf_model"
with open(out_dir / "best_xgb_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(out_dir / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open(out_dir / "best_xgb_params.json", "w", encoding="utf-8") as f:
    json.dump(search.best_params_, f, indent=2)

print("\nSaved:")
print("-", out_dir / "best_xgb_model.pkl")
print("-", out_dir / "label_encoder.pkl")
print("-", out_dir / "best_xgb_params.json")
