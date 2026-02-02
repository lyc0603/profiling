"""Script to train random forest multi-classifier on the processed features."""

import os
import pickle

import numpy as np
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    ParameterGrid,
    ParameterSampler,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from sklearn.ensemble import RandomForestClassifier
from environ.constant import PROCESSED_DATA_PATH
from tqdm import tqdm


os.makedirs(PROCESSED_DATA_PATH / "clf_model", exist_ok=True)

# Load features npy
features = np.load(PROCESSED_DATA_PATH / "train" / "features.npy")

# Load gt npy
gt = np.load(PROCESSED_DATA_PATH / "train" / "gt.npy")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, gt, test_size=0.2, random_state=42, stratify=gt, train_size=0.3
)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
base = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)

param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.2, 0.5],
}

grid = ParameterSampler(param_dist, n_iter=20, random_state=42)
# grid = ParameterGrid(param_dist)

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


best_score = 0
best_params = None

for params in tqdm(grid, desc="Grid Search"):
    fold_scores = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model = RandomForestClassifier(
            **params, random_state=42, class_weight="balanced", n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)
        fold_f1 = f1_score(y_val, y_val_pred, average="weighted")
        fold_scores.append(fold_f1)

    avg_score = np.mean(fold_scores)
    if avg_score > best_score:
        best_score = avg_score
        best_params = params

print("Best parameters found:", best_params)
best_clf = RandomForestClassifier(
    **best_params, random_state=42, class_weight="balanced", n_jobs=-1
)
best_clf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = best_clf.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Weighted F1 Score:", f1)
print("Classification Report:\n", report)


with open(PROCESSED_DATA_PATH / "clf_model" / "best_rf_model.pkl", "wb") as f:
    pickle.dump(best_clf, f)


# # Initialize and train the Random Forest Classifier with progress bar
# clf = RandomForestClassifier(
#     n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
# )

# clf.fit(X_train, y_train)
# # Make predictions on the test set
# y_pred = clf.predict(X_test)
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average="weighted")
# report = classification_report(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Weighted F1 Score:", f1)
# print("Classification Report:\n", report)

# # Summary of predictions


# # Get sorted labels (consistent order)
# labels = np.unique(gt)

# # Compute confusion matrix
# cm = confusion_matrix(y_test, y_pred, labels=labels)

# # Normalize by true class (row-wise)
# cm_norm = cm / cm.sum(axis=1, keepdims=True)

# # Plot
# fig, ax = plt.subplots(figsize=(9, 7))
# im = ax.imshow(cm_norm, cmap="Blues")

# # Ticks and labels
# ax.set_xticks(np.arange(len(labels)))
# ax.set_yticks(np.arange(len(labels)))
# ax.set_xticklabels(labels, rotation=45, ha="right")
# ax.set_yticklabels(labels)

# ax.set_xlabel("Predicted label")
# ax.set_ylabel("True label")
# ax.set_title("Normalized Confusion Matrix (Recall per Class)")

# # Annotate cells
# for i in range(len(labels)):
#     for j in range(len(labels)):
#         value = cm_norm[i, j]
#         if not np.isnan(value):
#             ax.text(
#                 j,
#                 i,
#                 f"{value:.2f}",
#                 ha="center",
#                 va="center",
#                 color="white" if value > 0.5 else "black",
#             )

# # Colorbar
# fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()
