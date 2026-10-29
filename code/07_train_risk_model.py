from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "processed" / "risk_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "collision_risk_model.pkl"


print("================================")
print("COLLISION-RISK ML MODEL")
print("================================")


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading risk dataset...")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Vehicles:", df["Vehicle_ID"].nunique())


# ==========================================
# FEATURES
# ==========================================

features = [
    "ego_speed_mps",
    "preceding_speed_mps",
    "closing_speed_mps",
    "v_Acc",
    "gap_m",
    "Time_Hdwy",
    "Lane_ID",
    "v_Class"
]

target = "risk_label"

X = df[features]
y = df[target]

groups = df["Vehicle_ID"]


# ==========================================
# DATA DISTRIBUTION
# ==========================================

print("\n================================")
print("CLASS DISTRIBUTION")
print("================================")

print(
    y.value_counts()
)

print("\nPercentages:")

print(
    y.value_counts(normalize=True).mul(100)
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

print("\n================================")
print("TRAIN / TEST SPLIT")
print("================================")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Training vehicles:",
    df.iloc[train_idx]["Vehicle_ID"].nunique()
)

print(
    "Testing vehicles:",
    df.iloc[test_idx]["Vehicle_ID"].nunique()
)


# ==========================================
# MODEL
# ==========================================

print("\n================================")
print("TRAINING RANDOM FOREST CLASSIFIER")
print("================================")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed [OK]")


# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# RESULTS
# ==========================================

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {auc:.4f}"
)


print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "SAFE",
            "HIGH RISK"
        ],
        zero_division=0
    )
)


print("\n================================")
print("CONFUSION MATRIX")
print("================================")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(
    importance.to_string(
        index=False
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)


print("\n================================")
print("MODEL SAVED [OK]")
print("================================")

print("\nSaved:")
print(MODEL_FILE)

print("\n================================")
print("STEP 7 COMPLETED [OK]")
print("================================")
