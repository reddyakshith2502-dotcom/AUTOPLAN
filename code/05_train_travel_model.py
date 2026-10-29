from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "processed" / "travel_time_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "travel_time_model.pkl"


print("================================")
print("TRAVEL-TIME ML MODEL TRAINING")
print("================================")


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading travel-time dataset...")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Vehicles:", df["Vehicle_ID"].nunique())


# ==========================================
# FEATURES
# ==========================================

features = [
    "speed_mps",
    "acceleration_mps2",
    "speed_change",
    "previous_speed",
    "previous_acceleration",
    "space_headway",
    "time_headway",
    "traffic_density",
    "relative_speed",
    "lane",
    "vehicle_type"
]

target = "future_travel_time"


X = df[features]
y = df[target]

groups = df["Vehicle_ID"]


print("\n================================")
print("FEATURES")
print("================================")

for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

print("\n================================")
print("TRAIN / TEST SPLIT")
print("================================")

# Split by vehicle instead of random rows.
# This prevents the same vehicle appearing
# in both training and testing data.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
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
# TRAIN RANDOM FOREST
# ==========================================

print("\n================================")
print("TRAINING RANDOM FOREST")
print("================================")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed ✅")


# ==========================================
# PREDICTION
# ==========================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ==========================================
# EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    f"MAE  : {mae:.4f} seconds"
)

print(
    f"RMSE : {rmse:.4f} seconds"
)

print(
    f"R²   : {r2:.4f}"
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

print(importance.to_string(index=False))


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
print("MODEL SAVED ✅")
print("================================")

print("\nSaved:")
print(MODEL_FILE)

print("\n================================")
print("STEP 5 COMPLETED ✅")
print("================================")