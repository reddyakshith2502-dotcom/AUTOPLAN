from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "processed" / "ngsim_features.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "collision_risk_dataset.csv"

print("==================================================")
print("COLLISION-RISK TARGET & DATASET GENERATION")
print("==================================================")

print("\nLoading NGSIM feature dataset...")
df = pd.read_csv(INPUT_FILE)

print("Total Rows:", len(df))
print("Total Vehicles:", df["Vehicle_ID"].nunique())


# ==========================================
# SURROGATE SAFETY METRICS CALCULATION
# ==========================================

print("\nCalculating physics-based surrogate safety metrics...")

closing_speed = np.maximum(0, df["relative_speed"])

# Time-To-Collision (TTC) in seconds
ttc = np.where(
    closing_speed > 0.1,
    df["space_headway"] / (closing_speed + 1e-5),
    10.0
)
df["TTC_seconds"] = np.clip(ttc, 0.0, 10.0)

# Time Headway (TH) in seconds
th = df["time_headway"].clip(lower=0.0, upper=10.0)
df["TH_seconds"] = th

# Deceleration Rate to Avoid Crash (DRAC) in m/s^2
drac = (closing_speed ** 2) / (2.0 * np.maximum(0.1, df["space_headway"]))
df["DRAC_mps2"] = np.clip(drac, 0.0, 10.0)


# ==========================================
# RISK LABEL DEFINITION
# ==========================================

print("\nCategorizing risk states into 3 levels (0: Safe, 1: Moderate, 2: High Risk)...")

conditions = [
    (df["TTC_seconds"] <= 1.5) | (df["TH_seconds"] <= 1.0) | (df["DRAC_mps2"] >= 3.4) | (df["acceleration_mps2"] <= -3.0),
    ((df["TTC_seconds"] > 1.5) & (df["TTC_seconds"] <= 3.5)) | ((df["TH_seconds"] > 1.0) & (df["TH_seconds"] <= 2.2)) | ((df["DRAC_mps2"] >= 1.5) & (df["DRAC_mps2"] < 3.4))
]

choices = [2, 1] # 2: High Risk, 1: Moderate Risk

df["risk_level"] = np.select(conditions, choices, default=0) # 0: Safe


# ==========================================
# FEATURE SELECTION FOR MODEL TRAINING
# ==========================================

feature_cols = [
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
    "vehicle_type",
    "TTC_seconds",
    "TH_seconds",
    "DRAC_mps2"
]

target_col = "risk_level"

dataset = df[
    feature_cols + [
        "Vehicle_ID",
        "Global_Time",
        "Local_X",
        "Local_Y",
        target_col
    ]
].copy()

dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

print("\n==================================================")
print("COLLISION-RISK DATASET SUMMARY")
print("==================================================")

print("Final Cleaned Samples:", len(dataset))
print("Unique Vehicles:", dataset["Vehicle_ID"].nunique())

print("\nRisk Level Distribution:")
risk_counts = dataset["risk_level"].value_counts().sort_index()
risk_names = {0: "Safe (0)", 1: "Moderate Risk (1)", 2: "High Risk (2)"}
for k, v in risk_counts.items():
    pct = (v / len(dataset)) * 100
    print(f"  {risk_names.get(k, k)}: {v:,} ({pct:.2f}%)")

# ==========================================
# SAVE DATASET
# ==========================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT_FILE, index=False)

print("\nSaved output to:")
print(OUTPUT_FILE)

print("==================================================")
print("STEP 6 COMPLETED SUCCESSFULLY [OK]")
print("==================================================")
