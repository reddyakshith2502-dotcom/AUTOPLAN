from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "processed" / "ngsim_features.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "risk_dataset.csv"
ALT_OUTPUT_FILE = BASE_DIR / "processed" / "collision_risk_dataset.csv"

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

print("\nCategorizing risk states into binary/multi-level risk labels...")

# Create risk_label (0: SAFE, 1: HIGH RISK) for Step 7 model compatibility
conditions_binary = (df["TTC_seconds"] <= 1.5) | (df["TH_seconds"] <= 1.0) | (df["DRAC_mps2"] >= 3.4) | (df["acceleration_mps2"] <= -3.0)
df["risk_label"] = np.where(conditions_binary, 1, 0)

# Create risk_level (0: Safe, 1: Moderate, 2: High)
conditions_multi = [
    (df["TTC_seconds"] <= 1.5) | (df["TH_seconds"] <= 1.0) | (df["DRAC_mps2"] >= 3.4) | (df["acceleration_mps2"] <= -3.0),
    ((df["TTC_seconds"] > 1.5) & (df["TTC_seconds"] <= 3.5)) | ((df["TH_seconds"] > 1.0) & (df["TH_seconds"] <= 2.2)) | ((df["DRAC_mps2"] >= 1.5) & (df["DRAC_mps2"] < 3.4))
]
df["risk_level"] = np.select(conditions_multi, [2, 1], default=0)

# Adapt features for Step 7
df["ego_speed_mps"] = df["speed_mps"]
df["preceding_speed_mps"] = np.maximum(0, df["speed_mps"] - df["relative_speed"])
df["closing_speed_mps"] = closing_speed
df["v_Acc"] = df["acceleration_mps2"] / 0.3048
df["gap_m"] = df["space_headway"]
df["Time_Hdwy"] = df["TH_seconds"]
df["Lane_ID"] = df["lane"]
df["v_Class"] = df["vehicle_type"]


# ==========================================
# FEATURE SELECTION
# ==========================================

feature_cols = [
    "ego_speed_mps",
    "preceding_speed_mps",
    "closing_speed_mps",
    "v_Acc",
    "gap_m",
    "Time_Hdwy",
    "Lane_ID",
    "v_Class",
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

dataset = df[
    feature_cols + [
        "Vehicle_ID",
        "Global_Time",
        "Local_X",
        "Local_Y",
        "risk_label",
        "risk_level"
    ]
].copy()

dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

print("\n==================================================")
print("COLLISION-RISK DATASET SUMMARY")
print("==================================================")

print("Final Cleaned Samples:", len(dataset))
print("Unique Vehicles:", dataset["Vehicle_ID"].nunique())

print("\nBinary Risk Label Distribution (0: Safe, 1: High Risk):")
binary_counts = dataset["risk_label"].value_counts().sort_index()
for k, v in binary_counts.items():
    pct = (v / len(dataset)) * 100
    label_str = "SAFE (0)" if k == 0 else "HIGH RISK (1)"
    print(f"  {label_str}: {v:,} ({pct:.2f}%)")

print("\nTTC Statistics (seconds):")
print(dataset["TTC_seconds"].describe())

print("\nDRAC Statistics (m/s^2):")
print(dataset["DRAC_mps2"].describe())

# ==========================================
# SAVE DATASET
# ==========================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT_FILE, index=False)
dataset.to_csv(ALT_OUTPUT_FILE, index=False)

print("\nSaved output to:")
print("  -", OUTPUT_FILE)
print("  -", ALT_OUTPUT_FILE)

print("==================================================")
print("STEP 6 COMPLETED SUCCESSFULLY [OK]")
print("==================================================")
