from pathlib import Path
import pandas as pd
import numpy as np

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "processed" / "ngsim_clean.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "ngsim_features.csv"

print("================================")
print("NGSIM FEATURE ENGINEERING")
print("================================")

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Vehicles:", df["Vehicle_ID"].nunique())

# --------------------------------------------------
# 1. Sort vehicle trajectories
# --------------------------------------------------

df = df.sort_values(
    ["Vehicle_ID", "Global_Time"]
).reset_index(drop=True)


# --------------------------------------------------
# 2. Convert speed and acceleration
# NGSIM speed/acceleration are in ft/s and ft/s²
# --------------------------------------------------

df["speed_mps"] = df["v_Vel"] * 0.3048

df["acceleration_mps2"] = (
    df["v_Acc"] * 0.3048
)


# --------------------------------------------------
# 3. Vehicle dimensions
# --------------------------------------------------

df["vehicle_area"] = (
    df["v_Length"] *
    df["v_Width"]
)


# --------------------------------------------------
# 4. Speed change
# --------------------------------------------------

df["speed_change"] = (
    df.groupby("Vehicle_ID")["speed_mps"]
    .diff()
    .fillna(0)
)


# --------------------------------------------------
# 5. Previous speed
# --------------------------------------------------

df["previous_speed"] = (
    df.groupby("Vehicle_ID")["speed_mps"]
    .shift(1)
)


# --------------------------------------------------
# 6. Previous acceleration
# --------------------------------------------------

df["previous_acceleration"] = (
    df.groupby("Vehicle_ID")["acceleration_mps2"]
    .shift(1)
)


# --------------------------------------------------
# 7. Relative position to preceding vehicle
# --------------------------------------------------

df["space_headway"] = df["Space_Hdwy"]

df["time_headway"] = df["Time_Hdwy"]


# --------------------------------------------------
# 8. Traffic density indicator
# --------------------------------------------------

df["traffic_density"] = (
    1 / (df["space_headway"] + 1)
)

df["traffic_density"] = (
    df["traffic_density"]
    .clip(upper=1)
)


# --------------------------------------------------
# 9. Lane information
# --------------------------------------------------

df["lane"] = df["Lane_ID"]

df["vehicle_type"] = df["v_Class"]


# --------------------------------------------------
# 10. Relative motion indicator
# --------------------------------------------------

df["relative_speed"] = (
    df["speed_mps"] -
    df["previous_speed"]
)


# --------------------------------------------------
# 11. Time in seconds
# --------------------------------------------------

df["timestamp_seconds"] = (
    df["Global_Time"] / 1000
)


# --------------------------------------------------
# 12. Remove invalid/infinite values
# --------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# --------------------------------------------------
# 13. Fill trajectory-start values
# --------------------------------------------------

df["previous_speed"] = (
    df["previous_speed"]
    .fillna(df["speed_mps"])
)

df["previous_acceleration"] = (
    df["previous_acceleration"]
    .fillna(df["acceleration_mps2"])
)

df["relative_speed"] = (
    df["relative_speed"]
    .fillna(0)
)


# --------------------------------------------------
# 14. Display features
# --------------------------------------------------

feature_columns = [
    "Vehicle_ID",
    "Global_Time",
    "Local_X",
    "Local_Y",
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

print("\n================================")
print("CREATED FEATURES")
print("================================")

for feature in feature_columns:
    print("-", feature)


# --------------------------------------------------
# 15. Save feature dataset
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n================================")
print("FEATURE ENGINEERING COMPLETED ✅")
print("================================")

print("\nFinal rows:", len(df))
print("Final columns:", len(df.columns))
print("Vehicles:", df["Vehicle_ID"].nunique())

print("\nSaved file:")
print(OUTPUT_FILE)