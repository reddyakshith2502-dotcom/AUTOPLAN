from pathlib import Path
import pandas as pd
import numpy as np

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "dataset" / "ngsim_us101.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "ngsim_clean.csv"

print("================================")
print("NGSIM DATA CLEANING")
print("================================")

print("\nLoading dataset...")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Remove duplicate records
before = len(df)

df = df.drop_duplicates()

print("\nDuplicates removed:", before - len(df))

# Remove physically invalid values
df = df[df["v_Vel"] >= 0]
df = df[df["v_Length"] > 0]
df = df[df["v_Width"] > 0]

# Reasonable acceleration range
df = df[
    (df["v_Acc"] >= -20) &
    (df["v_Acc"] <= 20)
]

# Replace infinite values
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Important columns
important_columns = [
    "Vehicle_ID",
    "Frame_ID",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "v_Vel",
    "v_Acc",
    "Lane_ID"
]

# Remove rows missing important information
df = df.dropna(
    subset=important_columns
)

# Sort trajectories correctly
df = df.sort_values(
    ["Vehicle_ID", "Global_Time"]
).reset_index(drop=True)

print("\nCleaned dataset:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nVehicles:", df["Vehicle_ID"].nunique())

print("\nRemaining missing values:")
print(df[important_columns].isnull().sum())

# Save
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n================================")
print("CLEANING COMPLETED ✅")
print("================================")

print("\nSaved file:")
print(OUTPUT_FILE)