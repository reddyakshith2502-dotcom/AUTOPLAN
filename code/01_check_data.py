from pathlib import Path
import pandas as pd

# Automatically find the main AdaptiveNavigation folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
FILE = BASE_DIR / "dataset" / "ngsim_us101.csv"

print("================================")
print("NGSIM DATASET CHECK")
print("================================")

print("\nDataset path:")
print(FILE)

# Check whether file exists
if not FILE.exists():
    print("\nERROR: Dataset not found!")
    exit()

print("\nDataset found successfully! ✅")
print("Loading dataset...")

df = pd.read_csv(FILE)

print("\n================================")
print("DATASET INFORMATION")
print("================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\n================================")
print("DATASET CHECK COMPLETED ✅")
print("================================")