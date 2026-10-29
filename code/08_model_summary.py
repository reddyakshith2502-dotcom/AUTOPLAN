from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

travel_file = BASE_DIR / "processed" / "travel_time_dataset.csv"
risk_file = BASE_DIR / "processed" / "risk_dataset.csv"

print("================================")
print("MODEL DATASET SUMMARY")
print("================================")

travel = pd.read_csv(travel_file)
risk = pd.read_csv(risk_file)

print("\nTRAVEL-TIME MODEL")
print("--------------------------------")
print("Samples:", len(travel))
print("Vehicles:", travel["Vehicle_ID"].nunique())
print("Mean target:", round(travel["future_travel_time"].mean(), 4))
print("Target std:", round(travel["future_travel_time"].std(), 4))

print("\nCOLLISION-RISK MODEL")
print("--------------------------------")
print("Samples:", len(risk))
print("Vehicles:", risk["Vehicle_ID"].nunique())
print("\nRisk distribution:")
print(risk["risk_label"].value_counts())

print("\n================================")
print("SUMMARY COMPLETED [OK]")
print("================================")
