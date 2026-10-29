from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "processed" / "ngsim_features.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "travel_time_dataset.csv"

print("================================")
print("FUTURE TRAVEL-TIME TARGET")
print("================================")

print("\nLoading feature dataset...")

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["Vehicle_ID", "Global_Time"]
).reset_index(drop=True)

print("Rows:", len(df))
print("Vehicles:", df["Vehicle_ID"].nunique())


# ==================================================
# TARGET DEFINITION
# ==================================================
# We predict the time required to travel approximately
# the next 100 feet of road.
#
# This creates a VARIABLE travel-time target.
# ==================================================

TARGET_DISTANCE = 100.0


# --------------------------------------------------
# Calculate future position
# --------------------------------------------------

grouped = df.groupby("Vehicle_ID", group_keys=False)

df["future_y"] = grouped["Local_Y"].shift(-1)

df["future_time"] = grouped["Global_Time"].shift(-1)


# --------------------------------------------------
# Instead of looking a fixed number of frames ahead,
# find a future point approximately TARGET_DISTANCE away.
#
# We use a forward-looking rolling search within each
# vehicle trajectory.
# --------------------------------------------------

results = []

for vehicle_id, vehicle in df.groupby("Vehicle_ID"):

    vehicle = vehicle.sort_values("Global_Time").reset_index(drop=True)

    y = vehicle["Local_Y"].to_numpy()
    t = vehicle["Global_Time"].to_numpy()

    n = len(vehicle)

    future_indices = np.full(n, -1, dtype=int)

    j = 1

    for i in range(n):

        if j <= i:
            j = i + 1

        while j < n and abs(y[j] - y[i]) < TARGET_DISTANCE:
            j += 1

        if j < n:
            future_indices[i] = j

    valid = future_indices >= 0

    current = vehicle.loc[valid].copy()

    future_idx = future_indices[valid]

    current["future_distance"] = (
        y[future_idx] - y[np.where(valid)[0]]
    )

    current["future_travel_time"] = (
        t[future_idx] -
        t[np.where(valid)[0]]
    ) / 1000.0

    results.append(current)


print("\nCombining vehicle trajectories...")

dataset = pd.concat(
    results,
    ignore_index=True
)


# --------------------------------------------------
# Select ML features
# --------------------------------------------------

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


dataset = dataset[
    features +
    [
        "Vehicle_ID",
        "Global_Time",
        "Local_X",
        "Local_Y",
        "future_distance",
        "future_travel_time"
    ]
].copy()


# --------------------------------------------------
# Clean target
# --------------------------------------------------

dataset = dataset.replace(
    [np.inf, -np.inf],
    np.nan
)

dataset = dataset.dropna()


dataset = dataset[
    dataset["future_distance"] >= TARGET_DISTANCE
]


dataset = dataset[
    dataset["future_travel_time"] > 0
]


# Remove unrealistic extreme values
dataset = dataset[
    dataset["future_travel_time"] <= 30
]


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n================================")
print("TARGET CREATION RESULTS")
print("================================")

print("Final samples:", len(dataset))

print(
    "Vehicles:",
    dataset["Vehicle_ID"].nunique()
)

print("\nFuture travel-time statistics:")

print(
    dataset["future_travel_time"].describe()
)


print("\nUnique travel-time values:")

print(
    dataset["future_travel_time"]
    .nunique()
)


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n================================")
print("TARGET CREATION COMPLETED ✅")
print("================================")

print("\nSaved:")
print(OUTPUT_FILE)