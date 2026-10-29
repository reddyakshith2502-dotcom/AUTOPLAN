import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation.multi_vehicle_sim import MultiVehicleSimulator

DOCS_DIR = BASE_DIR / "docs"
PROCESSED_DIR = BASE_DIR / "processed"

print("==================================================")
print("DYNAMIC RISK-AWARE A* BENCHMARK EXPERIMENTS")
print("==================================================")

scenarios = [
    {"name": "Scenario A (10 Vehicles, Normal Traffic)", "num_vehicles": 10},
    {"name": "Scenario B (20 Vehicles, Moderate Congestion)", "num_vehicles": 20},
    {"name": "Scenario C (30 Vehicles, Road Blockages)", "num_vehicles": 30},
    {"name": "Scenario D (50 Vehicles, Multi-Blockage & Risk Spikes)", "num_vehicles": 50},
]

algorithms = ["dijkstra", "astar", "proposed"]
alg_labels = {
    "dijkstra": "Dijkstra (Baseline 1)",
    "astar": "Standard A* (Baseline 2)",
    "proposed": "Proposed ML-Driven Risk-Aware A*"
}

results = []

for scen in scenarios:
    print(f"\n--- Running {scen['name']} ---")
    for alg in algorithms:
        sim = MultiVehicleSimulator(num_vehicles=scen['num_vehicles'], algorithm=alg, seed=42)
        res = sim.run_full_simulation(max_steps=15)
        res['scenario'] = scen['name']
        res['algorithm_label'] = alg_labels[alg]
        results.append(res)
        print(f"  [{alg_labels[alg]}] -> Avg Time: {res['avg_travel_time_sec']}s | Risk Exposure: {res['avg_risk_exposure']} | Completion Rate: {res['completion_rate_pct']}% | Replans: {res['total_replannings']}")

df_res = pd.DataFrame(results)

# Save results CSV
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df_res.to_csv(PROCESSED_DIR / "simulation_results.csv", index=False)

print("\n==================================================")
print("BENCHMARK EXPERIMENT RESULTS SUMMARY")
print("==================================================")
print(df_res[["scenario", "algorithm_label", "avg_travel_time_sec", "avg_risk_exposure", "completion_rate_pct", "total_replannings"]].to_string(index=False))

# ==========================================
# STEP 19 — GENERATE RESEARCH GRAPHS
# ==========================================

print("\nGenerating empirical benchmark figures...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chart 1: Average Travel Time Comparison
pivot_time = df_res.pivot(index="scenario", columns="algorithm_label", values="avg_travel_time_sec")
pivot_time.plot(kind="bar", ax=axes[0], color=["#e74c3c", "#f39c12", "#2ecc71"], width=0.8)
axes[0].set_title("Average Travel Time (Seconds)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Seconds")
axes[0].set_xticklabels(["Scen A", "Scen B", "Scen C", "Scen D"], rotation=0)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# Chart 2: Collision Risk Exposure
pivot_risk = df_res.pivot(index="scenario", columns="algorithm_label", values="avg_risk_exposure")
pivot_risk.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#f39c12", "#2ecc71"], width=0.8)
axes[1].set_title("Collision Risk Exposure Score", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Risk Exposure Units")
axes[1].set_xticklabels(["Scen A", "Scen B", "Scen C", "Scen D"], rotation=0)
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

# Chart 3: Total Dynamic Replannings
pivot_replan = df_res.pivot(index="scenario", columns="algorithm_label", values="total_replannings")
pivot_replan.plot(kind="bar", ax=axes[2], color=["#e74c3c", "#f39c12", "#2ecc71"], width=0.8)
axes[2].set_title("Total Replanning Events Triggered", fontsize=12, fontweight="bold")
axes[2].set_ylabel("Count")
axes[2].set_xticklabels(["Scen A", "Scen B", "Scen C", "Scen D"], rotation=0)
axes[2].grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()

figure_path = DOCS_DIR / "benchmark_experiments.png"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(figure_path, dpi=300)
print(f"Saved benchmark figure to: {figure_path}")

print("==================================================")
print("EXPERIMENTS COMPLETED SUCCESSFULLY [OK]")
print("==================================================")
