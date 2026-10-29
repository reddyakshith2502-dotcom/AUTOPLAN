import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import random
import importlib.util
import sys
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Dynamically import 08_dynamic_astar
spec = importlib.util.spec_from_file_location("dynamic_astar", BASE_DIR / "code" / "08_dynamic_astar.py")
dynamic_astar = importlib.util.module_from_spec(spec)
sys.modules["dynamic_astar"] = dynamic_astar
spec.loader.exec_module(dynamic_astar)

RoadNetworkGraph = dynamic_astar.RoadNetworkGraph
DynamicAStarPlanner = dynamic_astar.DynamicAStarPlanner

OUTPUT_FILE = BASE_DIR / "processed" / "simulation_results.csv"

print("==================================================")
print("MULTI-VEHICLE DYNAMIC TRAFFIC SIMULATION & BENCHMARK")
print("==================================================")

class VehicleAgent:
    def __init__(self, vehicle_id, start_node, goal_node, strategy="ml_astar"):
        self.id = vehicle_id
        self.start_node = start_node
        self.goal_node = goal_node
        self.strategy = strategy
        self.current_node = start_node
        self.path = []
        self.path_index = 0
        self.status = "IDLE"
        self.total_time_taken = 0.0
        self.risk_exposure = 0.0
        self.reroute_count = 0

    def plan_route(self, planner: DynamicAStarPlanner):
        if self.strategy == "dijkstra":
            self.path = planner.run_dijkstra(self.current_node, self.goal_node)
        elif self.strategy == "std_astar":
            self.path = planner.run_standard_astar(self.current_node, self.goal_node)
        else: # ml_astar
            self.path = planner.run_dynamic_ml_astar(self.current_node, self.goal_node)

        self.path_index = 0
        if self.path:
            self.status = "EN_ROUTE"
        else:
            self.status = "BLOCKED"

    def step(self, planner: DynamicAStarPlanner, current_step_time):
        if self.status in ["ARRIVED", "BLOCKED"]:
            return

        if self.path_index >= len(self.path) - 1:
            self.status = "ARRIVED"
            return

        u = self.path[self.path_index]
        v = self.path[self.path_index + 1]

        edge = planner.graph.edges.get((u, v))
        if edge and (edge["is_blocked"] or edge["accident_flag"]):
            if self.strategy != "dijkstra":
                old_path = list(self.path)
                self.plan_route(planner)
                if self.path != old_path:
                    self.reroute_count += 1
                    self.status = "REROUTED"
                if not self.path:
                    self.status = "BLOCKED"
                    return
            else:
                self.status = "BLOCKED"
                return

        cost, time, risk = planner.predict_edge_cost(u, v)
        self.total_time_taken += time
        self.risk_exposure += risk
        self.path_index += 1
        self.current_node = self.path[self.path_index]

        if self.current_node == self.goal_node:
            self.status = "ARRIVED"


def run_fleet_simulation(strategy_name, fleet_size=15):
    random.seed(42)
    np.random.seed(42)

    graph = RoadNetworkGraph()
    planner = DynamicAStarPlanner(graph)

    entry_nodes = ["N01", "N02"]
    exit_nodes = ["N17", "N18", "N19", "N20"]

    vehicles = []
    for i in range(fleet_size):
        start = random.choice(entry_nodes)
        goal = random.choice(exit_nodes)
        v = VehicleAgent(f"V_{i+1:02d}", start, goal, strategy=strategy_name)
        v.plan_route(planner)
        vehicles.append(v)

    simulation_duration = 30
    hazard_injected = False

    for t in range(simulation_duration):
        if t == 10 and not hazard_injected:
            graph.edges[("N10", "N14")]["accident_flag"] = True
            graph.edges[("N10", "N14")]["hazard_level"] = 0.95
            graph.edges[("N10", "N14")]["is_blocked"] = True
            hazard_injected = True

        for v in vehicles:
            v.step(planner, t)

    arrived = [v for v in vehicles if v.status == "ARRIVED"]
    blocked = [v for v in vehicles if v.status == "BLOCKED"]
    total_time = sum(v.total_time_taken for v in vehicles)
    avg_time = total_time / fleet_size
    avg_risk = sum(v.risk_exposure for v in vehicles) / fleet_size
    total_reroutes = sum(v.reroute_count for v in vehicles)
    completion_rate = (len(arrived) / fleet_size) * 100

    return {
        "Strategy": strategy_name,
        "Fleet_Size": fleet_size,
        "Arrived_Vehicles": len(arrived),
        "Blocked_Vehicles": len(blocked),
        "Completion_Rate_%": round(completion_rate, 2),
        "Avg_Travel_Time_s": round(avg_time, 2),
        "Avg_Risk_Exposure": round(avg_risk, 4),
        "Total_Reroute_Events": total_reroutes
    }


if __name__ == "__main__":
    strategies = ["dijkstra", "std_astar", "ml_astar"]
    results = []

    for strat in strategies:
        res = run_fleet_simulation(strat, fleet_size=15)
        results.append(res)

    results_df = pd.DataFrame(results)

    print("\n==================================================")
    print("SIMULATION BENCHMARK SUMMARY COMPARISON")
    print("==================================================")
    print(results_df.to_string(index=False))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSimulation benchmark metrics saved to: {OUTPUT_FILE}")
    print("==================================================")
    print("STEP 9 COMPLETED SUCCESSFULLY [OK]")
    print("==================================================")
