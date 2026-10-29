import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import math
import heapq
import numpy as np
import pandas as pd
import joblib

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAVEL_MODEL_FILE = BASE_DIR / "models" / "travel_time_model.pkl"
RISK_MODEL_FILE = BASE_DIR / "models" / "collision_risk_model.pkl"

# ==========================================
# COMPLEX ROAD NETWORK ("CRAZY MAP") GENERATOR
# ==========================================

class RoadNetworkGraph:
    def __init__(self):
        self.nodes = {}  # node_id -> (x, y, name, node_type)
        self.edges = {}  # (u, v) -> edge_data dict
        self.adj = {}    # u -> list of neighbor v
        self._build_crazy_map()

    def _build_crazy_map(self):
        node_definitions = [
            ("N01", 0, 0, "South Entry Mainline A", "entry"),
            ("N02", 0, 30, "South Entry Express B", "entry"),
            ("N03", 200, 0, "Highway 101 South Merge", "junction"),
            ("N04", 200, 30, "Express Lane Corridor 1", "junction"),
            ("N05", 200, -40, "Service Bypass South", "bypass"),
            ("N06", 450, 0, "Mid-Corridor Junction", "junction"),
            ("N07", 450, 40, "Express Overpass West", "junction"),
            ("N08", 450, -50, "Commercial Service Zone", "junction"),
            ("N09", 450, 100, "Northbound Interchange Entry", "interchange"),
            ("N10", 700, 10, "Central Toll Plaza", "toll"),
            ("N11", 700, 60, "High-Speed Bypass Loop", "bypass"),
            ("N12", 700, -60, "Industrial Freight Ramp", "ramp"),
            ("N13", 750, 120, "Interchange Flyover Bridge", "interchange"),
            ("N14", 950, 20, "Highway 101 North Bottleneck", "bottleneck"),
            ("N15", 950, 70, "Express Corridor Exit", "junction"),
            ("N16", 950, -40, "Service Bypass North", "bypass"),
            ("N17", 1200, 20, "North Main Terminal", "exit"),
            ("N18", 1200, 80, "Airport Connector Exit", "exit"),
            ("N19", 1200, -30, "Downtown Logistics Hub", "exit"),
            ("N20", 1250, 130, "Tech Corridor Gateway", "exit"),
        ]

        for nid, x, y, name, ntype in node_definitions:
            self.nodes[nid] = {"x": x, "y": y, "name": name, "type": ntype}
            self.adj[nid] = []

        edge_connections = [
            ("N01", "N03", 200, 25.0, 4, 0.2),
            ("N02", "N04", 202, 30.0, 2, 0.1),
            ("N01", "N05", 204, 18.0, 2, 0.05),
            ("N03", "N06", 250, 22.0, 4, 0.35),
            ("N04", "N07", 251, 30.0, 2, 0.15),
            ("N05", "N08", 250, 18.0, 2, 0.1),
            ("N04", "N09", 260, 25.0, 1, 0.2),
            ("N03", "N05", 50,  15.0, 1, 0.05),
            ("N05", "N03", 50,  15.0, 1, 0.05),
            ("N06", "N10", 250, 20.0, 3, 0.5),
            ("N07", "N11", 250, 32.0, 2, 0.12),
            ("N08", "N12", 250, 18.0, 2, 0.1),
            ("N09", "N13", 300, 28.0, 2, 0.18),
            ("N06", "N07", 40,  20.0, 1, 0.2),
            ("N07", "N06", 40,  20.0, 1, 0.2),
            ("N06", "N08", 50,  15.0, 1, 0.1),
            ("N10", "N14", 250, 18.0, 2, 0.65),
            ("N11", "N15", 250, 30.0, 2, 0.15),
            ("N12", "N16", 250, 20.0, 2, 0.12),
            ("N13", "N15", 210, 28.0, 1, 0.25),
            ("N10", "N11", 50,  20.0, 1, 0.2),
            ("N12", "N10", 80,  15.0, 1, 0.3),
            ("N14", "N17", 250, 25.0, 3, 0.3),
            ("N15", "N18", 250, 30.0, 2, 0.15),
            ("N16", "N19", 250, 20.0, 2, 0.1),
            ("N15", "N20", 300, 32.0, 2, 0.1),
            ("N14", "N15", 50,  20.0, 1, 0.2),
            ("N16", "N14", 70,  18.0, 1, 0.2),
        ]

        for u, v, dist, max_spd, lanes, density in edge_connections:
            edge_data = {
                "u": u,
                "v": v,
                "distance": dist,
                "max_speed": max_spd,
                "lanes": lanes,
                "density": density,
                "is_blocked": False,
                "accident_flag": False,
                "hazard_level": 0.0
            }
            self.edges[(u, v)] = edge_data
            self.adj[u].append(v)

    def get_heuristic(self, u, v):
        x1, y1 = self.nodes[u]["x"], self.nodes[u]["y"]
        x2, y2 = self.nodes[v]["x"], self.nodes[v]["y"]
        return math.hypot(x2 - x1, y2 - y1)

# ==========================================
# ML DYNAMIC ROUTING ENGINE
# ==========================================

class DynamicAStarPlanner:
    def __init__(self, graph: RoadNetworkGraph):
        self.graph = graph
        self.travel_model = None
        self.risk_model = None
        self._load_ml_models()

    def _load_ml_models(self):
        if TRAVEL_MODEL_FILE.exists():
            self.travel_model = joblib.load(TRAVEL_MODEL_FILE)
            print("Loaded Travel-Time ML Model [OK]")
        else:
            print("Warning: Travel-Time ML Model not found.")

        if RISK_MODEL_FILE.exists():
            self.risk_model = joblib.load(RISK_MODEL_FILE)
            print("Loaded Collision-Risk ML Model [OK]")
        else:
            print("Warning: Collision-Risk ML Model not found.")

    def predict_edge_cost(self, u, v, alpha=1.0, beta=15.0):
        edge = self.graph.edges.get((u, v))
        if not edge or edge["is_blocked"]:
            return float("inf"), float("inf"), float("inf")

        distance = edge["distance"]
        max_speed = edge["max_speed"]
        density = edge["density"] + edge["hazard_level"]

        current_speed = max(2.0, max_speed * (1.0 - 0.7 * density))
        space_headway = max(5.0, (1.0 / (density + 1e-4)))
        time_headway = space_headway / current_speed

        travel_df = pd.DataFrame([{
            "speed_mps": current_speed,
            "acceleration_mps2": 0.0,
            "speed_change": 0.0,
            "previous_speed": current_speed,
            "previous_acceleration": 0.0,
            "space_headway": space_headway,
            "time_headway": time_headway,
            "traffic_density": density,
            "relative_speed": 0.0,
            "lane": 1,
            "vehicle_type": 1
        }])

        risk_df = pd.DataFrame([{
            "speed_mps": current_speed,
            "acceleration_mps2": 0.0,
            "speed_change": 0.0,
            "previous_speed": current_speed,
            "previous_acceleration": 0.0,
            "space_headway": space_headway,
            "time_headway": time_headway,
            "traffic_density": density,
            "relative_speed": 0.0,
            "lane": 1,
            "vehicle_type": 1,
            "TTC_seconds": min(10.0, space_headway / 0.1),
            "TH_seconds": min(10.0, time_headway),
            "DRAC_mps2": 0.0
        }])

        if self.travel_model:
            pred_time = float(self.travel_model.predict(travel_df)[0])
            pred_time = max(distance / max_speed, pred_time)
        else:
            pred_time = distance / current_speed

        if self.risk_model:
            risk_probs = self.risk_model.predict_proba(risk_df)[0]
            classes = self.risk_model.classes_
            pred_risk_score = sum(c * p for c, p in zip(classes, risk_probs))
        else:
            pred_risk_score = density * 2.0

        if edge["accident_flag"]:
            pred_risk_score += 5.0
            pred_time *= 3.0

        total_cost = distance + (alpha * pred_time * 5.0) + (beta * pred_risk_score * 20.0)

        return total_cost, pred_time, pred_risk_score

    def run_dijkstra(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {node: float("inf") for node in self.graph.nodes}
        g_score[start] = 0.0

        while open_set:
            current_dist, current = heapq.heappop(open_set)
            if current == goal:
                break

            for nxt in self.graph.adj[current]:
                edge = self.graph.edges[(current, nxt)]
                if edge["is_blocked"]:
                    continue
                tentative = g_score[current] + edge["distance"]
                if tentative < g_score[nxt]:
                    g_score[nxt] = tentative
                    came_from[nxt] = current
                    heapq.heappush(open_set, (tentative, nxt))

        return self._reconstruct_path(came_from, start, goal)

    def run_standard_astar(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {node: float("inf") for node in self.graph.nodes}
        g_score[start] = 0.0

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break

            for nxt in self.graph.adj[current]:
                cost, time, _ = self.predict_edge_cost(current, nxt, alpha=1.0, beta=0.0)
                if math.isinf(cost):
                    continue
                tentative = g_score[current] + time
                if tentative < g_score[nxt]:
                    g_score[nxt] = tentative
                    came_from[nxt] = current
                    h = self.graph.get_heuristic(nxt, goal) / 30.0
                    heapq.heappush(open_set, (tentative + h, nxt))

        return self._reconstruct_path(came_from, start, goal)

    def run_dynamic_ml_astar(self, start, goal, alpha=1.2, beta=20.0):
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {node: float("inf") for node in self.graph.nodes}
        g_score[start] = 0.0

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break

            for nxt in self.graph.adj[current]:
                cost, _, _ = self.predict_edge_cost(current, nxt, alpha=alpha, beta=beta)
                if math.isinf(cost):
                    continue
                tentative = g_score[current] + cost
                if tentative < g_score[nxt]:
                    g_score[nxt] = tentative
                    came_from[nxt] = current
                    h = self.graph.get_heuristic(nxt, goal)
                    heapq.heappush(open_set, (tentative + h, nxt))

        return self._reconstruct_path(came_from, start, goal)

    def _reconstruct_path(self, came_from, start, goal):
        if goal not in came_from and start != goal:
            return []
        path = [goal]
        curr = goal
        while curr in came_from:
            curr = came_from[curr]
            path.append(curr)
        path.reverse()
        return path if path[0] == start else []

    def evaluate_path_metrics(self, path):
        if not path or len(path) < 2:
            return {"total_distance": 0.0, "total_time": 0.0, "avg_risk_score": 0.0, "total_cost": 0.0}

        total_dist = 0.0
        total_time = 0.0
        risk_scores = []
        total_cost = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            cost, time, risk = self.predict_edge_cost(u, v)
            edge = self.graph.edges.get((u, v), {})
            total_dist += edge.get("distance", 0.0)
            total_time += time
            risk_scores.append(risk)
            total_cost += cost

        avg_risk = float(np.mean(risk_scores)) if risk_scores else 0.0
        return {
            "total_distance": round(total_dist, 2),
            "total_time": round(total_time, 2),
            "avg_risk_score": round(avg_risk, 4),
            "total_cost": round(total_cost, 2)
        }

if __name__ == "__main__":
    print("==================================================")
    print("TESTING DYNAMIC RISK-AWARE A* NAVIGATION ENGINE")
    print("==================================================")

    graph = RoadNetworkGraph()
    planner = DynamicAStarPlanner(graph)

    start_node = "N01"
    goal_node = "N17"

    print(f"\nRouting from {start_node} -> {goal_node}")

    graph.edges[("N10", "N14")]["accident_flag"] = True
    graph.edges[("N10", "N14")]["hazard_level"] = 0.85

    path_dijkstra = planner.run_dijkstra(start_node, goal_node)
    path_std_astar = planner.run_standard_astar(start_node, goal_node)
    path_ml_astar = planner.run_dynamic_ml_astar(start_node, goal_node)

    print("\n1. Static Dijkstra Path:")
    print("   ", " -> ".join(path_dijkstra))
    print("   Metrics:", planner.evaluate_path_metrics(path_dijkstra))

    print("\n2. Standard A* Path:")
    print("   ", " -> ".join(path_std_astar))
    print("   Metrics:", planner.evaluate_path_metrics(path_std_astar))

    print("\n3. Proposed Dynamic Risk-Aware ML-A* Path:")
    print("   ", " -> ".join(path_ml_astar))
    print("   Metrics:", planner.evaluate_path_metrics(path_ml_astar))

    print("\n==================================================")
    print("STEP 8 ENGINE VERIFICATION COMPLETED [OK]")
    print("==================================================")
