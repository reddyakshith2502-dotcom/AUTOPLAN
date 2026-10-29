import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import math
import heapq
import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

TRAVEL_MODEL_FILE = BASE_DIR / "models" / "travel_time_model.pkl"
RISK_MODEL_FILE = BASE_DIR / "models" / "collision_risk_model.pkl"

# ==========================================
# REAL-WORLD URBAN CITY GRAPH GENERATOR
# ==========================================
# Recreates the exact urban driving map from user reference image:
# - Roundabout (Église Saint Jean-Baptiste junction)
# - Traffic Lights (Rue Clément Ader & Avenue Sadi Lecointe)
# - Weak GPS Signal Corridor (Blue route along Square des Cèdres)
# - Main Avenues (Avenue Louis Breguet, Rue Brindejonc, A86/D57)
# ==========================================

class UrbanCityGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.adj = {}
        self._build_urban_map()

    def _build_urban_map(self):
        # (id, x, y, name, node_type)
        node_defs = [
            ("N_START",      50,  300, "Rue Brindejonc Entry", "entry"),
            ("R_ROUNDABOUT", 150, 200, "Roundabout (Église St Jean-Baptiste)", "roundabout"),
            ("TL_ADER",      260, 240, "Traffic Light (Rue Clément Ader)", "traffic_light"),
            ("TL_SADI",      380, 260, "Traffic Light (Avenue Sadi Lecointe)", "traffic_light"),
            ("N_LECOINTE",   480, 230, "Médiathèque de Vélizy", "junction"),
            ("GPS_START",    300, 360, "Weak GPS Corridor Entry", "weak_gps_entry"),
            ("GPS_MID",      480, 380, "Square des Cèdres (Weak GPS)", "weak_gps_mid"),
            ("GPS_END",      650, 310, "Weak GPS Exit", "weak_gps_exit"),
            ("N_BREGUET_1",  220, 100, "Avenue Louis Breguet South", "junction"),
            ("N_BREGUET_2",  420, 120, "Stade Sadi Lecointe", "junction"),
            ("N_BREGUET_3",  600, 180, "Audi/Porsche Mobility Zone", "junction"),
            ("A86_JUNCTION",  500, 50,  "A86 / E5 Highway Interchange", "highway"),
            ("N_DEST",       720, 280, "Vélizy-Villacoublay Connector", "exit")
        ]

        for nid, x, y, name, ntype in node_defs:
            self.nodes[nid] = {"x": x, "y": y, "name": name, "type": ntype}
            self.adj[nid] = []

        # (u, v, distance, max_speed_mps, lanes, density, has_signal, is_roundabout, is_weak_gps)
        edge_defs = [
            ("N_START",      "R_ROUNDABOUT", 120, 13.8, 2, 0.25, False, False, False),
            ("R_ROUNDABOUT", "TL_ADER",      130, 11.1, 2, 0.35, True,  True,  False),
            ("R_ROUNDABOUT", "N_BREGUET_1",  140, 13.8, 2, 0.20, False, True,  False),
            ("TL_ADER",      "TL_SADI",      140, 11.1, 2, 0.40, True,  False, False),
            ("TL_SADI",      "N_LECOINTE",   110, 11.1, 2, 0.30, False, False, False),
            ("N_LECOINTE",   "N_DEST",       150, 13.8, 2, 0.25, False, False, False),
            
            # Blue Weak GPS Route Corridor
            ("N_START",      "GPS_START",    180, 16.6, 2, 0.15, False, False, False),
            ("GPS_START",    "GPS_MID",      200, 16.6, 2, 0.10, False, False, True),  # Weak GPS!
            ("GPS_MID",      "GPS_END",      190, 16.6, 2, 0.12, False, False, True),  # Weak GPS!
            ("GPS_END",      "N_DEST",       100, 13.8, 2, 0.20, False, False, False),
            
            # Breguet & A86 Highway Alternative Route
            ("N_BREGUET_1",  "N_BREGUET_2",  210, 13.8, 2, 0.30, False, False, False),
            ("N_BREGUET_2",  "N_BREGUET_3",  190, 13.8, 2, 0.20, False, False, False),
            ("N_BREGUET_3",  "N_DEST",       140, 13.8, 2, 0.25, False, False, False),
            ("N_BREGUET_2",  "A86_JUNCTION", 120, 22.2, 3, 0.15, False, False, False),
            ("A86_JUNCTION", "N_BREGUET_3",  150, 22.2, 3, 0.15, False, False, False),
            
            # Inter-connectors
            ("TL_ADER",      "GPS_START",    80,  11.1, 1, 0.20, False, False, False),
            ("N_BREGUET_2",  "TL_SADI",      100, 11.1, 1, 0.25, False, False, False),
        ]

        for u, v, dist, spd, lanes, dens, signal, rabout, weak_gps in edge_defs:
            edge_data = {
                "u": u,
                "v": v,
                "distance": dist,
                "max_speed": spd,
                "lanes": lanes,
                "density": dens,
                "has_signal": signal,
                "signal_state": "GREEN" if not signal else "RED",  # Dynamic traffic light state
                "is_roundabout": rabout,
                "is_weak_gps": weak_gps,
                "gps_noise_std": 15.0 if weak_gps else 0.5,       # GPS noise in meters
                "is_blocked": False,
                "accident_flag": False
            }
            self.edges[(u, v)] = edge_data
            self.adj[u].append(v)

    def get_heuristic(self, u, v):
        x1, y1 = self.nodes[u]["x"], self.nodes[u]["y"]
        x2, y2 = self.nodes[v]["x"], self.nodes[v]["y"]
        return math.hypot(x2 - x1, y2 - y1)

# ==========================================
# EXTENDED KALMAN FILTER (EKF) FOR WEAK GPS
# ==========================================

class EKFLocalization:
    def __init__(self):
        # State vector x = [x, y, v, yaw]
        self.state = np.zeros(4)
        self.P = np.diag([1.0, 1.0, 0.5, 0.1]) # Covariance matrix

    def predict(self, dt=0.5, accel=0.0, yaw_rate=0.0):
        # Kinematic Motion Model
        x, y, v, yaw = self.state
        v += accel * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        self.state = np.array([x, y, v, yaw])

        # Jacobian F
        F = np.array([
            [1, 0, math.cos(yaw)*dt, -v*math.sin(yaw)*dt],
            [0, 1, math.sin(yaw)*dt,  v*math.cos(yaw)*dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        Q = np.diag([0.1, 0.1, 0.05, 0.01]) # Process noise
        self.P = F @ self.P @ F.T + Q

    def update_gps(self, z_gps, gps_noise_std=0.5):
        # Measurement z = [x_gps, y_gps]
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        R = np.diag([gps_noise_std**2, gps_noise_std**2])

        y_res = z_gps - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y_res
        self.P = (np.eye(4) - K @ H) @ self.P
        
        # Returns localization uncertainty radius (m)
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))

# ==========================================
# URBAN ADAPTIVE RISK-AWARE A* PLANNER
# ==========================================

class UrbanAdaptivePlanner:
    def __init__(self, graph: UrbanCityGraph):
        self.graph = graph
        self.travel_model = None
        self.risk_model = None
        self._load_ml_models()

    def _load_ml_models(self):
        if TRAVEL_MODEL_FILE.exists():
            self.travel_model = joblib.load(TRAVEL_MODEL_FILE)
        if RISK_MODEL_FILE.exists():
            self.risk_model = joblib.load(RISK_MODEL_FILE)

    def predict_edge_cost(self, u, v, alpha=1.2, beta=20.0, gamma_signal=15.0, delta_gps=10.0):
        edge = self.graph.edges.get((u, v))
        if not edge or edge["is_blocked"]:
            return float("inf"), float("inf"), float("inf"), float("inf")

        distance = edge["distance"]
        max_speed = edge["max_speed"]
        density = edge["density"]
        
        current_speed = max(2.0, max_speed * (1.0 - 0.6 * density))
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

        # 1. ML Travel Time Prediction
        if self.travel_model:
            pred_time = float(self.travel_model.predict(travel_df)[0])
            pred_time = max(distance / max_speed, pred_time)
        else:
            pred_time = distance / current_speed

        # 2. ML Collision Risk Prediction
        if self.risk_model:
            risk_probs = self.risk_model.predict_proba(risk_df)[0]
            classes = self.risk_model.classes_
            pred_risk = sum(c * p for c, p in zip(classes, risk_probs))
        else:
            pred_risk = density * 2.0

        # 3. Traffic Signal Delay Penalty
        signal_delay = 0.0
        if edge["has_signal"] and edge["signal_state"] == "RED":
            signal_delay = gamma_signal # RED light adds 15s wait penalty

        # 4. Roundabout Yield Delay
        if edge["is_roundabout"]:
            pred_time += 4.0 # Yielding & circular entry delay

        # 5. Weak GPS Uncertainty Penalty
        gps_penalty = 0.0
        if edge["is_weak_gps"]:
            gps_penalty = edge["gps_noise_std"] * delta_gps # Uncertainty penalty

        if edge["accident_flag"]:
            pred_risk += 5.0
            pred_time *= 3.0

        total_cost = (
            distance +
            (alpha * pred_time * 5.0) +
            (beta * pred_risk * 15.0) +
            (signal_delay * 8.0) +
            (gps_penalty * 2.0)
        )

        return total_cost, pred_time + signal_delay, pred_risk, gps_penalty

    def run_urban_ml_astar(self, start, goal, alpha=1.2, beta=20.0):
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
                cost, _, _, _ = self.predict_edge_cost(current, nxt, alpha=alpha, beta=beta)
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

    def evaluate_urban_path(self, path):
        if not path or len(path) < 2:
            return {"total_distance": 0.0, "total_time": 0.0, "avg_risk": 0.0, "gps_uncertainty": 0.0}

        total_dist, total_time, risk_scores, gps_penalties = 0.0, 0.0, [], []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            cost, time, risk, gps_p = self.predict_edge_cost(u, v)
            edge = self.graph.edges.get((u, v), {})
            total_dist += edge.get("distance", 0.0)
            total_time += time
            risk_scores.append(risk)
            gps_penalties.append(gps_p)

        return {
            "total_distance": round(total_dist, 2),
            "total_time": round(total_time, 2),
            "avg_risk": round(float(np.mean(risk_scores)), 4) if risk_scores else 0.0,
            "max_gps_uncertainty": round(float(np.max(gps_penalties)), 2) if gps_penalties else 0.0
        }

if __name__ == "__main__":
    print("==================================================")
    print("TESTING URBAN CITY NAVIGATION & SENSOR FUSION ENGINE")
    print("==================================================")

    graph = UrbanCityGraph()
    planner = UrbanAdaptivePlanner(graph)
    ekf = EKFLocalization()

    start, goal = "N_START", "N_DEST"
    path = planner.run_urban_ml_astar(start, goal)
    metrics = planner.evaluate_urban_path(path)

    print(f"\nCalculated Optimal Route: {' -> '.join(path)}")
    print(f"Path Metrics: {metrics}")

    # Test EKF Sensor Fusion in Weak GPS Zone
    print("\nSimulating EKF Localization in Weak GPS Zone...")
    ekf.state = np.array([300.0, 360.0, 12.0, 0.1])
    ekf.predict(dt=0.5, accel=0.1)
    uncert_normal = ekf.update_gps(np.array([306.1, 360.6]), gps_noise_std=0.5)
    print(f"Normal GPS Localization Uncertainty: {uncert_normal:.3f} meters")

    ekf.predict(dt=0.5, accel=0.0)
    uncert_weak = ekf.update_gps(np.array([320.5, 362.8]), gps_noise_std=15.0)
    print(f"Weak GPS Localization Uncertainty  : {uncert_weak:.3f} meters")

    print("==================================================")
    print("URBAN ENGINE TEST COMPLETED SUCCESSFULLY [OK]")
    print("==================================================")
