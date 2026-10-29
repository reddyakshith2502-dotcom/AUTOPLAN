import time
import pandas as pd
import numpy as np

from simulation.road_network import UrbanRoadNetwork
from simulation.dijkstra import dijkstra_route
from simulation.astar import standard_astar_route
from simulation.adaptive_astar import ml_dynamic_risk_aware_astar
from simulation.dynamic_events import DynamicTrafficEnvironment

class AutonomousVehicle:
    def __init__(self, vehicle_id, start_node, goal_node, algorithm='proposed'):
        self.vehicle_id = vehicle_id
        self.start_node = start_node
        self.goal_node = goal_node
        self.current_node = start_node
        self.algorithm = algorithm  # 'dijkstra', 'astar', 'proposed'
        
        self.path = []
        self.total_distance = 0.0
        self.total_time = 0.0
        self.risk_exposure = 0.0
        self.replan_count = 0
        self.completed = False

    def plan_route(self, road_network):
        start_time = time.perf_counter()
        
        if self.algorithm == 'dijkstra':
            path, cost = dijkstra_route(road_network, self.current_node, self.goal_node)
        elif self.algorithm == 'astar':
            path, cost = standard_astar_route(road_network, self.current_node, self.goal_node)
        else:  # 'proposed' (ML-Driven Dynamic Risk-Aware A*)
            path, cost = ml_dynamic_risk_aware_astar(road_network, self.current_node, self.goal_node)

        calc_time = (time.perf_counter() - start_time) * 1000.0  # ms

        if path:
            self.path = path
            return True, calc_time
        return False, calc_time

class MultiVehicleSimulator:
    def __init__(self, num_vehicles=20, algorithm='proposed', seed=42):
        self.network = UrbanRoadNetwork()
        self.env = DynamicTrafficEnvironment(self.network, seed=seed)
        self.algorithm = algorithm
        self.vehicles = []
        self.seed = seed
        self._initialize_fleet(num_vehicles)

    def _initialize_fleet(self, num_vehicles):
        rng = np.random.default_rng(self.seed)
        node_keys = list(self.network.nodes.keys())

        for i in range(1, num_vehicles + 1):
            start, goal = rng.choice(node_keys, size=2, replace=False)
            veh = AutonomousVehicle(i, start, goal, algorithm=self.algorithm)
            veh.plan_route(self.network)
            self.vehicles.append(veh)

    def run_simulation_step(self, step_index, inject_events=True):
        events_occurred = False
        # Inject dynamic events at specific simulation frames
        if inject_events:
            if step_index == 2:
                self.env.trigger_traffic_jam(('B', 'E'), delay_factor=5.0)
                self.env.trigger_traffic_jam(('E', 'H'), delay_factor=4.5)
                events_occurred = True
            elif step_index == 4:
                self.env.trigger_road_blockage(('E', 'F'))
                self.env.trigger_road_blockage(('D', 'E'))
                events_occurred = True
            elif step_index == 6:
                self.env.trigger_high_risk_zone(('B', 'C'), risk_level=0.95)
                self.env.trigger_high_risk_zone(('H', 'I'), risk_level=0.88)
                events_occurred = True

        total_step_calc_time = 0.0

        for veh in self.vehicles:
            if veh.completed:
                continue

            # Check if remaining route has blocked edge OR high risk (only proposed replans for risk/delay)
            replan_needed = False

            if len(veh.path) >= 2:
                for idx in range(len(veh.path) - 1):
                    u, v = veh.path[idx], veh.path[idx+1]
                    if self.network.graph.has_edge(u, v):
                        edge_info = self.network.graph[u][v]
                        # All algorithms must replan if path is physically blocked
                        if edge_info.get('is_blocked', False):
                            replan_needed = True
                            break
                        # Proposed algorithm proactively replans if risk or traffic delay spikes along path
                        if self.algorithm == 'proposed' and events_occurred:
                            if edge_info.get('collision_risk', 0.0) > 0.5 or edge_info.get('predicted_travel_time', 0.0) > 60.0:
                                replan_needed = True
                                break

            if replan_needed or len(veh.path) < 2:
                veh.replan_count += 1
                success, calc_t = veh.plan_route(self.network)
                total_step_calc_time += calc_t

            if len(veh.path) >= 2:
                next_node = veh.path[1]
                edge_data = self.network.graph[veh.current_node][next_node]
                
                veh.total_distance += edge_data['distance']
                veh.total_time += edge_data['predicted_travel_time'] + edge_data.get('signal_delay', 0.0)
                veh.risk_exposure += edge_data.get('collision_risk', 0.0)
                
                veh.current_node = next_node
                veh.path.pop(0)

                if veh.current_node == veh.goal_node:
                    veh.completed = True

        return total_step_calc_time

    def run_full_simulation(self, max_steps=15):
        for step in range(1, max_steps + 1):
            self.run_simulation_step(step)

        completed_vehicles = [v for v in self.vehicles if v.completed]
        completion_rate = (len(completed_vehicles) / len(self.vehicles)) * 100.0

        avg_time = np.mean([v.total_time for v in self.vehicles]) if self.vehicles else 0.0
        avg_dist = np.mean([v.total_distance for v in self.vehicles]) if self.vehicles else 0.0
        avg_risk = np.mean([v.risk_exposure for v in self.vehicles]) if self.vehicles else 0.0
        total_replans = sum([v.replan_count for v in self.vehicles])

        return {
            'algorithm': self.algorithm,
            'total_vehicles': len(self.vehicles),
            'completed_vehicles': len(completed_vehicles),
            'completion_rate_pct': round(completion_rate, 2),
            'avg_travel_time_sec': round(avg_time, 2),
            'avg_distance_m': round(avg_dist, 2),
            'avg_risk_exposure': round(avg_risk, 4),
            'total_replannings': total_replans
        }
