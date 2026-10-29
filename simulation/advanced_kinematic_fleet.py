import time
import numpy as np

from simulation.road_network import UrbanRoadNetwork
from simulation.adaptive_astar import ml_dynamic_risk_aware_astar
from simulation.kinematics import IntelligentDriverModel, QuinticSplinePlanner, KinematicBicycleModel
from simulation.dynamic_events import DynamicTrafficEnvironment

class AdvancedKinematicVehicle:
    """
    Autonomous Vehicle with Dual-Layer Hierarchical Planning:
    - Global Layer: ML Dynamic Risk-Aware A*
    - Local Layer: Intelligent Driver Model (IDM) + Quintic Spline Smooth Tracking
    """
    def __init__(self, vehicle_id, start_node, goal_node, color="#0284C7", is_ego=False, desired_speed=14.0):
        self.vehicle_id = vehicle_id
        self.start_node = start_node
        self.goal_node = goal_node
        self.current_node = start_node
        self.color = color
        self.is_ego = is_ego
        
        # Longitudinal Controller (IDM)
        self.idm = IntelligentDriverModel(
            desired_speed=desired_speed,
            time_headway=1.2,
            min_spacing=3.0,
            max_accel=2.2,
            comfortable_decel=2.8
        )
        
        # Kinematic State
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.velocity = 0.0  # m/s
        self.acceleration = 0.0  # m/s^2
        self.jerk = 0.0  # m/s^3
        self.steering_angle = 0.0
        
        # Navigation & Spline Tracking
        self.node_path = []
        self.spline_trajectory = []  # List of (x, y, target_speed)
        self.trajectory_index = 0
        
        # Performance & Safety Telemetry
        self.total_distance = 0.0
        self.total_time = 0.0
        self.risk_exposure = 0.0
        self.replan_count = 0
        self.completed = False
        self.current_ttc = 10.0
        self.current_gap = 50.0

    def plan_global_route(self, road_network):
        """Global Layer: Dynamic Risk-Aware A*"""
        path, cost = ml_dynamic_risk_aware_astar(road_network, self.current_node, self.goal_node)
        if path and len(path) >= 1:
            self.node_path = path
            self._generate_smooth_spline(road_network)
            return True
        return False

    def _generate_smooth_spline(self, road_network):
        """Local Layer: Generates C^2 Continuous Quintic Splines through corridor nodes"""
        if len(self.node_path) < 2:
            return
        
        all_points = []
        for i in range(len(self.node_path) - 1):
            u = self.node_path[i]
            v = self.node_path[i+1]
            pos_u = road_network.get_node_pos(u)
            pos_v = road_network.get_node_pos(v)
            
            # Determine headings
            heading_u = np.arctan2(pos_v[1] - pos_u[1], pos_v[0] - pos_u[0])
            
            if i + 2 < len(self.node_path):
                w = self.node_path[i+2]
                pos_w = road_network.get_node_pos(w)
                heading_v = np.arctan2(pos_w[1] - pos_v[1], pos_w[0] - pos_v[0])
            else:
                heading_v = heading_u
                
            segment_pts = QuinticSplinePlanner.generate_quintic_curve(
                pos_u, pos_v, heading_u, heading_v, num_points=25
            )
            
            # Append points, avoiding duplicate endpoints
            if not all_points:
                all_points.extend(segment_pts)
            else:
                all_points.extend(segment_pts[1:])
                
        self.spline_trajectory = all_points
        self.trajectory_index = 0
        if self.spline_trajectory:
            self.x, self.y = self.spline_trajectory[0]

    def update_kinematics(self, leading_vehicle=None, traffic_light_state="GREEN", dt=0.05):
        """
        Executes one local kinematic update cycle using IDM and pure-pursuit trajectory following.
        """
        if self.completed or not self.spline_trajectory:
            return

        # 1. Measure gap and relative velocity to leading vehicle
        dist_to_lead = None
        speed_lead = None
        if leading_vehicle is not None:
            dx = leading_vehicle.x - self.x
            dy = leading_vehicle.y - self.y
            dist_to_lead = max(0.5, np.hypot(dx, dy) - 4.5)  # 4.5m vehicle length
            speed_lead = leading_vehicle.velocity
            self.current_gap = dist_to_lead
            
            # Calculate dynamic Time-To-Collision (TTC)
            closing_speed = self.velocity - speed_lead
            if closing_speed > 0.1:
                self.current_ttc = min(10.0, dist_to_lead / closing_speed)
            else:
                self.current_ttc = 10.0
        else:
            self.current_gap = 100.0
            self.current_ttc = 10.0

        # 2. Check virtual stop condition for RED traffic signals
        # If approaching a red traffic light at upcoming node
        if traffic_light_state == "RED" and len(self.spline_trajectory) - self.trajectory_index < 30:
            dist_to_signal = max(0.5, (len(self.spline_trajectory) - self.trajectory_index) * 1.5)
            if dist_to_lead is None or dist_to_signal < dist_to_lead:
                dist_to_lead = dist_to_signal
                speed_lead = 0.0

        # 3. IDM Acceleration computation
        new_accel = self.idm.compute_acceleration(
            ego_speed=self.velocity,
            leading_speed=speed_lead,
            distance_gap=dist_to_lead
        )
        
        # Calculate jerk (da / dt)
        self.jerk = (new_accel - self.acceleration) / dt
        self.acceleration = new_accel

        # 4. Velocity integration
        self.velocity = max(0.0, self.velocity + self.acceleration * dt)
        
        # 5. Advance along smooth spline curve
        dist_to_advance = self.velocity * dt
        self.total_distance += dist_to_advance
        self.total_time += dt

        # Pure-pursuit lookahead to advance spline index
        travelled = 0.0
        while self.trajectory_index < len(self.spline_trajectory) - 1 and travelled < dist_to_advance:
            p_curr = np.array(self.spline_trajectory[self.trajectory_index])
            p_next = np.array(self.spline_trajectory[self.trajectory_index + 1])
            step_len = np.linalg.norm(p_next - p_curr)
            travelled += step_len
            self.trajectory_index += 1

        if self.trajectory_index < len(self.spline_trajectory):
            target_pt = self.spline_trajectory[self.trajectory_index]
            self.yaw = np.arctan2(target_pt[1] - self.y, target_pt[0] - self.x)
            self.x, self.y = target_pt
        else:
            self.completed = True


class AdvancedKinematicFleetSimulator:
    """
    Multi-Vehicle Simulator equipped with Advanced Kinematics (Upgrade 1).
    """
    def __init__(self, num_vehicles=5, seed=42):
        self.network = UrbanRoadNetwork()
        self.env = DynamicTrafficEnvironment(self.network, seed=seed)
        self.vehicles = []
        self._init_fleet(num_vehicles)

    def _init_fleet(self, num_vehicles):
        colors = ["#0284C7", "#EF4444", "#EAB308", "#F97316", "#16A34A", "#8B5CF6", "#EC4899"]
        od_pairs = [
            ("A", "I"),
            ("A", "F"),
            ("D", "I"),
            ("B", "H"),
            ("G", "C")
        ]
        
        for i in range(min(num_vehicles, len(od_pairs))):
            start, goal = od_pairs[i]
            veh = AdvancedKinematicVehicle(
                vehicle_id=f"AV_{i+1}",
                start_node=start,
                goal_node=goal,
                color=colors[i % len(colors)],
                is_ego=(i == 0),
                desired_speed=12.0 + (i * 0.8)
            )
            veh.plan_global_route(self.network)
            self.vehicles.append(veh)

    def step(self, traffic_signal_state="GREEN", dt=0.05):
        # Update vehicles in topological order along road
        for i, veh in enumerate(self.vehicles):
            leading_veh = self.vehicles[i-1] if i > 0 and not self.vehicles[i-1].completed else None
            veh.update_kinematics(leading_vehicle=leading_veh, traffic_light_state=traffic_signal_state, dt=dt)
