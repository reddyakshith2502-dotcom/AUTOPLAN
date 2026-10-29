import random

class DynamicTrafficEnvironment:
    """
    Manages real-time dynamic events occurring during simulation steps:
    1. Traffic Jams (Travel Time Spike)
    2. Road Blockages (Lane/Edge Closed)
    3. Collision Risk Escalation (Dangerous nearby vehicle)
    4. Traffic Signal Delay (Red Light Cycle)
    """
    def __init__(self, road_network, seed=42):
        self.road_network = road_network
        self.rng = random.Random(seed)
        self.active_events = []

    def trigger_traffic_jam(self, edge, delay_factor=3.5):
        u, v = edge
        if self.road_network.graph.has_edge(u, v):
            edge_data = self.road_network.graph[u][v]
            base_time = edge_data['distance'] / edge_data['speed_limit_mps']
            new_time = base_time * delay_factor
            self.road_network.update_edge_condition(u, v, travel_time=new_time)
            self.active_events.append(('traffic_jam', edge, new_time))
            return True
        return False

    def trigger_road_blockage(self, edge):
        u, v = edge
        if self.road_network.graph.has_edge(u, v):
            self.road_network.update_edge_condition(u, v, is_blocked=True)
            self.active_events.append(('road_blockage', edge, True))
            return True
        return False

    def trigger_high_risk_zone(self, edge, risk_level=0.85):
        u, v = edge
        if self.road_network.graph.has_edge(u, v):
            self.road_network.update_edge_condition(u, v, risk=risk_level)
            self.active_events.append(('high_risk', edge, risk_level))
            return True
        return False

    def trigger_signal_red(self, edge, wait_seconds=15.0):
        u, v = edge
        if self.road_network.graph.has_edge(u, v):
            self.road_network.update_edge_condition(u, v, signal_delay=wait_seconds)
            self.active_events.append(('signal_red', edge, wait_seconds))
            return True
        return False

    def clear_all_events(self):
        self.active_events.clear()
        self.road_network.reset_network()
