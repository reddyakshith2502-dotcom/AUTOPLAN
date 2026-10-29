import numpy as np
import networkx as nx

class UrbanRoadNetwork:
    """
    Grid & Highway Road Network for Autonomous Vehicle Navigation.
    Nodes: Intersections / Key Highway Landmarks (e.g. A..I or 0..N)
    Edges: Directed road segments carrying physical, traffic, signal, and ML risk properties.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_default_network()

    def _build_default_network(self):
        # 3x3 Grid / Highway Nodes with 2D Position Coordinates (x, y in meters)
        self.nodes = {
            'A': (0, 0),    'B': (500, 0),   'C': (1000, 0),
            'D': (0, 500),  'E': (500, 500),  'F': (1000, 500),
            'G': (0, 1000), 'H': (500, 1000), 'I': (1000, 1000)
        }

        for node_id, pos in self.nodes.items():
            self.graph.add_node(node_id, pos=pos)

        # Standard Bidirectional Edges with default properties
        edges = [
            ('A', 'B'), ('B', 'C'),
            ('D', 'E'), ('E', 'F'),
            ('G', 'H'), ('H', 'I'),
            ('A', 'D'), ('D', 'G'),
            ('B', 'E'), ('E', 'H'),
            ('C', 'F'), ('F', 'I')
        ]

        for u, v in edges:
            pos_u = np.array(self.nodes[u])
            pos_v = np.array(self.nodes[v])
            dist = float(np.linalg.norm(pos_v - pos_u))
            
            edge_attr = {
                'distance': dist,
                'speed_limit_mps': 15.0,  # 15 m/s (~54 km/h)
                'predicted_travel_time': dist / 15.0,
                'collision_risk': 0.05,
                'signal_delay': 0.0,
                'gps_uncertainty': 0.5,
                'is_blocked': False,
                'current_density': 10.0
            }

            self.graph.add_edge(u, v, **edge_attr)
            self.graph.add_edge(v, u, **edge_attr)

    def get_node_pos(self, node_id):
        return self.nodes.get(node_id, (0, 0))

    def update_edge_condition(self, u, v, travel_time=None, risk=None, signal_delay=None, is_blocked=None):
        if self.graph.has_edge(u, v):
            edge = self.graph[u][v]
            if travel_time is not None:
                edge['predicted_travel_time'] = float(travel_time)
            if risk is not None:
                edge['collision_risk'] = float(risk)
            if signal_delay is not None:
                edge['signal_delay'] = float(signal_delay)
            if is_blocked is not None:
                edge['is_blocked'] = bool(is_blocked)

    def reset_network(self):
        self._build_default_network()
