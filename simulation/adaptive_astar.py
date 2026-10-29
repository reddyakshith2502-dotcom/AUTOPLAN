import heapq
import numpy as np

def ml_dynamic_risk_aware_cost(edge_data, alpha=1.2, beta=50.0, gamma=1.0, delta=2.0):
    """
    Computes multi-criterion risk-aware edge cost incorporating ML model outputs:
    Cost(e) = Distance + alpha * PredictedTravelTime + beta * CollisionRisk + gamma * SignalDelay + delta * GPSUncertainty
    """
    if edge_data.get('is_blocked', False):
        return float('inf')

    distance = edge_data.get('distance', 1.0)
    pred_time = edge_data.get('predicted_travel_time', distance / 15.0)
    risk_score = edge_data.get('collision_risk', 0.0)
    signal_delay = edge_data.get('signal_delay', 0.0)
    gps_noise = edge_data.get('gps_uncertainty', 0.5)

    cost = (
        distance +
        (alpha * pred_time * 10.0) +
        (beta * risk_score * 100.0) +
        (gamma * signal_delay * 5.0) +
        (delta * gps_noise)
    )

    return cost

def adaptive_risk_heuristic(road_network, node_a, node_b):
    pos_a = np.array(road_network.get_node_pos(node_a))
    pos_b = np.array(road_network.get_node_pos(node_b))
    dist = float(np.linalg.norm(pos_b - pos_a))
    # Heuristic estimates minimum distance cost
    return dist

def ml_dynamic_risk_aware_astar(road_network, start_node, goal_node, alpha=1.2, beta=50.0, gamma=1.0, delta=2.0):
    """
    Proposed Method — ML-Driven Dynamic Risk-Aware A* (DAA)
    Fuses dual ML Random Forest predictions (Travel Time Regressor + Collision Risk Classifier)
    with real-time signal delays and road blockages.
    """
    graph = road_network.graph

    if start_node not in graph or goal_node not in graph:
        return None, float('inf')

    g_score = {node: float('inf') for node in graph.nodes}
    f_score = {node: float('inf') for node in graph.nodes}
    previous_nodes = {node: None for node in graph.nodes}

    g_score[start_node] = 0.0
    f_score[start_node] = adaptive_risk_heuristic(road_network, start_node, goal_node)

    open_set = [(f_score[start_node], start_node)]

    while open_set:
        current_f, current_node = heapq.heappop(open_set)

        if current_node == goal_node:
            break

        if current_f > f_score[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            edge_data = graph[current_node][neighbor]

            edge_cost = ml_dynamic_risk_aware_cost(
                edge_data,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta
            )

            if np.isinf(edge_cost):
                continue

            tentative_g = g_score[current_node] + edge_cost

            if tentative_g < g_score[neighbor]:
                previous_nodes[neighbor] = current_node
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + adaptive_risk_heuristic(road_network, neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr)
        curr = previous_nodes[curr]

    path.reverse()

    if not path or path[0] != start_node:
        return None, float('inf')

    return path, g_score[goal_node]
