import heapq
import numpy as np

def euclidean_heuristic(network, node_a, node_b):
    pos_a = np.array(network.get_node_pos(node_a))
    pos_b = np.array(network.get_node_pos(node_b))
    return float(np.linalg.norm(pos_b - pos_a))

def standard_astar_route(road_network, start_node, goal_node):
    """
    Baseline 2 — Standard A* Algorithm
    f(n) = g(n) + h(n)
    where g(n) = accumulated distance cost so far
          h(n) = Euclidean distance heuristic to destination
    Ignores dynamic ML safety risk scores and traffic delays.
    """
    graph = road_network.graph

    if start_node not in graph or goal_node not in graph:
        return None, float('inf')

    g_score = {node: float('inf') for node in graph.nodes}
    f_score = {node: float('inf') for node in graph.nodes}
    previous_nodes = {node: None for node in graph.nodes}

    g_score[start_node] = 0.0
    f_score[start_node] = euclidean_heuristic(road_network, start_node, goal_node)

    open_set = [(f_score[start_node], start_node)]

    while open_set:
        current_f, current_node = heapq.heappop(open_set)

        if current_node == goal_node:
            break

        if current_f > f_score[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            edge_data = graph[current_node][neighbor]

            if edge_data.get('is_blocked', False):
                continue

            tentative_g = g_score[current_node] + edge_data.get('distance', 1.0)

            if tentative_g < g_score[neighbor]:
                previous_nodes[neighbor] = current_node
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + euclidean_heuristic(road_network, neighbor, goal_node)
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
