import heapq
import numpy as np

def dijkstra_route(road_network, start_node, goal_node, weight_key='distance'):
    """
    Baseline 1 — Standard Dijkstra Algorithm
    Calculates the route minimizing accumulated spatial distance or static edge weight.
    Ignores dynamic ML risk and real-time travel-time predictions.
    """
    graph = road_network.graph

    if start_node not in graph or goal_node not in graph:
        return None, float('inf')

    distances = {node: float('inf') for node in graph.nodes}
    previous_nodes = {node: None for node in graph.nodes}
    distances[start_node] = 0.0

    pq = [(0.0, start_node)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_node == goal_node:
            break

        if current_dist > distances[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            edge_data = graph[current_node][neighbor]
            
            # If road segment is blocked, treat edge cost as infinite
            if edge_data.get('is_blocked', False):
                continue

            cost = edge_data.get(weight_key, edge_data.get('distance', 1.0))
            distance = current_dist + cost

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    # Reconstruct path
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr)
        curr = previous_nodes[curr]

    path.reverse()

    if not path or path[0] != start_node:
        return None, float('inf')

    return path, distances[goal_node]
