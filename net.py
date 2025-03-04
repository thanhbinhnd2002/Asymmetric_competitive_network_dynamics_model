# Undirected connection will be converted to 2 directed connections
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import networkx as nx
import numpy as np
from tqdm import tqdm

INF = 10000


def import_network(filename):
    with open(filename, "r") as f:
        data = f.readlines()

    count = 0
    net = nx.MultiDiGraph()
    for line in data:
        if count == 0:
            count += 1
            continue
        from_node, to_node, direction, weight = line.strip().split("\t")
        direction = int(direction)
        weight = int(weight)

        net.add_edge(from_node, to_node, weight=weight)

        if direction == 0:
            net.add_edge(to_node, from_node, weight=weight)

    return net


def get_node_edge(net):
    nodes = list(net.nodes())
    edges = list(net.edges())
    return nodes, edges


def extract_adj_matrix(nodes, edges):
    n_nodes = len(nodes)

    # Node dictionary
    node_dict = {}
    for i in range(n_nodes):
        node_dict[nodes[i]] = i

    # Adjacent directed matrix
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    # Update adj matrix and neighbors. Neighbors
    # is used to only loop through neighbors of each node
    # instead of iterating through all nodes
    neighbors = {}
    for edge in edges:
        from_node, to_node = edge
        from_node_idx = node_dict[from_node]
        to_node_idx = node_dict[to_node]
        adj_matrix[from_node_idx][to_node_idx] += 1

        if neighbors.get(to_node) is None:
            neighbors[to_node] = set()

        if from_node not in neighbors[to_node]:
            neighbors[to_node].add(from_node)

    return adj_matrix, neighbors, node_dict


def compute_distance_matrix(dataset, adj_matrix, node_dict):
    d = np.where(adj_matrix > 0, 1, 0)
    d[(d == 0) & (~np.eye(d.shape[0], dtype=bool))] = INF

    # Floyd-Warshall algorithm for distance matrix D calculation
    loop_obj = tqdm(node_dict.keys())
    for k in loop_obj:
        loop_obj.set_description("Computing distance matrix")
        for u in node_dict.keys():
            for v in node_dict.keys():
                from_node = node_dict[u]
                to_node = node_dict[v]
                mid_node = node_dict[k]

                candidate = d[from_node][mid_node] + d[mid_node][to_node]
                if d[from_node][to_node] > candidate:
                    d[from_node][to_node] = candidate

    np.savetxt(f"distance_matrix/{dataset}_distance_matrix.csv", d, delimiter=",", fmt='%d')

    return d


def compete(alpha, adj_matrix, neighbors, node_dict, n_edges):
    alpha_id = node_dict[alpha]

    # Init outside competitor
    beta_id = len(node_dict)

    # Init node state (for beta as well)
    n_nodes = len(node_dict)
    states = {}
    for i in range(n_nodes + 1):
        states[i] = 0
    states[alpha_id] = 1
    states[beta_id] = -1

    n_steps = n_nodes * n_edges
    # Connect Beta to each normal agent
    for node in node_dict.keys():
        node_id = node_dict[node]
        # Beta cannot connect to Beta and Alpha
        if node_id == alpha_id or node_id == beta_id:
            continue

        # Handle node with no in-edge
        if neighbors.get(node, 0) == 0:
            neighbors[node] = set()

        # Set of V edges
        neighbors[node].add("Beta")

        # Undirected connection turned to 2 directed connections
        deg_max = max(np.sum(adj_matrix, axis=-1))
        epsilon = 1 / deg_max

        t = 0
        while True:
            converging = 0

            for u in node_dict.keys():
                u_id = node_dict[u]
                if u_id == alpha_id or u_id == beta_id:
                    continue

                # Node with no out-edge
                if neighbors.get(u, 0) == 0:
                    continue

                s = 0
                # Updating based on neighbors
                for v in neighbors[u]:
                    if v == "Beta":
                        s += states[beta_id] - states[u_id]
                    else:
                        v_id = node_dict[v]
                        s += adj_matrix[v_id][u_id] * (states[v_id] - states[u_id])

                old_u_state = states[u_id]
                states[u_id] = old_u_state + epsilon * s
                converging = converging + abs(states[u_id] - old_u_state)

            t += 1

            if not (converging > epsilon and t < n_steps):
                break

        # Remove connection from B
        neighbors[node].remove("Beta")

    return alpha_id, states


def compute_influence_matrix(states, distance_matrix, node_dict):
    n_nodes = len(node_dict)
    influence_matrix = np.zeros((n_nodes, n_nodes))

    for u in node_dict.keys():
        for v in node_dict.keys():
            u_id = node_dict[u]
            v_id = node_dict[v]
            if distance_matrix[u_id][v_id] != 0:
                influence_matrix[u_id][v_id] = states[v_id] / (distance_matrix[v_id][u_id] ** 2)

    return influence_matrix


def compute_total_support(alpha_id, influence_matrix, node_dict, states):
    def sign(value):
        if value > 0:
            return 1
        elif value == 0:
            return 0
        else:
            return -1

    support = 0
    for node in node_dict.keys():
        node_id = node_dict[node]
        if node_id == alpha_id:
            continue

        support += sign(influence_matrix[alpha_id][node_id] - states[node_id])

    return support


def compute_total_support_all(alpha, adj_matrix, distance_matrix, neighbors, node_dict, n_edges):
    alpha_id = node_dict[alpha]
    states = compete(alpha_id, adj_matrix, neighbors, node_dict, n_edges)
    influence_matrix = compute_influence_matrix(states, distance_matrix, node_dict)
    total_support = compute_total_support(alpha_id, influence_matrix, node_dict, states)
    return alpha, total_support


def main():
    datasets = os.listdir("./data")
    data_objects = [(os.path.join("./data", dataset), dataset.split(".txt")[0].strip()) for dataset in datasets]
    print(data_objects)

    # dataset = "human_cancer_signaling"
    # path = "./data/4-Human cancer signaling - Input.txt"

    # dataset = "test"
    # path = "./data/test.txt"

    # Network generation
    for path, dataset in data_objects:
        net = import_network(path)
        nodes, edges = get_node_edge(net)
        adj_matrix, neighbors, node_dict = extract_adj_matrix(nodes, edges)

        # Compute distance matrix
        distance_matrix = compute_distance_matrix(dataset, adj_matrix, node_dict)
        # distance_matrix = np.loadtxt(f"distance_matrix    /{dataset}_distance_matrix.csv", delimiter=",")

        # Outside competition
        n_edges = len(edges)
        with ProcessPoolExecutor() as executor:
            states = list(tqdm(executor.map(
                partial(compete, adj_matrix=adj_matrix,
                        neighbors=neighbors, node_dict=node_dict,
                        n_edges=n_edges),
                node_dict.keys())))

        total_supports = {}
        for alpha_id, state in states:
            influence_matrix = compute_influence_matrix(state, distance_matrix, node_dict)
            support = compute_total_support(alpha_id, influence_matrix, node_dict, state)
            total_supports[alpha_id] = support

        with open(f"total_support/{dataset}_total_supports.csv", "w") as f:
            id_to_node = {v: k for k, v in node_dict.items()}
            f.write("Node_ID, Node, Total Support\n")
            for node, support in total_supports.items():
                f.write(f"{node}, {id_to_node[node]}, {support}\n")


if __name__ == "__main__":
    main()
