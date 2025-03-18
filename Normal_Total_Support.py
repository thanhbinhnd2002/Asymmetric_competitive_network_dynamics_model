import os
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

def import_network(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    
    net = nx.DiGraph()
    for line in data[1:]:  # Bỏ qua dòng đầu tiên (tiêu đề nếu có)
        from_node, to_node, direction, weight = line.strip().split("\t")
        direction = int(direction)
        weight = int(weight)
        
        net.add_edge(from_node, to_node, weight=weight)
        if direction == 0:
            net.add_edge(to_node, from_node, weight=weight)
    
    return net

def extract_adj_matrix(nodes, edges, net):
    node_dict = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=float)  # Chuyển thành float để lưu trọng số
    neighbors = {i: set() for i in range(n_nodes)}
    
    for from_node, to_node in edges:
        from_idx, to_idx = node_dict[from_node], node_dict[to_node]
        weight = net[from_node][to_node]['weight']  # Lấy trọng số từ đồ thị
        adj_matrix[from_idx][to_idx] += weight  # Sử dụng trọng số thay vì chỉ đếm liên kết	
        neighbors[to_idx].add(from_idx)
    
    return adj_matrix, neighbors, node_dict


def compute_total_support(alpha_id, states):
    support = 0
    for node_id in range(len(states)):
        if node_id == alpha_id:
            continue
        if states[node_id] > 0:
            support += 1
        elif states[node_id] < 0:
            support -= 1
    return support

def update_states(x_A, x_B, adj_matrix, neighbors, decay, gamma, lambda_, alpha_id, beta_id, max_state=1000):
    n = len(x_A)
    new_x_A = np.copy(x_A)
    new_x_B = np.copy(x_B)
            
    for i in range(n):
        if i == alpha_id or i == beta_id:
            continue  # Bỏ qua nếu là gen cố định
            
        influence_A = sum(adj_matrix[j, i] * x_A[j] * (1 - gamma * x_B[j]) for j in neighbors[i])
        influence_B = sum(adj_matrix[j, i] * x_B[j] * (1 - lambda_ * x_A[j]) for j in neighbors[i])

        new_x_A[i] = influence_A - decay * x_A[i]
        new_x_B[i] = influence_B - decay * x_B[i]
            
        # Giới hạn giá trị trong khoảng tác động của gen A và gen B


        # new_x_A[i] = max(min(new_x_A[i], x_A[alpha_id]), x_A[beta_id])
        # new_x_B[i] = max(min(new_x_B[i], x_B[beta_id]), x_B[alpha_id]) 

        new_x_A[i] = np.clip(new_x_A[i], -max_state, max_state)
        new_x_B[i] = np.clip(new_x_B[i], -max_state, max_state)
        
        # new_x_A, new_x_B = temp_x_A, temp_x_B
    
    return new_x_A, new_x_B

def compete(gen_a, gen_b, adj_matrix, neighbors, node_dict, gamma, lambda_, decay, max_iterations, alpha, beta):
    alpha_id = node_dict[gen_a]
    beta_id = node_dict[gen_b]
    n_nodes = len(node_dict)
    x_A = np.zeros(n_nodes)
    x_B = np.zeros(n_nodes)
    x_A[alpha_id] = alpha
    x_B[beta_id] = beta
    
    for _ in range(max_iterations):
        prev_x_A, prev_x_B = np.copy(x_A), np.copy(x_B)
        x_A, x_B = update_states(x_A, x_B, adj_matrix, neighbors, decay, gamma, lambda_, alpha_id, beta_id)
        
        if np.linalg.norm(x_A - prev_x_A) < 1e-6 and np.linalg.norm(x_B - prev_x_B) < 1e-6:
            break
    
    total_support_A = compute_total_support(alpha_id, x_A)
    total_support_B = compute_total_support(beta_id, x_B)
    
    if total_support_A > total_support_B:
        winner = "Gen A"
    elif total_support_B > total_support_A:
        winner = "Gen B"
    else:
        winner = "Draw"
    
    strength = abs(total_support_A - total_support_B)
    
    return winner, strength


def run_competition(data_dir, results_dir,gamma, lambda_, decay, max_iterations, alpha, beta):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # **Tạo thư mục theo giá trị beta_init**
    beta_folder = os.path.join(results_dir, f"Beta_{beta}")
    os.makedirs(beta_folder, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            dataset_name = filename.replace(".txt", "")
            file_path = os.path.join(data_dir, filename)    
            net = import_network(file_path)
            nodes, edges = list(net.nodes()), list(net.edges())
            adj_matrix, neighbors, node_dict = extract_adj_matrix(nodes, edges, net)
            gen_list = list(node_dict.keys())
            results = []

            for gen_a in tqdm(gen_list, desc=f"Processing {dataset_name}"):
                for gen_b in gen_list:
                    if gen_a == gen_b:
                        continue
                    
                    winner, strength = compete(gen_a, gen_b, adj_matrix, neighbors, node_dict, gamma, lambda_, decay, max_iterations,alpha, beta)
                    results.append([gen_a, gen_b, winner, strength])

            # **Lưu kết quả vào thư mục Beta_{beta_init}**
            df = pd.DataFrame(results, columns=["Gen A", "Gen B", "Winner", "Strength"])
            result_path = os.path.join(beta_folder, f"{dataset_name}.csv")
            df.to_csv(result_path, index=False, sep="\t")
            print(f"Results saved to {result_path}")

def main():
    data_dir = "data"
    results_dir = "results"
    
    # **Giá trị khởi tạo của beta, được sử dụng để đặt tên thư mục**
    alpha = 1
    beta = 1  # Giá trị có thể thay đổi tùy theo thử nghiệm
    gamma = 0.5
    lambda_ = 1.5
    decay = 0.1
    max_iterations = 100
    
    # **Gọi run_competition với beta_init**
    run_competition(data_dir, results_dir, gamma, lambda_, decay, max_iterations, alpha, beta)

if __name__ == "__main__":
    main()
