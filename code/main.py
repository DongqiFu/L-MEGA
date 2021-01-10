import time
import math
import numpy as np
from collections import defaultdict
from numpy.linalg import inv
from scipy.sparse import lil_matrix
import scipy.sparse as sp


def get_updated_edge_set(new_arrival_graph):
    edge_set = set()
    updates_edges = open(new_arrival_graph, 'r')
    for line in updates_edges.readlines():
        items = line.strip().split(',')
        node_0 = int(items[0])
        node_1 = int(items[1])
        edge_set.add((node_0, node_1))
    return edge_set


def update_transition_tensor(updated_edge_set, local_cluster, adj_matrix, unfold_transition_tensor):
    num_nodes = adj_matrix.shape[0]
    saved_updated_edges_set = set()

    for edge in updated_edge_set:
        node_u = edge[0]
        node_v = edge[1]

        node_u_one_hop = np.nonzero(adj_matrix[node_u, :])[0]
        node_v_one_hop = np.nonzero(adj_matrix[node_v, :])[0]

        if node_u not in local_cluster and node_v not in local_cluster and bool(set(node_u_one_hop) & set(local_cluster)) is False and bool(set(node_v_one_hop) & set(local_cluster)) is False:
            saved_updated_edges_set.add((node_u, node_v))

        else:  # --- (node_u, node_v) cannot be filtered, update tensors concerning (node_u, node_v) --- #
            adj_matrix[node_u, node_v] = 1
            adj_matrix[node_v, node_u] = 1

            involving_nodes = set(node_u_one_hop) & set(node_v_one_hop)
            for node in involving_nodes:
                non_zero_index = np.nonzero(unfold_transition_tensor[node, :])
                node_new_value = 1 / (len(non_zero_index) + 1)
                unfold_transition_tensor[node, :][unfold_transition_tensor[node, :] > 0.0] = node_new_value
                unfold_transition_tensor[node, (node_u * num_nodes + node_v)] = node_new_value
                unfold_transition_tensor[node, (node_v * num_nodes + node_u)] = node_new_value

                non_zero_index = np.nonzero(unfold_transition_tensor[node_u, :])
                node_u_new_value = 1 / (len(non_zero_index) + 1)
                unfold_transition_tensor[node_u, :][unfold_transition_tensor[node_u, :] > 0.0] = node_u_new_value
                unfold_transition_tensor[node_u, (node * num_nodes + node_v)] = node_u_new_value
                unfold_transition_tensor[node_u, (node_v * num_nodes + node)] = node_u_new_value

                non_zero_index = np.nonzero(unfold_transition_tensor[node_v, :])
                node_v_new_value = 1 / (len(non_zero_index) + 1)
                unfold_transition_tensor[node_v, :][unfold_transition_tensor[node_v, :] > 0.0] = node_v_new_value
                unfold_transition_tensor[node_v, (node * num_nodes + node_u)] = node_v_new_value
                unfold_transition_tensor[node_v, (node_u * num_nodes + node)] = node_v_new_value

                # for i in range(num_nodes):
                #     unfold_transition_tensor[:, (node + i * num_nodes)][unfold_transition_tensor[:, (node + i * num_nodes)] > 0.0] = node_new_value
                #     unfold_transition_tensor[:, (node_u + i * num_nodes)][unfold_transition_tensor[:, (node_u + i * num_nodes)] > 0.0] = node_u_new_value
                #     unfold_transition_tensor[:, (node_v + i * num_nodes)][unfold_transition_tensor[:, (node_v + i * num_nodes)] > 0.0] = node_v_new_value

    return saved_updated_edges_set, adj_matrix, unfold_transition_tensor.tocsr()


def motif_push_operation(alpha, pagerank, residual_vector, transition_tensor, next_transition_tensor, push_operation_threshold):
    # calculate the initial residual vector which is inherited from the last time stamp --- #
    residual_vector = residual_vector + alpha * (next_transition_tensor - transition_tensor).dot(np.kron(pagerank, pagerank))
    residual_vector = residual_vector / np.sum(residual_vector)

    num_nodes = transition_tensor.shape[0]

    while np.amax(residual_vector) > push_operation_threshold:
        ri = np.amax(residual_vector)
        index = np.argmax(residual_vector)
        basis_vector = np.zeros((num_nodes,))
        basis_vector[index] = 1
        last_pagerank = pagerank
        pagerank = last_pagerank + ri * basis_vector
        residual_vector = residual_vector - ri * basis_vector + alpha * ri * next_transition_tensor.dot(np.kron(basis_vector, last_pagerank))

        pagerank = pagerank / np.sum(pagerank)
        residual_vector = residual_vector / np.sum(residual_vector)

    return pagerank, residual_vector


def incremental_sweep_cut(local_cluster, pagerank, adj_matrix, deg_matrix, upper_bound_phi):
    num_nodes = adj_matrix.shape[0]

    mu_v = np.sum(deg_matrix)

    score = pagerank / np.sum(deg_matrix, axis=0)
    score_rank_index = np.argsort(score, kind='heapsort')[::-1]

    # --- find longest common length --- #
    q = 0
    for i in range(len(local_cluster)):
        if local_cluster[i] == score_rank_index[i]:
            q += 1
        else:
            break

    if q < 2:
        q = 2

    next_cluster = score_rank_index[:q + 1]
    next_volume = np.sum(deg_matrix[:, next_cluster])
    next_temp = np.sum(adj_matrix[next_cluster, :][:, next_cluster])
    next_phi = (next_volume - next_temp) / min(next_volume, mu_v - next_volume)

    for j in range(q + 2, num_nodes):
        cluster = next_cluster
        phi = next_phi

        next_cluster = score_rank_index[:j + 1]
        next_volume = np.sum(deg_matrix[:, next_cluster])
        next_temp = np.sum(adj_matrix[next_cluster, :][:, next_cluster])
        next_phi = (next_volume - next_temp) / min(next_volume, mu_v - next_volume)

        if phi <= upper_bound_phi:
            if phi < next_phi:
                return cluster, phi

    # --- does not find satisfied local cluster --- #
    return [], None


if __name__ == '__main__':
    graph_name_list = ['Dataset/Alpha/alpha', 'Dataset/OTC/otc']
    for graph_name in graph_name_list:
        print('-----------------------------------------------------')
        print("Data set: " + graph_name.split('/')[1])

        print('---> Loading corresponding matrices at t0 ...')
        adj_matrix = np.load(graph_name + '_t0_adj_mat.npy')
        pagerank = np.load(graph_name + '_t0_pr_vec.npy')
        local_cluster = np.load(graph_name + '_t0_clus_vec.npy')
        transition_tensor = sp.load_npz(graph_name + '_t0_trans_tens.npz')

        seed_node = 500
        upper_bound_phi = 0.48
        alpha = 0.85
        push_operation_threshold = 0.1

        # --- calculate the residual vector of HOSPLOC at t0 --- #
        num_nodes = adj_matrix.shape[0]
        personalized_vector = np.zeros((num_nodes,), dtype=np.float32)
        personalized_vector[seed_node] = 1
        residual_vector = alpha * transition_tensor.dot(np.kron(pagerank, pagerank)) + (1 - alpha) * personalized_vector - pagerank
        residual_vector = residual_vector / np.sum(residual_vector)

        if len(local_cluster) != 0:  # --- HOSPLOC finds the local cluster at t0 --- #
            print('---> L-MEGA starts to update transition tensor, and track multilinear PageRank, and track local cluster from t0 to t10 ...')
            total_time = 0
            saved_updated_edge_set = {}
            for t in range(10):
                new_arrival_graph = graph_name + '_d' + str(t) + '.txt'
                updated_edge_set = get_updated_edge_set(new_arrival_graph)
                updated_edge_set = updated_edge_set.union(saved_updated_edge_set)

                # --- start to update transition tensor --- #
                print('------> L-MEGA starts to update transition tensor at t' + str(t + 1) + '.')
                start_time = time.time()

                saved_updated_edge_set, adj_matrix, transition_tensor = update_transition_tensor(
                    updated_edge_set, local_cluster, adj_matrix, transition_tensor.tolil())

                updating_time = time.time() - start_time
                np.save(graph_name + '_t' + str(t + 1) + '_adj_mat', adj_matrix)
                sp.save_npz(graph_name + '_t' + str(t + 1) + '_trans_tens', transition_tensor)

                # --- start to track multilinear PageRank --- #
                print('------> L-MEGA starts to track multilinear PageRank at t' + str(t + 1) + '.')
                transition_tensor = sp.load_npz(graph_name + '_t' + str(t) + '_trans_tens.npz')
                next_transition_tensor = sp.load_npz(graph_name + '_t' + str(t + 1) + '_trans_tens.npz')
                start_time = time.time()

                pagerank, residual_vector = motif_push_operation(alpha, pagerank, residual_vector, transition_tensor,
                                                          next_transition_tensor, push_operation_threshold)

                tracking_time = time.time() - start_time

                # --- start to track local cluster --- #
                e = np.ones((num_nodes,), dtype=int)
                deg_matrix = np.diag(np.dot(adj_matrix, e))
                for i in range(num_nodes):  # add self-loops for dangling nodes
                    if deg_matrix[i, i] == 0:
                        deg_matrix[i, i] = 1
                print('------> L-MEGA starts to track local cluster at t' + str(t + 1) + '.')
                start_time = time.time()

                local_cluster, phi = incremental_sweep_cut(local_cluster, pagerank, adj_matrix, deg_matrix, upper_bound_phi)

                clustering_time = time.time() - start_time
                np.save(graph_name + '_t' + str(t + 1) + '_clus_vec', local_cluster)
                print('------> L-MEGA finishes at t' + str(t + 1) + '.')
                total_time += tracking_time
                total_time += tracking_time
                total_time += clustering_time
            print('---> L-MEGA costs time: ' + str(total_time / 10) + ' sec in each time stamp.')
            print('---> L-MEGA finds the local cluster under the conductance: ' + str(phi) + '.')

            print('---> Start evaluating performance of local cluster in terms of third-order conductance and triangle density at t10.')
            print('------> Start evaluating third-order conductance.')
            weighted_adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if adj_matrix[i, j] == 1:
                        for k in range(j, num_nodes):
                            if adj_matrix[j, k] == 1 and adj_matrix[k, i] == 1:
                                weighted_adj_matrix[i, j] += 1
                                weighted_adj_matrix[i, k] += 1
                                weighted_adj_matrix[j, i] += 1
                                weighted_adj_matrix[j, k] += 1
                                weighted_adj_matrix[k, i] += 1
                                weighted_adj_matrix[k, j] += 1
            # --- third-order conductance --- #
            mu = np.sum(weighted_adj_matrix)
            volume = np.sum(weighted_adj_matrix[:, local_cluster])
            temp = np.sum(weighted_adj_matrix[local_cluster, :][:, local_cluster])
            if volume == 0 or volume == mu:
                third_conductance = 1
            else:
                third_conductance = (volume - temp) / min(volume, mu - volume)
            print('------> Third-order conductance: ' + str(third_conductance))
            # --- triangle density --- #
            volume = np.sum(weighted_adj_matrix[local_cluster, :][:, local_cluster])
            triangle_density = volume / mu
            print('------> Triangle density: ' + str(triangle_density))
        else:
            print('---> HOSPOLC does not find suitable local cluster at t0.')