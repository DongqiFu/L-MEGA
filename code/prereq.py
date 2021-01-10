# This piece of code provides the initial clustering result at t0 for triggering the incremental L-MEGA algorithm. #

import time
import math
import numpy as np
from collections import defaultdict
from numpy.linalg import inv
from scipy.sparse import lil_matrix
import scipy.sparse as sp


def build_matrices(graph_name):
    nodes_set = set()
    t_10_graph = open(graph_name + '_t10.txt', 'r')
    for line in t_10_graph.readlines():
        items = line.strip().split(',')
        node_0 = int(items[0])
        node_1 = int(items[1])
        nodes_set.add(node_0)
        nodes_set.add(node_1)
    t_10_graph.close()
    num_nodes = len(nodes_set)

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    t_0_graph = open(graph_name + '_t0.txt', 'r')
    for line in t_0_graph.readlines():
        items = line.strip().split(',')
        node_0 = int(items[0])
        node_1 = int(items[1])
        adj_matrix[node_0, node_1] = 1
        adj_matrix[node_1, node_0] = 1
    t_0_graph.close()

    e = np.ones((num_nodes,), dtype=int)
    deg_matrix = np.diag(np.dot(adj_matrix, e))
    for i in range(num_nodes):  # add self-loops for dangling nodes
        if deg_matrix[i, i] == 0:
            deg_matrix[i, i] = 1

    id_matrix = np.diag(e)  # identity matrix

    trans_matrix = (np.dot(np.transpose(adj_matrix), inv(deg_matrix)) + id_matrix) / 2

    # --- build indicator tensor and transition tensor at t0 --- #
    unfold_indicator_tensor = lil_matrix((num_nodes, num_nodes * num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] == 1:
                for k in range(j + 1, num_nodes):
                    if adj_matrix[j, k] == 1 and adj_matrix[k, i] == 1:
                        unfold_indicator_tensor[i, (j * num_nodes + k)] = 1
                        unfold_indicator_tensor[i, (k * num_nodes + j)] = 1
                        unfold_indicator_tensor[j, (i * num_nodes + k)] = 1
                        unfold_indicator_tensor[j, (k * num_nodes + i)] = 1
                        unfold_indicator_tensor[k, (i * num_nodes + j)] = 1
                        unfold_indicator_tensor[k, (j * num_nodes + i)] = 1
    unfold_transition_tensor = lil_matrix((num_nodes, num_nodes * num_nodes), dtype=np.float32)
    # --- build transition tensor at t0 --- #
    for i in range(num_nodes):
        total_num = unfold_indicator_tensor[i, :].sum()
        if total_num != 0:
            for j in range(i + 1, num_nodes):
                if adj_matrix[i, j] == 1:
                    for k in range(j + 1, num_nodes):
                        if adj_matrix[j, k] == 1 and adj_matrix[k, i] == 1:
                            unfold_transition_tensor[i, (j * num_nodes + k)] = 1 / total_num
                            unfold_transition_tensor[i, (k * num_nodes + j)] = 1 / total_num
                            unfold_transition_tensor[j, (i * num_nodes + k)] = 1 / total_num
                            unfold_transition_tensor[j, (k * num_nodes + i)] = 1 / total_num
                            unfold_transition_tensor[k, (i * num_nodes + j)] = 1 / total_num
                            unfold_transition_tensor[k, (j * num_nodes + i)] = 1 / total_num
    return adj_matrix, deg_matrix, trans_matrix, unfold_transition_tensor.tocsr()


def hosploc(seed_node, adj_matrix, deg_matrix, trans_matrix, unfold_transition_tensor, upper_bound_phi):
    num_nodes = adj_matrix.shape[0]
    pagerank_1 = np.zeros((num_nodes,), dtype=np.float32)
    pagerank_1[seed_node] = 1

    # -------Parameter Set-------- #
    c1 = 200
    c4 = 140
    b = 3
    mu_v = np.sum(deg_matrix)
    l = math.ceil(math.log((mu_v / 2), 2))
    t_last = (l + 1) * math.ceil(
        (2 / math.pow(upper_bound_phi, 2)) * math.log(c1 * (l + 2) * math.sqrt(mu_v / 2), math.e))
    t_max = int(t_last)
    epsilon = 1 / (1800 * (l + 2) * t_last * math.pow(2, b))
    # --------------------------- #

    pagerank_2 = np.dot(trans_matrix, pagerank_1)
    pagerank_2 = pagerank_2 / np.sum(pagerank_2)

    # cold_start: spacey random walk iterates 10 times
    for iteration in range(10):
        pagerank = unfold_transition_tensor.dot((np.kron(pagerank_2, pagerank_1)))
        pagerank = pagerank / np.sum(pagerank)
        pagerank_1 = pagerank_2
        pagerank_2 = pagerank

    # start iteration
    for t in range(t_max):
        pagerank = unfold_transition_tensor.dot((np.kron(pagerank_2, pagerank_1)))
        pagerank = pagerank / np.sum(pagerank)
        pagerank_1 = pagerank_2
        pagerank_2 = pagerank

        score = pagerank / np.sum(deg_matrix, axis=0)
        pi = np.argsort(score, kind='heapsort')[::-1]

        next_cluster = pi[:3]
        next_volume = np.sum(deg_matrix[:, next_cluster])
        next_temp = np.sum(adj_matrix[next_cluster, :][:, next_cluster])
        if next_volume == 0 or next_volume == mu_v:
            next_phi = 1
        else:
            next_phi = (next_volume - next_temp) / min(next_volume, mu_v - next_volume)

        for j in range(3, num_nodes):
            cluster = next_cluster
            volume = next_volume
            phi = next_phi
            node_index = cluster[-1]
            I_x = pagerank_2[node_index] / deg_matrix[node_index, node_index]

            next_cluster = pi[:j+1]
            next_volume = np.sum(deg_matrix[:, next_cluster])
            next_temp = np.sum(adj_matrix[next_cluster, :][:, next_cluster])
            if next_volume == 0 or next_volume == mu_v:
                next_phi = 1
            else:
                next_phi = (next_volume - next_temp) / min(next_volume, mu_v - next_volume)

            if I_x >= epsilon / (c4 * (l + 2) * math.pow(2, b)):  # --- pass probability check --- #
                if math.pow(2, b) <= volume:  # --- pass volume check --- #
                    if phi <= upper_bound_phi:  # --- pass phi check --- #
                        if phi < next_phi:  # ---- local minimum --- #
                            return cluster, phi, pagerank  # --- return first satisfied local minimum --- #

    return [], None, None  # --- HOSPLOC algorithm did not find satisfied cluster under conditions c1-c3 --- #


if __name__ == '__main__':
    graph_name_list = ['Dataset/Alpha/alpha', 'Dataset/OTC/otc']
    for graph_name in graph_name_list:
        print('-----------------------------------------------------')
        print("Data set: " + graph_name.split('/')[1])

        print('---> Building corresponding matrices ...')
        adj_matrix, deg_matrix, trans_matrix, unfold_transition_tensor = build_matrices(graph_name)
        print('---> Building finished.')

        # --- save matrices --- #
        np.save(graph_name + '_t0_adj_mat', adj_matrix)
        sp.save_npz(graph_name + '_t0_trans_tens', unfold_transition_tensor)

        seed_node = 500
        upper_bound_phi = 0.5

        print('---> Start HOSPLOC clustering ...')
        cluster, phi, pagerank = hosploc(seed_node, adj_matrix, deg_matrix, trans_matrix, unfold_transition_tensor, upper_bound_phi)
        if len(cluster) == 0:
            print('---> HOSPLOC could not find a cluster under conductance ' + str(upper_bound_phi) + '. Please enlarge the upper bound.')
        else:
            print('---> HOSPLOC finds the local cluster whose conductance is ' + str(phi) + '.')
            np.save(graph_name + '_t0_pr_vec', pagerank)
            np.save(graph_name + '_t0_clus_vec', cluster)
