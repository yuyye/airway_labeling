import os
import numpy as np
import torch
path = "/home/yuy/code/data/graph_data_n_third_level_v3_test_pred/"
save_path = "/home/yuy/code/transformer/Spatial Encoding/spd_test_new/"
file = os.listdir(path)
file.sort()
num = len(file) // 5
def floyd(edge_index):
    node_num = np.max(edge_index) + 1
    adj = np.full((node_num,node_num), np.inf)
    for i in range(node_num):
        adj[i,i] = 0
    for idx in range(edge_index.shape[1]):
        adj[edge_index[0][idx]][edge_index[1][idx]] = 1
        adj[edge_index[1][idx]][edge_index[0][idx]] = 1
    a = adj.copy()
    # print(adjacent_matrix)
    for k in range(node_num):
        for i in range(node_num):
            for j in range(node_num):
                if a[i][j]>a[i][k]+a[k][j]:
                    a[i][j]=a[i][k]+a[k][j]
    return a


for i in range(num):
    print(i)
    edge = np.load(os.path.join(path, file[i * 5]), allow_pickle=True)
    patient = file[i * 5].split('.')[0][:-5]
    N = np.max(edge) + 1
    adj = torch.zeros([N, N], dtype=torch.bool)
    for idx in range(edge.shape[1]):
        adj[edge[0][idx]][edge[1][idx]] = True
        adj[edge[1][idx]][edge[0][idx]] = True
    spd, path = algos.floyd_warshall(adj.numpy())
    np.save(save_path+patient+"_spd.npy",spd)
