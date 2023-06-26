import os
import numpy as np
import torch
path = "/home/yuy/code/data/graph_data_n_third_level_v3_train/"
save_path = "/home/yuy/code/data/spd/"
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

def floyd(edge_index):
    node_num = np.max(edge_index) + 1
    D = np.full((node_num,node_num), np.inf)
    P = np.zeros((node_num,node_num))
    for j in range(node_num):
        P[:,j] = j
    for i in range(node_num):
        D[i,i] = 0
    for idx in range(edge_index.shape[1]):
        D[edge_index[0][idx]][edge_index[1][idx]] = 1
        D[edge_index[1][idx]][edge_index[0][idx]] = 1
    # print(adjacent_matrix)
    for k in range(node_num):
        for i in range(node_num):
            for j in range(node_num):
                if D[i][j]>D[i][k]+D[k][j]:
                    D[i][j]=D[i][k]+D[k][j]
                    P[i][j] = P[i][k]

    return D,P

edge = np.load(path +  "2586574_edge.npy",allow_pickle= True)
N = np.max(edge) + 1
spd, path_spd = floyd(edge)
np.save(save_path+"2586574_spd.npy",spd)
np.save(save_path+ "2586574_path.npy", path_spd)
'''for i in range(num):
    print(i)
    patient = file[i * 5].split('.')[0][:-5]
    edge = np.load(path + patient + "_edge.npy",allow_pickle= True)
    N = np.max(edge) + 1
    spd, path_spd = floyd(edge)
    np.save(save_path+patient+"_spd.npy",spd)
    np.save(save_path + patient + "_path.npy", path_spd)'''





