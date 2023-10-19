# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:54:00 2022

@author: Yu
"""

import torch
from torch_geometric.data import Data, DataLoader
import torch_geometric.nn as pyg_nn
import os
import numpy as np
from queue import Queue
from torch_geometric.utils import degree
import copy

def reduce_classes(y0):
    y = y0.copy()
    for j in range(len(y)):
        if y[j] <= 3:
            y[j] = 1
        elif (y[j] > 3) & (y[j] <= 7):
            y[j] = 2
        elif (y[j] > 7) & (y[j] <= 10):
            y[j] = 3
        elif (y[j] > 10) & (y[j] <= 12):
            y[j] = 4
        elif (y[j] > 12) & (y[j] <= 17):
            y[j] = 5
        else:
            y[j] = 0
    return y

def to_adj(edge_index):
    node_num = torch.max(edge_index)+1
    adj = torch.zeros(node_num,node_num,device=edge_index.device)
    for idx in range(edge_index.shape[1]):
        adj[edge_index[0][idx]][edge_index[1][idx]] = 1
        adj[edge_index[1][idx]][edge_index[0][idx]] = 1


    adj = adj + torch.eye(node_num,device=edge_index.device)

    return adj

def to_degree(adj):
    node_num = adj.shape[0]
    degree = torch.zeros(node_num, node_num, device=adj.device)
    for idx in range(node_num):
        degree[idx][idx] = pow(torch.sum(adj[idx]),-0.5)
    return degree

def to_degree_dir(adj):
    node_num = adj.shape[0]
    #adj = adj - torch.eye(node_num, device=adj.device)
    degree = torch.zeros(node_num, node_num, device=adj.device)
    for idx in range(node_num):
        degree[idx][idx] = pow(torch.sum(adj[idx]) + torch.sum(adj[:][idx]),-1)
    #degree = degree - torch.eye(node_num, device=degree.device)
    return degree

def to_Anorm(edge_index):
    node_num = np.max(edge_index) + 1
    adj = np.zeros((node_num, node_num))
    for idx in range(edge_index.shape[1]):
        adj[edge_index[0][idx]][edge_index[1][idx]] = 1
        adj[edge_index[1][idx]][edge_index[0][idx]] = 1

    adj = adj + np.eye(node_num)

    degree = np.zeros((node_num, node_num))
    for idx in range(node_num):
        degree[idx][idx] = pow(np.sum(adj[idx]), -0.5)

    Tensor_l_DotProd = np.matmul(degree, adj)
    Anorm = np.matmul(Tensor_l_DotProd, degree)
    return Anorm

def multitask_dataset_atm(path3,path_spd,path_lca, train=False, test=False):
    file = os.listdir(path3)
    file.sort()
    num = len(file)//3
    dataset = []
    max = 0

    for i in range(num):
        patient = file[i*3][0:3]
        if patient == "034":
            continue
        x = np.load(path3 + patient +"_x.npy", allow_pickle=True)
        edge = np.load(path3 + patient + "_edge.npy", allow_pickle=True)
        edge_prop = np.load(path3 + patient + "_edge_feature.npy", allow_pickle=True)
        edge = edge[:, edge_prop > 0]

        A_norm = to_Anorm(edge)
        A_norm = torch.from_numpy(A_norm).float()

        spd = np.array(np.load(path_spd+patient+"_spd.npy"))
        spd = np.array(spd)
        spd = np.where(spd > 29, 29, spd)
        spd = torch.from_numpy(spd).long()


        lca = np.array(np.load(path_lca + patient + "_lca.npy"))

        lca_d = get_lca_d(lca,x[:,0])
        for i in range(lca_d.shape[0]):
            for j in range(lca_d.shape[1]):
                lca_d[i,j] = lca_d[i,j] / x[i,0] * 29
        lca_d = torch.from_numpy(lca_d).long()

        gen = torch.from_numpy(generation_dict(x)).long()
        x = (torch.from_numpy(x)).float()

        data = Data(x=x,patient = patient,spd=spd,gen=gen,lca = lca_d,A_norm = A_norm)
        dataset.append(data)

    return dataset



def multitask_dataset(path2, path3,path_spd,path_lca, train=False, test=False):
    file = os.listdir(path3)
    file.sort()
    num = len(file) // 5
    dataset = []
    max = 0

    for i in range(num):
        edge = np.load(os.path.join(path3, file[i * 5]), allow_pickle=True)
        edge_prop = np.load(os.path.join(path3, file[i * 5 + 1]), allow_pickle=True)
        x = np.load(os.path.join(path3, file[i * 5 + 3]), allow_pickle=True)
        y_subseg = np.load(os.path.join(path3, file[i * 5 + 4]), allow_pickle=True)
        patient = file[i * 5].split('.')[0][:-5]
        edge = edge[:, edge_prop > 0]
        weight = np.ones(y_subseg.shape)
        y_seg = np.load(os.path.join(path2, file[i * 5 + 4]), allow_pickle=True)
        y_lobar = reduce_classes(y_seg)
        spd = np.array(np.load(path_spd+patient+"_spd.npy"))
        spd = np.array(spd)
        spd = np.where(spd > 29, 29, spd)
        spd = torch.from_numpy(spd).long()

        lca = np.array(np.load(path_lca + patient + "_lca.npy"))
        lca_d = get_lca_d(lca, x[:, 0])
        for i in range(lca_d.shape[0]):
            for j in range(lca_d.shape[1]):
                lca_d[i, j] = lca_d[i, j] / x[i, 0] * 29
        lca_d = torch.from_numpy(lca_d).long()

        gen = torch.from_numpy(generation_dict(x)).long()
        A_norm = to_Anorm(edge)
        A_norm = torch.from_numpy(A_norm).float()

        #lca_d += gen


        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()
        #spd = (torch.from_numpy(spd))
        x = (torch.from_numpy(x)).float()
        y_subseg = (torch.from_numpy(y_subseg)).float()
        y_seg = (torch.from_numpy(y_seg)).float()
        y_lobar = (torch.from_numpy(y_lobar)).float()

        data = Data(x=x[:,0:20], edge_index=edge_index, y_lobar=y_lobar, y_seg=y_seg, y_subseg=y_subseg, edge_attr=edge_prop,
                    patient=patient, weights=weights,spd=spd,gen=gen,lca = lca_d,A_norm = A_norm)

        if x.shape[0] == y_subseg.shape[0]:
            dataset.append(data)
        else:
            print(file[i * 4])
    return dataset


def generation_dict(x):
    node_num = x.shape[0]
    dict = np.zeros((node_num,node_num))
    for i in range(node_num):
        for j in range(node_num):
            dict[i][j] = abs(x[i,0]-x[j,0])
    return dict

def get_lca_d(lca,depth):
    lca_d = np.zeros_like(lca)
    for i in range(lca_d.shape[0]):
        for j in range(lca_d.shape[1]):
            lca_d[i,j] = depth[i] - depth[lca[i,j]]
    return lca_d


