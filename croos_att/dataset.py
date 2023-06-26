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


def lobar_dataset(path, train=False, test=False):
    file = os.listdir(path)
    file.sort()
    num = len(file) // 5
    dataset = []

    for i in range(num):
        edge = np.load(os.path.join(path, file[i * 5]), allow_pickle=True)
        edge_prop = np.load(os.path.join(path, file[i * 5 + 1]), allow_pickle=True)
        x = np.load(os.path.join(path, file[i * 5 + 3]), allow_pickle=True)
        y = np.load(os.path.join(path, file[i * 5 + 4]), allow_pickle=True)
        patient = file[i * 5].split('.')[0][:-5]
        edge = edge[:, edge_prop > 0]
        weight = np.ones(y.shape)
        y_lobar = reduce_classes(y)

        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()

        x = (torch.from_numpy(x)).float()
        y = (torch.from_numpy(y_lobar)).float()
        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_prop, patient=patient, weights=weights)

        if x.shape[0] == y.shape[0]:
            dataset.append(data)
        else:
            print(file[i * 4])
    return dataset


def segmental_dataset(path, train=False, test=False):
    file = os.listdir(path)
    file.sort()
    num = len(file) // 5
    dataset = []

    pred_path = "/opt/Data1/yuweihao/tnn_v2/results/lobar_transformer_2layer_dim128_heads4_hdim32_mlp256_postnorm_adam1e-3_0800/"

    for i in range(num):
        edge = np.load(os.path.join(path, file[i * 5]), allow_pickle=True)
        edge_prop = np.load(os.path.join(path, file[i * 5 + 1]), allow_pickle=True)
        x = np.load(os.path.join(path, file[i * 5 + 3]), allow_pickle=True)
        y = np.load(os.path.join(path, file[i * 5 + 4]), allow_pickle=True)
        patient = file[i * 5].split('.')[0][:-5]
        edge = edge[:, edge_prop > 0]
        weight = np.ones(y.shape)

        if test:
            pred = np.load(os.path.join(pred_path, file[i * 5 + 2]), allow_pickle=True)
        else:
            pred = np.load(os.path.join(path, file[i * 5 + 2]), allow_pickle=True)

        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()

        pred111 = np.zeros((pred.shape[0], 3))
        pred111[:, 0] = pred // 4
        pred = pred - pred111[:, 0] * 4
        pred111[:, 1] = pred // 2
        pred = pred - pred111[:, 1] * 2
        pred111[:, 2] = pred // 1

        pred_ = pred.copy()
        mask = np.zeros((y.shape[0], y.shape[0]), dtype=np.uint8)
        for ii in range(6):
            seg_label = (pred_ == ii).astype(np.uint8)
            seg_index = np.where(seg_label == 1)[0]
            mask[seg_index] = seg_label
        masks = torch.from_numpy(mask).float()

        # x = np.concatenate([x[:,0:20], pred111], axis=1)
        x = np.concatenate([x, pred111], axis=1)

        x = (torch.from_numpy(x)).float()
        y = (torch.from_numpy(y)).float()
        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_prop, patient=patient, weights=weights, mask=masks)

        if x.shape[0] == y.shape[0]:
            dataset.append(data)
        else:
            print(file[i * 4])
    return dataset


def subsegmental_dataset(path, train=False, test=False):
    file = os.listdir(path)
    file.sort()
    num = len(file) // 5
    dataset = []

    pred_path = "/opt/Data1/yuweihao/hypergraph/results/seg_hyper_datav3_20_arg_attagg2_binary_randnum_800_0700_lobarcos/"

    for i in range(num):
        edge = np.load(os.path.join(path, file[i * 5]), fix_imports=True, encoding='latin1')
        edge_prop = np.load(os.path.join(path, file[i * 5 + 1]), fix_imports=True, encoding='latin1')
        x = np.load(os.path.join(path, file[i * 5 + 3]), fix_imports=True, encoding='latin1')
        y = np.load(os.path.join(path, file[i * 5 + 4]), fix_imports=True, encoding='latin1')
        patient = file[i * 5].split('.')[0][:-5]
        edge = edge[:, edge_prop > 0]
        weight = np.ones(y.shape)

        if test:
            pred = np.load(os.path.join(pred_path, file[i * 5 + 2]), allow_pickle=True, fix_imports=True,
                           encoding='latin1')
        else:
            pred = np.load(os.path.join(path, file[i * 5 + 2]), allow_pickle=True, fix_imports=True, encoding='latin1')

        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()

        pred111 = np.zeros((pred.shape[0], 5))
        pred111[:, 0] = pred // 16
        pred = pred - pred111[:, 1] * 16
        pred111[:, 1] = pred // 8
        pred = pred - pred111[:, 1] * 8
        pred111[:, 2] = pred // 4
        pred = pred - pred111[:, 1] * 4
        pred111[:, 3] = pred // 2
        pred = pred - pred111[:, 1] * 2
        pred111[:, 4] = pred // 1

        pred_ = pred.copy()
        pred_[pred_ >= 18] = 18
        mask = np.zeros((y.shape[0], y.shape[0]), dtype=np.uint8)
        for ii in range(19):
            seg_label = (pred_ == ii).astype(np.uint8)
            seg_index = np.where(seg_label == 1)[0]
            mask[seg_index] = seg_label
        masks = torch.from_numpy(mask).float()

        x = np.concatenate([x, pred111], axis=1)

        x = (torch.from_numpy(x)).float()
        y = (torch.from_numpy(y)).float()
        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_prop, patient=patient, weights=weights, mask=masks)

        if x.shape[0] == y.shape[0]:
            dataset.append(data)
        else:
            print(file[i * 4])
    return dataset


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


def multitask_dataset(path2, path3,path_spd, train=False, test=False):
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

        if spd.max()>max:
            max = spd.max()


        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()
        #spd = (torch.from_numpy(spd))
        x = (torch.from_numpy(x)).float()
        y_subseg = (torch.from_numpy(y_subseg)).float()
        y_seg = (torch.from_numpy(y_seg)).float()
        y_lobar = (torch.from_numpy(y_lobar)).float()

        data = Data(x=x, edge_index=edge_index, y_lobar=y_lobar, y_seg=y_seg, y_subseg=y_subseg, edge_attr=edge_prop,
                    patient=patient, weights=weights,spd = spd)

        if x.shape[0] == y_subseg.shape[0]:
            dataset.append(data)
        else:
            print(file[i * 4])
    return dataset

def multitask_hg(path2, path3,train=False, test=False):
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
        trachea = loc_trachea(x)

        parent_map, children_map = parent_children_map(edge, x.shape[0])
        hypertree = hyper_tree(parent_map, children_map)
        # hypergraph = hypergraph_airwaytree(hypertree, children_map, pred, num=15)
        hypergraph = hypergraph_airwaytree_full(hypertree, children_map, trachea, num=40)
        A_hg = hyper_A(hypergraph)
        A_hg = (torch.from_numpy(A_hg)).long()

        dict_gen = generation_dict(x)
        dict_gen = (torch.from_numpy(dict_gen)).long()


        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()

        x = (torch.from_numpy(x)).float()
        y_subseg = (torch.from_numpy(y_subseg)).float()
        y_seg = (torch.from_numpy(y_seg)).float()
        y_lobar = (torch.from_numpy(y_lobar)).float()

        data = Data(x=x, edge_index=edge_index, y_lobar=y_lobar, y_seg=y_seg, y_subseg=y_subseg, edge_attr=edge_prop,
                    patient=patient, weights=weights,A_hg = A_hg,dict = dict_gen)

        if x.shape[0] == y_subseg.shape[0]:
            dataset.append(data)
        else:
            print(file[i * 4])
    return dataset



def loc_trachea(x):
    idx = np.argmax(x[:,13])
    return idx


def hypergraph_airwaytree_full(hypertree, children_map, trachea, num=10):
    hypertree = hypertree + np.eye(children_map.shape[0])
    hypergraph = []
    clique_lobar = children_map[trachea, :]
    hypergraph.append(clique_lobar)
    lobar_nodes = np.where(children_map[trachea, :] == 1)[0]
    start_nodes = Queue()
    for i in range(len(lobar_nodes)):
        children = np.where(children_map[lobar_nodes[i], :] == 1)[0]
        if children is not None:
            for child in children:
                if child not in lobar_nodes:
                    start_nodes.put(child)
                    hypertree[child, lobar_nodes[i]] = 1

    while (not start_nodes.empty()):
        cur = start_nodes.get()
        hypergraph.append(hypertree[cur, :])
        children = np.where(children_map[cur, :] == 1)[0]
        if children is not None:
            for child in children:
                hypertree[child, cur] = 1
                hypergraph.append(hypertree[child, :])
                num_children = np.sum(hypertree[child, :])
                if num_children > num:
                    start_nodes.put(child)

    hypergraph = np.array(hypergraph)
    return hypergraph

def prepare_dataset_subsegmental_hg_pred(path1, path2, node_weight=1, train=False,
                                         test=False):  # path1为seg，path2为subseg
    file = os.listdir(path2)
    file.sort()
    num = len(file) // 5
    dataset = []
    max = 0

    pred_path = ""

    for i in range(num):
        edge = np.load(os.path.join(path2, file[i * 5]))
        edge_prop = np.load(os.path.join(path2, file[i * 5 + 1]))
        x = np.load(os.path.join(path2, file[i * 5 + 3]))
        y_subseg = np.load(os.path.join(path2, file[i * 5 + 4]))
        patient = file[i * 5].split('.')[0][:-5]
        edge = edge[:, edge_prop > 0]

        y_seg = np.load(os.path.join(path1, file[i * 5 + 4]), allow_pickle=True)
        y_lobar = reduce_classes(y_seg)

        '''if test:
            pred = np.load(os.path.join(pred_path, file[i * 5 + 2]))
        else:
            pred = np.load(os.path.join(path2, file[i * 5 + 2]))'''
        pred = np.load(os.path.join(path2, file[i * 5 + 2]))

        parse_num = 20
        weight = np.ones(y_subseg.shape)

        parent_map, children_map = parent_children_map(edge, x.shape[
            0])  # parent_map N维，每个node的parents，children_map，N*N,a[i,j]=1说明j是i的child
        hypertree = hyper_tree(parent_map, children_map)
        hypergraph = hypergraph_airwaytree_sub(hypertree, children_map, pred, num=parse_num)
        edge_hg = hypergraph_edge_detection(hypergraph)
        A_hg = hyper_A(hypergraph)
        if max < np.max(A_hg):
            max  = np.max(A_hg)
        hypergraph = (torch.from_numpy(hypergraph)).float()
        A_hg = (torch.from_numpy(A_hg)).long()
        edge_hg = (torch.from_numpy(edge_hg)).long()

        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weight = (torch.from_numpy(weight)).float()

        '''pred111 = np.zeros((pred.shape[0], 5))
        pred111[:, 0] = pred // 16
        pred = pred - pred111[:, 1] * 16
        pred111[:, 1] = pred // 8
        pred = pred - pred111[:, 1] * 8
        pred111[:, 2] = pred // 4
        pred = pred - pred111[:, 1] * 4
        pred111[:, 3] = pred // 2
        pred = pred - pred111[:, 1] * 2
        pred111[:, 4] = pred // 1

        x = np.concatenate([x, pred111], axis=1)'''

        x = (torch.from_numpy(x)).float()
        y_subseg = (torch.from_numpy(y_subseg)).float()
        y_seg = (torch.from_numpy(y_seg)).float()
        y_lobar = (torch.from_numpy(y_lobar)).float()
        data = Data(x=x, edge_index=edge_index, y_lobar=y_lobar, y_seg=y_seg, y_subseg=y_subseg, edge_attr=edge_prop,
                    patient=patient, weights=weight,
                    hypergraph=hypergraph, edge_hg=edge_hg, A_hg=A_hg)

        edge_node_he = []
        node_num = x.shape[0]
        index = np.arange(0, node_num)
        loc = np.where(hypergraph.data.numpy() == 1)
        for m in range(len(loc[0])):
            edge_node_he.append((loc[0][m] + node_num, loc[1][m]))
        edge_node_he = np.array(edge_node_he)
        index = np.arange(0, node_num)
        edge_node_he = np.transpose(edge_node_he, (1, 0))
        edge_node_he_index = (torch.from_numpy(edge_node_he)).long()
        data.__setattr__("edge_node_he", edge_node_he_index)
        data.__setattr__("node_num", node_num)
        index = (torch.from_numpy(index)).long()
        data.__setattr__("index", index)

        if x.shape[0] == y_subseg.shape[0]:
            dataset.append(data)
        else:
            print(file[i * 4])
    print("max",max)
    return dataset


def hypergraph_airwaytree_sub(hypertree, children_map, y, num=10):
    hypertree = hypertree + np.eye(y.shape[0])
    y_lobar = y  # 感觉应该是y_seg
    hypergraph = []
    clique_lobar = (y_lobar >= 18).astype(np.uint8)
    hypergraph.append(clique_lobar)
    lobar_nodes = np.where(y_lobar >= 18)[0]  # 》=18即为主干和主支气管的集合
    start_nodes = Queue()
    for i in range(len(lobar_nodes)):
        children = np.where(children_map[lobar_nodes[i], :] == 1)[0]
        if children is not None:
            for child in children:
                if child not in lobar_nodes:
                    start_nodes.put(child)
                    hypertree[child, lobar_nodes[i]] = 1

    while (not start_nodes.empty()):
        cur = start_nodes.get()
        hypergraph.append(hypertree[cur, :])
        children = np.where(children_map[cur, :] == 1)[0]
        if children is not None:
            for child in children:
                hypertree[child, cur] = 1
                hypergraph.append(hypertree[child, :])
                num_children = np.sum(hypertree[child, :])
                if num_children > num:
                    start_nodes.put(child)

    hypergraph = np.array(hypergraph)
    return hypergraph


def data_augmentation_pred_sub(pred, weight, edge, node_weight):
    main_list = []
    child_list = []
    for i in range(edge.shape[1]):
        if pred[edge[0, i]] != pred[edge[1, i]] and pred[edge[0, i]] >= 18:  # 起点是主支气管的edge
            main_list.append(edge[0, i])
            child_list.append(edge[1, i])
    for i in range(len(main_list)):
        mode = np.random.randint(10)
        if mode < 6:
            weight[main_list[i]] = node_weight
            weight[child_list[i]] = node_weight
        elif mode < 8:
            pred[main_list[i]] = pred[child_list[i]]
            weight[main_list[i]] = node_weight
        else:
            if pred[main_list[i]] >= 18:
                pred[child_list[i]] = 18
                weight[child_list[i]] = node_weight
    return pred, weight


def parent_children_map(edge, N):
    parent_map = np.zeros(N, dtype=np.uint16)
    children_map = np.zeros((N, N), dtype=np.uint8)
    for i in range(edge.shape[1]):
        parent_map[edge[1, i]] = edge[0, i]
        children_map[edge[0, i], edge[1, i]] = 1
    return parent_map, children_map


def hyper_tree(parent_map, children_map):  # hypertree[i,:]处于同一条超边中，且i是超边中最顶上的那个节点
    hypertree = children_map.copy()
    children_map_copy = children_map.copy()
    N = len(parent_map)
    ends = np.zeros(N)
    while (children_map_copy.sum() != 0):  # 若=0，说明没有不归为超边的边了
        cur_children = np.sum(children_map_copy, axis=1)
        for i in range(N):
            if ends[i] == 0 and cur_children[i] == 0:  # 选超边意义上的叶节点（超边的最顶点）
                hypertree[parent_map[i], :] += hypertree[i, :]  # 并入父节点
                children_map_copy[parent_map[i], i] = 0  # 消除点原超边与父节点直接的edge
                ends[i] = 1  # 已经归为超边中，不可再用
    return hypertree


def hypergraph_edge_detection(hypergraph):
    node_num = hypergraph.shape[0]
    edge = []
    for i in range(1, node_num - 1):
        for j in range(i + 1, node_num):
            if (hypergraph[i, :] * hypergraph[j, :]).sum() == 1:
                edge.append([i, j])
                edge.append([j, i])
    edge = np.array(edge)
    if len(edge) > 0:
        edge = np.transpose(edge, (1, 0))
    else:
        edge = np.array([[0], [0]])
    return edge


def hyper_A(hypergraph):  # A[i,j]表示
    node_num = hypergraph.shape[1]
    hyper_A = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            hyper_A[i, j] = (hypergraph[:, i] * hypergraph[:, j]).sum()
            hyper_A[j, i] = hyper_A[i, j]
    return hyper_A

def generation_dict(x):
    node_num = x.shape[0]
    dict = np.zeros((node_num,node_num))
    for i in range(node_num):
        for j in range(node_num):
            dict[i][j] = abs(x[i,0]-x[j,0])
    return dict







