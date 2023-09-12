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


def multitask_dataset(path2, path3,path_spd,path_dir, train=False, test=False):
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

        '''dir = np.array(np.load(path_dir + patient + "_spd_direct.npy"))
        dir = np.array(dir)
        dir = np.where(dir > 29, 29, dir)'''
        dir = np.array(np.load(path_dir + patient + "_spd_directZ.npy"))
        dir = np.array(dir) + 15
        dir = np.where(dir > 29, 29, dir)

        dir = torch.from_numpy(dir).long()



        gen = torch.from_numpy(generation_dict(x)).long()


        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()
        #spd = (torch.from_numpy(spd))
        x = (torch.from_numpy(x)).float()
        y_subseg = (torch.from_numpy(y_subseg)).float()
        y_seg = (torch.from_numpy(y_seg)).float()
        y_lobar = (torch.from_numpy(y_lobar)).float()

        data = Data(x=x, edge_index=edge_index, y_lobar=y_lobar, y_seg=y_seg, y_subseg=y_subseg, edge_attr=edge_prop,
                    patient=patient, weights=weights,spd=spd,gen=gen)

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







