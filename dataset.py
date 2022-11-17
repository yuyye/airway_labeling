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

def lobar_dataset(path,train=False,test=False):
    file = os.listdir(path)
    file.sort()
    num = len(file)//5
    dataset = []
   
    for i in range(num):
        edge = np.load(os.path.join(path, file[i*5]),allow_pickle=True)
        edge_prop = np.load(os.path.join(path, file[i*5+1]),allow_pickle=True)
        x = np.load(os.path.join(path, file[i*5+3]),allow_pickle=True)
        y = np.load(os.path.join(path, file[i*5+4]),allow_pickle=True)
        patient = file[i*5].split('.')[0][:-5]
        edge = edge[:,edge_prop>0]
        weight = np.ones(y.shape)
        y_lobar = reduce_classes(y)
        
        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()
        
        x = (torch.from_numpy(x)).float()
        y = (torch.from_numpy(y_lobar)).float()
        data = Data(x = x, edge_index = edge_index, y = y, edge_attr=edge_prop, patient=patient, weights=weights)
        
        if x.shape[0]==y.shape[0]:
            dataset.append(data)
        else:
            print(file[i*4])
    return dataset

def segmental_dataset(path,train=False,test=False):
    file = os.listdir(path)
    file.sort()
    num = len(file)//5
    dataset = []
    
    pred_path = "/opt/Data1/yuweihao/tnn_v2/results/lobar_transformer_2layer_dim128_heads4_hdim32_mlp256_postnorm_adam1e-3_0800/"
    
    for i in range(num):
        edge = np.load(os.path.join(path, file[i*5]),allow_pickle=True)
        edge_prop = np.load(os.path.join(path, file[i*5+1]),allow_pickle=True)
        x = np.load(os.path.join(path, file[i*5+3]),allow_pickle=True)
        y = np.load(os.path.join(path, file[i*5+4]),allow_pickle=True)
        patient = file[i*5].split('.')[0][:-5]
        edge = edge[:,edge_prop>0]
        weight = np.ones(y.shape)
        
        if test:
            pred = np.load(os.path.join(pred_path, file[i*5+2]),allow_pickle=True)
        else:
            pred = np.load(os.path.join(path, file[i*5+2]),allow_pickle=True)
        
        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()
        
        pred111 = np.zeros((pred.shape[0],3))
        pred111[:,0] = pred//4
        pred = pred-pred111[:,0]*4
        pred111[:,1] = pred//2
        pred = pred-pred111[:,1]*2
        pred111[:,2] = pred//1
        
        pred_ = pred.copy()
        mask = np.zeros((y.shape[0],y.shape[0]),dtype=np.uint8)
        for ii in range(6):
            seg_label = (pred_==ii).astype(np.uint8)
            seg_index = np.where(seg_label==1)[0]
            mask[seg_index] = seg_label
        masks = torch.from_numpy(mask).float()
        
        # x = np.concatenate([x[:,0:20], pred111], axis=1)
        x = np.concatenate([x, pred111], axis=1)
        
        x = (torch.from_numpy(x)).float()
        y = (torch.from_numpy(y)).float()
        data = Data(x = x, edge_index = edge_index, y = y, edge_attr=edge_prop, patient=patient, weights=weights, mask=masks)
        
        if x.shape[0]==y.shape[0]:
            dataset.append(data)
        else:
            print(file[i*4])
    return dataset

def subsegmental_dataset(path,train=False,test=False):
    file = os.listdir(path)
    file.sort()
    num = len(file)//5
    dataset = []
    
    pred_path = "/opt/Data1/yuweihao/hypergraph/results/seg_hyper_datav3_20_arg_attagg2_binary_randnum_800_0700_lobarcos/"
    
    for i in range(num):
        edge = np.load(os.path.join(path, file[i*5]))
        edge_prop = np.load(os.path.join(path, file[i*5+1]))
        x = np.load(os.path.join(path, file[i*5+3]))
        y = np.load(os.path.join(path, file[i*5+4]))
        patient = file[i*5].split('.')[0][:-5]
        edge = edge[:,edge_prop>0]
        weight = np.ones(y.shape)
        
        if test:
            pred = np.load(os.path.join(pred_path, file[i*5+2]))
        else:
            pred = np.load(os.path.join(path, file[i*5+2]))
        
        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()
        
        pred111 = np.zeros((pred.shape[0],5))
        pred111[:,0] = pred//16
        pred = pred-pred111[:,1]*16
        pred111[:,1] = pred//8
        pred = pred-pred111[:,1]*8
        pred111[:,2] = pred//4
        pred = pred-pred111[:,1]*4
        pred111[:,3] = pred//2
        pred = pred-pred111[:,1]*2
        pred111[:,4] = pred//1
        
        pred_ = pred.copy()
        pred_[pred_>=18] = 18
        mask = np.zeros((y.shape[0],y.shape[0]),dtype=np.uint8)
        for ii in range(19):
            seg_label = (pred_==ii).astype(np.uint8)
            seg_index = np.where(seg_label==1)[0]
            mask[seg_index] = seg_label
        masks = torch.from_numpy(mask).float()
        
        x = np.concatenate([x, pred111], axis=1)
        
        x = (torch.from_numpy(x)).float()
        y = (torch.from_numpy(y)).float()
        data = Data(x = x, edge_index = edge_index, y = y, edge_attr=edge_prop, patient=patient, weights=weights, mask=masks)
        
        if x.shape[0]==y.shape[0]:
            dataset.append(data)
        else:
            print(file[i*4])
    return dataset

def reduce_classes(y0):
    y = y0.copy()
    for j in range(len(y)):
        if y[j]<=3:
            y[j] = 1
        elif (y[j]>3)&(y[j]<=7):
            y[j] = 2
        elif (y[j]>7)&(y[j]<=10):
            y[j] = 3
        elif (y[j]>10)&(y[j]<=12):
            y[j] = 4
        elif (y[j]>12)&(y[j]<=17):
            y[j] = 5
        else:
            y[j] = 0
    return y

def multitask_dataset(path2, path3, train=False,test=False):
    file = os.listdir(path3)
    file.sort()
    num = len(file)//5
    dataset = []
    
    for i in range(num):
        edge = np.load(os.path.join(path3, file[i*5]),allow_pickle=True)
        edge_prop = np.load(os.path.join(path3, file[i*5+1]),allow_pickle=True)
        x = np.load(os.path.join(path3, file[i*5+3]),allow_pickle=True)
        y_subseg = np.load(os.path.join(path3, file[i*5+4]),allow_pickle=True)
        patient = file[i*5].split('.')[0][:-5]
        edge = edge[:,edge_prop>0]
        weight = np.ones(y_subseg.shape)
        y_seg = np.load(os.path.join(path2, file[i*5+4]),allow_pickle=True)
        y_lobar = reduce_classes(y_seg)
        
        edge_prop = (torch.from_numpy(edge_prop)).float()
        edge_index = (torch.from_numpy(edge)).long()
        weights = (torch.from_numpy(weight)).float()
        
        x = (torch.from_numpy(x)).float()
        y_subseg = (torch.from_numpy(y_subseg)).float()
        y_seg = (torch.from_numpy(y_seg)).float()
        y_lobar = (torch.from_numpy(y_lobar)).float()
        data = Data(x = x, edge_index = edge_index, y_lobar = y_lobar, y_seg=y_seg,y_subseg=y_subseg, edge_attr=edge_prop, patient=patient, weights=weights)
        
        if x.shape[0]==y_subseg.shape[0]:
            dataset.append(data)
        else:
            print(file[i*4])
    return dataset