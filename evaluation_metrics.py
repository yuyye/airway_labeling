import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pickle
import random

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
import os
import numpy as np
import sys
#sys.path.append("att_merge_new/")
#print(sys.path)
from airformer_design.transformer import AirwayFormer_se

seed = 333
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
import time
import shutil
import sys
import pickle
import matplotlib.pyplot as plt


from att_merge_new.dataset import multitask_dataset
from loss_functions import LabelSmoothCrossEntropyLoss, DependenceLoss
from utils import *


def seg2lobor(y0):
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


def subseg2seg(y0, trachea):
    y = y0.copy()
    for j in range(len(y)):
        if y[j] == 0:
            y[j] = 18
        else:
            y[j] = (y[j] - 1) // 7
    y[trachea] = 19
    return y


def subseg2seg(y0, trachea):
    book = np.zeros(127)
    book[0] = 18

    y = y0.copy()
    for j in range(len(y)):
        if y[j] == 0:
            y[j] = 18
        else:
            y[j] = (y[j] - 1) // 7
    y[trachea] = 19
    return y


def loc_trachea(x):
    idx = np.argmax(x[:, 13])
    return idx


def consistence(pred1, pred2, lable):
    index1 = np.arange(0, pred1.shape[0])
    index2 = np.arange(0, pred2.shape[0])
    error1 = index1[pred1 != lable]
    error2 = index2[pred2 != lable]
    mask = np.in1d(error1, error2, invert=True)
    ratio = np.sum(mask.astype(np.uint8)) / pred1.shape[0]
    return ratio
def evaluation(label,pred):
    class_num = label.max()+1
    '''true = [pred[i] == label[i] for i in range(pred.shape[0])]
    true = np.array(true).astype(np.uint8)
    false = [pred[i] != label[i] for i in range(pred.shape[0])]
    false = np.array(false).astype(np.uint8)'''
    RC = []
    PR = []
    F1 = []
    for class_i in range(class_num):
        positive = [label_i == class_i for label_i in label]
        positive = np.array([positive]).astype(np.uint8)
        if np.sum(positive) ==0:
            continue
        negative = [label_i != class_i for label_i in label]
        negative = np.array(negative).astype(np.uint8)

        true =[pred_i == class_i for pred_i in pred]
        true = np.array([true]).astype(np.uint8)
        false = [pred_i != class_i for pred_i in pred]
        false = np.array([false]).astype(np.uint8)


        TP = np.sum(positive * true)
        FP = np.sum(true * negative)
        FN = np.sum(positive * false)
        RC_i = TP / (TP + FN)

        if np.sum(true) == 0:
            PR_i = 0
            F1_i = 0
            RC.append(RC_i)
            PR.append(PR_i)
            F1.append(F1_i)
            continue

        PR_i = TP /(TP + FP)
        if(RC_i+PR_i ==0):
            F1_i = 0
            RC.append(RC_i)
            PR.append(PR_i)
            F1.append(F1_i)
            continue
        F1_i = (2*RC_i*PR_i)/(RC_i+PR_i)
        RC.append(RC_i)
        PR.append(PR_i)
        F1.append(F1_i)

    PR = np.array(PR)
    RC = np.array(RC)
    F1 =np.array(F1)
    RC_mean = np.mean(RC)
    PR_mean = np.mean(PR)
    F1_mean = np.mean(F1)
    return RC_mean,PR_mean,F1_mean

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



def to_Anorm(A_hat,D_hat):
    Tensor_l_DotProd = torch.matmul(D_hat, A_hat)
    Anorm = torch.matmul(Tensor_l_DotProd, D_hat)
    return Anorm


train_path2 = "/home/yuy/code/data/graph_ht_pred_train_v6_v3/"
test_path2 = "/home/yuy/code/data/graph_ht_pred_test_v6_v3/"
train_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_train/"
test_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_test_pred/"
spd_train = "/home/yuy/code/transformer/Spatial Encoding/spd_train/"
spd_test = "/home/yuy/code/transformer/Spatial Encoding/spd_test/"
epochs = 800
dataset1 = multitask_dataset(train_path2, train_path3, spd_train,train=True)
#dataset1 = multitask_hg(train_path2, train_path3,train=True)
train_loader_case = DataLoader(dataset1, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)


dataset2 = multitask_dataset(test_path2, test_path3, spd_test,test=True)
#dataset2 = multitask_hg(test_path2, test_path3,test=True)
test_loader_case = DataLoader(dataset2, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

max_acc = 0
# torch.set_default_dtype(torch.float64)


seed = [333,444,555,666]
Acc = []
RC = []
PR = []
F1 = []
alpha = 0.4
for i in range(4):
    #name = "/home/yuy/code/transformer/Spatial Encoding/checkpoints/att_2_se_except_soft0.1_dense_headmask_0.1_1_1_1_1_1_seed{}_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt".format(seed[i])
    #my_net = AirwayFormer_att_se(input_dim=23, num_classes1=6, num_classes2=20, num_classes3=127, dim=128, heads=4,
                              #mlp_dim=256, dim_head=32, dropout = 0., emb_dropout=0.,alpha = 0.1)
    name = "/home/yuy/code/transformer/airformer_design/checkpoints/baseline_spd_seed{}/best.ckpt".format(seed[i])
    my_net = AirwayFormer_se(input_dim=23, num_classes1=6, num_classes2=20, num_classes3=127, dim=128, heads=4,
                             mlp_dim=256, dim_head=32, dropout=0., emb_dropout=0.)
    checkpoint = torch.load(name)
    my_net.load_state_dict(checkpoint['state_dict'])

    # my_net = AirwayFormer_hierarchy(input_dim=23, num_classes1=6,num_classes2=20,num_classes3=127, dim=128, depth=2, heads=4, mlp_dim=256, dim_head=32,dropout=0.0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_net = my_net.to(device)
    test_accuracy1 =[]
    test_accuracy2=[]
    test_accuracy2_1=[]
    test_accuracy3=[]
    test_accuracy3_1=[]
    test_accuracy3_2=[]
    test_RC1 = []
    test_RC2 = []
    test_RC3 = []
    test_PR1 = []
    test_PR2 = []
    test_PR3 = []
    test_F11 = []
    test_F12 = []
    test_F13 = []
    for case in test_loader_case:
        trachea = loc_trachea(case.x)
        edge = case.edge_index.to(device)
        # x = case.x.type(torch.DoubleTensor).to(device)
        x = case.x.to(device)
        edge_prop = case.edge_attr
        y_lobar = case.y_lobar.to(device)
        y_lobar = y_lobar.long()
        y_seg = case.y_seg.to(device)
        y_seg = y_seg.long()
        y_subseg = case.y_subseg.to(device)
        y_subseg = y_subseg.long()

        spd = case.spd.to(device)
        where_are_inf = torch.isinf(spd)
        # nan替换成0,inf替换成nan
        spd[where_are_inf] = 30


        A_hat = to_adj(edge)
        D_hat = to_degree(A_hat)
        A_norm = to_Anorm(A_hat,D_hat)


        pred1,pred2,pred3 = my_net(x,spd.detach(),0.1)

        pred1 = pred1.max(dim=1)
        pred2 = pred2.max(dim=1)
        pred3 = pred3.max(dim=1)

        label1 = y_lobar.cpu().data.numpy()
        label2 = y_seg.cpu().data.numpy()
        label3 = y_subseg.cpu().data.numpy()
        pred1 = pred1[1].cpu().data.numpy()
        acc1 = np.sum((label1 == pred1).astype(np.uint8)) / (y_lobar.shape[0])

        pred2 = pred2[1].cpu().data.numpy()
        pred2_1 = seg2lobor(pred2)
        acc2 = np.sum((label2 == pred2).astype(np.uint8)) / (label2.shape[0])
        acc2_1 = np.sum((label1 == pred2_1).astype(np.uint8)) / (label1.shape[0])

        pred3 = pred3[1].cpu().data.numpy()
        pred3_2 = subseg2seg(pred3, trachea)
        pred3_1 = seg2lobor(pred3_2)
        acc3 = np.sum((label3 == pred3).astype(np.uint8)) / (label3.shape[0])
        acc3_2 = np.sum((label2 == pred3_2).astype(np.uint8)) / (label2.shape[0])
        acc3_1 = np.sum((label1 == pred3_1).astype(np.uint8)) / (label1.shape[0])


        RC1, PR1, F11 = evaluation(label1,pred1)#TODO
        RC2, PR2, F12 = evaluation(label2, pred2)#TODO
        RC3, PR3, F13 = evaluation(label3, pred3)
        test_RC1.append(RC1)
        test_RC2.append(RC2)
        test_RC3.append(RC3)

        test_PR1.append(PR1)
        test_PR2.append(PR2)
        test_PR3.append(PR3)

        test_F11.append(F11)
        test_F12.append(F12)
        test_F13.append(F13)


        '''con1_3 = consistence(pred1, pred3_1, label1)  # wrong->right
        con2_3 = consistence(pred2, pred3_2, label2)
        con3_1 = consistence(pred3_1, pred1, label1)  # right->wrong
        con3_2 = consistence(pred3_2, pred2, label2)
        test_consist1_3.append(con1_3)
        test_consist2_3.append(con2_3)
        test_consist3_1.append(con3_1)
        test_consist3_2.append(con3_2)'''

        test_accuracy1.append(acc1)#TODO
        test_accuracy2.append(acc2)#TODO
        test_accuracy2_1.append(acc2_1)
        test_accuracy3.append(acc3)
        test_accuracy3_1.append(acc3_1)
        test_accuracy3_2.append(acc3_2)
    test_accuracy1 = np.array(test_accuracy1)
    test_accuracy2 = np.array(test_accuracy2)
    test_accuracy2_1 = np.array(test_accuracy2_1)
    test_accuracy3 = np.array(test_accuracy3)
    test_accuracy3_2 = np.array(test_accuracy3_2)
    test_accuracy3_1 = np.array(test_accuracy3_1)
    test_RC1 = np.array(test_RC1)
    test_RC2 = np.array(test_RC2)
    test_RC3 = np.array(test_RC3)
    test_PR1 = np.array(test_PR1)
    test_PR2 = np.array(test_PR2)
    test_PR3 = np.array(test_PR3)
    test_F11 = np.array(test_F11)
    test_F12 = np.array(test_F12)
    test_F13 = np.array(test_F13)
    mean_RC1 = test_RC1.mean()
    mean_RC2 = test_RC2.mean()
    mean_RC3 = test_RC3.mean()
    mean_PR1 = test_PR1.mean()
    mean_PR2 = test_PR2.mean()
    mean_PR3 = test_PR3.mean()
    mean_F11 = test_F11.mean()
    mean_F12 = test_F12.mean()
    mean_F13 = test_F13.mean()
    mean_acc1 = test_accuracy1.mean()
    mean_acc2 = test_accuracy2.mean()
    mean_acc2_1 = test_accuracy2_1.mean()
    mean_acc3 = test_accuracy3.mean()
    mean_acc3_2 = test_accuracy3_2.mean()
    mean_acc3_1 = test_accuracy3_1.mean()
    '''test_consist1_3 = np.array(test_consist1_3)
    test_consist2_3 = np.array(test_consist2_3)
    test_consist3_1 = np.array(test_consist3_1)
    test_consist3_2 = np.array(test_consist3_2)
    con_mean_1_3 = np.mean(test_consist1_3)
    con_mean_2_3 = np.mean(test_consist2_3)
    con_mean_3_1 = np.mean(test_consist3_1)
    con_mean_3_2 = np.mean(test_consist3_2)'''
    # print("Accuracy of Test Samples:{}, {}({}), {}({}，{}) r->w({},{}) w->r({},{})".format(mean_acc1,mean_acc2,mean_acc2_1,mean_acc3,mean_acc3_1,mean_acc3_2,con_mean_3_1,con_mean_3_2,con_mean_1_3,con_mean_2_3))
    '''print("Accuracy of Test Samples:{}, {}({}), {}({}，{})".format(mean_acc1, mean_acc2,
                                                                  mean_acc2_1, mean_acc3,
                                                                  mean_acc3_1, mean_acc3_2, ))
    print("RC of Test Samples:{},{},{}".format(mean_RC1,mean_RC2,mean_RC3))
    print("PR of Test Samples:{},{},{}".format(mean_PR1,mean_PR2,mean_PR3))
    print("F1 of Test Samples:{},{},{}".format(mean_F11,mean_F12,mean_F13))'''
    Acc.append([mean_acc1, mean_acc2,mean_acc3])
    RC.append([mean_RC1,mean_RC2,mean_RC3])
    PR.append([mean_PR1,mean_PR2,mean_PR3])
    F1.append(([mean_F11,mean_F12,mean_F13]))
Acc = np.array(Acc)
RC = np.array(RC)
PR = np.array(PR)
F1 = np.array(F1)
acc_mean = np.mean(Acc,axis=0)
PR_mean = np.mean(PR,axis=0)
RC_mean = np.mean(RC,axis=0)
F1_mean = np.mean(F1,axis=0)
acc_dif = np.maximum(acc_mean-np.min(Acc,axis=0) ,np.max(Acc,axis=0)-acc_mean)
RC_dif = np.maximum(RC_mean-np.min(RC,axis=0) ,np.max(RC,axis=0)-RC_mean)
PR_dif = np.maximum(PR_mean-np.min(PR,axis=0) ,np.max(PR,axis=0)-PR_mean)
F1_dif = np.maximum(F1_mean-np.min(F1,axis=0) ,np.max(F1,axis=0)-F1_mean)
print("acc_mean",acc_mean,acc_dif)
print("rc",RC_mean,RC_dif)
print("pr",PR_mean,PR_dif)
print("f1",F1_mean,F1_dif)
for i in range(3):
    print(acc_mean[i],PR_mean[i],RC_mean[i],F1_mean[i])

