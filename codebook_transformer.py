# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:26:37 2022

@author: Yu
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
import os
import numpy as np
from dataset import multitask_dataset
from transformer import AirwayFormer_jump, AirwayFormer_hierarchy, AirwayFormer_dense, AirwayFormer_unidrop
from loss_functions import LabelSmoothCrossEntropyLoss, DependenceLoss
from utils import *

torch.manual_seed(222)  # cpu
torch.cuda.manual_seed(222)  # gpu
np.random.seed(222)  # numpy
import time
import shutil
import sys
import pickle
import matplotlib.pyplot as plt
from transformer_edge import AirwayFormer_codebook


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


train_path2 = "/home/yuy/code/data/graph_ht_pred_train_v6_v3/"
test_path2 = "/home/yuy/code/data/graph_ht_pred_test_v6_v3/"
train_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_train/"
test_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_test_pred/"
epochs = 600

dataset3 = multitask_dataset(train_path2, train_path3)
train_loader_val = DataLoader(dataset3, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
dataset2 = multitask_dataset(test_path2, test_path3, test=True)
test_loader_case = DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
max_acc = 0
# torch.set_default_dtype(torch.float64)

save_dir = "checkpoints/dense_true_4epochs_codebook_dictdrop0_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = os.path.join(save_dir, 'log')
sys.stdout = Logger(logfile)
pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
for f in pyfiles:
    shutil.copy(f, os.path.join(save_dir, f))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir2 = "analysis/dense/"
if not os.path.exists(save_dir2):
    os.makedirs(save_dir2)

name = "checkpoints/dense_all_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
test_net = AirwayFormer_dense(input_dim=23, num_classes1=6, num_classes2=20, num_classes3=127, dim=128, depth=2,
                              heads=4, mlp_dim=256, dim_head=32, dropout=0.0)
test_net.to(device)
checkpoint = torch.load(name)
test_net.load_state_dict(checkpoint['state_dict'])

'''dict_test_list = []
test_net.eval()

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

    _, _, pred3 = test_net(x)
    pred3 = pred3.max(dim=1)

    pred3 = pred3[1].cpu().data.numpy()
    pred3_2 = subseg2seg(pred3, trachea)

    tmp2 = np.tile(pred3_2.T, (pred3_2.shape[0], 1))
    tmp1 = tmp2.T
    dict_test_list.append(np.abs(tmp1 - tmp2))'''
# name = "checkpoints/dloss_hierachy_ploss_1_2/0100.ckpt"
my_net = AirwayFormer_codebook(input_dim=23, num_classes1=6, num_classes2=20, num_classes3=127, dim=128, heads=4,
                               mlp_dim=256, dim_head=32, dropout=0.)

# checkpoint = torch.load(name)
# my_net.load_state_dict(checkpoint['state_dict'])

# my_net = AirwayFormer_hierarchy(input_dim=23, num_classes1=6,num_classes2=20,num_classes3=127, dim=128, depth=2, heads=4, mlp_dim=256, dim_head=32,dropout=0.0)
step_t = []  # 用于存放横坐标
loss1_plt = []  # 用于存放train_loss
train_mean_loss = 100

my_net = my_net.to(device)
# optimizer = torch.optim.SGD(my_net.parameters(), lr=1e-4, momentum=0.9)
optimizer = torch.optim.Adam(my_net.parameters(), lr=5e-4, eps=1e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,300], gamma=0.1)

book = np.zeros(127)
for j in range(127):
    if j == 0:
        book[j] = 18
    else:
        book[j] = (j - 1) // 7

dict_list = []
dataset1 = multitask_dataset(train_path2, train_path3, train=True)
train_loader_case = DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
for case in train_loader_case:
    dict = np.zeros([case.x.shape[0], case.x.shape[0]])
    dict_list.append(dict)
for epoch in range(epochs):
    my_net.train()
    time1 = time.time()
    test_accuracy1 = []
    train_accuracy1 = []
    test_accuracy2 = []
    train_accuracy2 = []
    test_accuracy3 = []
    train_accuracy3 = []
    train_accuracy3_1 = []
    train_accuracy3_2 = []
    train_accuracy2_1 = []
    test_accuracy3_1 = []
    test_accuracy3_2 = []
    test_accuracy2_1 = []
    '''train_consist1_3 = []
    train_consist2_3 = []
    train_consist3_2 = []
    train_consist3_1 = []
    test_consist1_3 = []
    test_consist2_3 = []
    test_consist3_2 = []
    test_consist3_1 = []'''
    train_loss1 = []
    train_loss = []
    index_case = 0

    for case in train_loader_case:

        trachea = loc_trachea(case.x)
        edge = case.edge_index.to(device)
        edge_prop = case.edge_attr
        # x = case.x.type(torch.DoubleTensor).to(device)
        x = case.x.to(device)
        y_lobar = case.y_lobar.to(device)
        y_lobar = y_lobar.long()
        y_seg = case.y_seg.to(device)
        y_seg = y_seg.long()
        y_subseg = case.y_subseg.to(device)
        y_subseg = y_subseg.long()

        optimizer.zero_grad()
        output1, output2, output3 = my_net(x, torch.tensor(dict_list[index_case]).to(device))

        weights = case.weights.to(device)
        loss_function = LabelSmoothCrossEntropyLoss(weight=weights, smoothing=0.02)
        loss = loss_function(output1, y_lobar) + loss_function(output2, y_seg) + loss_function(output3, y_subseg)

        for name, parms in my_net.named_parameters():

            if epoch > 0:
                assert not (torch.isnan(parms.grad.data).sum() > 0), ['before', "loss", loss, name, "parm", parms,
                                                                      "grad",
                                                                      parms.grad, "out1", "out2", output1, "out3",
                                                                      output2,
                                                                      output3]
                assert not (torch.isinf(parms.grad.data).sum() > 0), ['before', "loss", loss, name, "parm", parms,
                                                                      "grad",
                                                                      parms.grad, "out1", "out2", output1, "out3",
                                                                      output2,
                                                                      output3]
            assert not (torch.isnan(parms).sum() > 0), ['before', "loss", loss, name, "parm", parms, "grad",
                                                        parms.grad, "out1", "out2", output1, "out3", output2, output3]
            assert not (torch.isinf(parms).sum() > 0), ['before', "loss", loss, name, "parm", parms, "grad",
                                                        parms.grad, "out1", "out2", output1, "out3", output2, output3]

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=my_net.parameters(), max_norm=10, norm_type=2)

        pred1 = output1.max(dim=1)
        label1 = y_lobar.cpu().data.numpy()
        pred1 = pred1[1].cpu().data.numpy()

        acc1 = np.sum((label1 == pred1).astype(np.uint8)) / (label1.shape[0])
        train_accuracy1.append(acc1)
        pred2 = output2.max(dim=1)
        label2 = y_seg.cpu().data.numpy()
        pred2 = pred2[1].cpu().data.numpy()
        pred2_1 = seg2lobor(pred2)
        acc2 = np.sum((label2 == pred2).astype(np.uint8)) / (label2.shape[0])
        acc2_1 = np.sum((label1 == pred2_1).astype(np.uint8)) / (label1.shape[0])
        train_accuracy2.append(acc2)
        train_accuracy2_1.append(acc2_1)
        pred3 = output3.max(dim=1)

        label3 = y_subseg.cpu().data.numpy()
        pred3 = pred3[1].cpu().data.numpy()
        pred3_2 = subseg2seg(pred3, trachea)
        pred3_1 = seg2lobor(pred3_2)

        tmp2 = np.tile(pred3_2.T, (pred3_2.shape[0], 1))
        tmp1 = tmp2.T
        # print(pred3.shape,tmp1.shape,tmp2.shape)
        dict_list[index_case] = np.abs(tmp1 - tmp2)
        '''for i in range(dict.shape[0]):
            for j in range(dict.shape[1]):
                dict[i][j] = pred3[i]-pred3[j]'''

        '''con1_3 = consistence(pred1,pred3_1,label1) #wrong->right
        con2_3 = consistence(pred2,pred3_2,label2)
        con3_1 = consistence(pred3_1, pred1, label1)  # right->wrong
        con3_2 = consistence(pred3_2,pred2,label2)
        train_consist1_3.append(con1_3)
        train_consist2_3.append(con2_3)
        train_consist3_1.append(con3_1)
        train_consist3_2.append(con3_2)'''

        acc3 = np.sum((label3 == pred3).astype(np.uint8)) / (label3.shape[0])
        acc3_2 = np.sum((label2 == pred3_2).astype(np.uint8)) / (label2.shape[0])
        acc3_1 = np.sum((label1 == pred3_1).astype(np.uint8)) / (label1.shape[0])
        train_accuracy3.append(acc3)
        train_accuracy3_1.append(acc3_1)
        train_accuracy3_2.append(acc3_2)
        train_loss.append(loss.item())
        for name, parms in my_net.named_parameters():
            assert not (torch.isnan(parms.grad.data).sum() > 0), ['back', "loss", loss, name, "parm", parms, "grad",
                                                                  parms.grad, "out1", "out2", output1, "out3", output2,
                                                                  output3]
            assert not (torch.isinf(parms.grad.data).sum() > 0), ['back', "loss", loss, name, "parm", parms, "grad",
                                                                  parms.grad, "out1", "out2", output1, "out3", output2,
                                                                  output3]
            assert not (torch.isnan(parms).sum() > 0), ['back', "loss", loss, name, "parm", parms, "grad",
                                                        parms.grad, "out1", "out2", output1, "out3", output2, output3]
            assert not (torch.isinf(parms).sum() > 0), ['back', "loss", loss, name, "parm", parms, "grad",
                                                        parms.grad, "out1", "out2", output1, "out3", output2, output3]
        optimizer.step()
        for name, parms in my_net.named_parameters():
            assert not (torch.isnan(parms.grad.data).sum() > 0), ['step', "loss", loss, name, "parm", parms, "grad",
                                                                  parms.grad, "out1", "out2", output1, "out3", output2,
                                                                  output3]
            assert not (torch.isinf(parms.grad.data).sum() > 0), ['step', "loss", loss, name, "parm", parms, "grad",
                                                                  parms.grad, "out1", "out2", output1, "out3", output2,
                                                                  output3]
            assert not (torch.isnan(parms).sum() > 0), ['step', "loss", loss, name, "parm", parms, "grad",
                                                        parms.grad, "out1", "out2", output1, "out3", output2, output3]
            assert not (torch.isinf(parms).sum() > 0), ['step', "loss", loss, name, "parm", parms, "grad",
                                                        parms.grad, "out1", "out2", output1, "out3", output2, output3]

        index_case += 1

    # scheduler.step()

    train_accuracy1 = np.array(train_accuracy1)
    train_accuracy2 = np.array(train_accuracy2)
    train_accuracy2_1 = np.array(train_accuracy2_1)
    train_accuracy3 = np.array(train_accuracy3)
    train_accuracy3_2 = np.array(train_accuracy3_2)
    train_accuracy3_1 = np.array(train_accuracy3_1)
    train_loss = np.array(train_loss)
    train_mean_acc1 = np.mean(train_accuracy1)
    train_mean_acc2 = np.mean(train_accuracy2)
    train_mean_acc2_1 = np.mean(train_accuracy2_1)
    train_mean_acc3 = np.mean(train_accuracy3)
    train_mean_acc3_2 = np.mean(train_accuracy3_2)
    train_mean_acc3_1 = np.mean(train_accuracy3_1)
    train_mean_loss = np.mean(train_loss)
    '''train_consist1_3 = np.array(train_consist1_3)
    train_consist2_3 = np.array(train_consist2_3)
    train_consist3_1 = np.array(train_consist3_1)
    train_consist3_2 = np.array(train_consist3_2)
    con_mean_1_3 = np.mean(train_consist1_3)
    con_mean_2_3 = np.mean(train_consist2_3)
    con_mean_3_1 = np.mean(train_consist3_1)
    con_mean_3_2 = np.mean(train_consist3_2)'''

    print(
        "epoch:{},loss:{}，acc:{}, {}({}), {}({}，{})time:{}".format(epoch + 1, train_mean_loss,
                                                                   train_mean_acc1,
                                                                   train_mean_acc2,
                                                                   train_mean_acc2_1,
                                                                   train_mean_acc3,
                                                                   train_mean_acc3_1,
                                                                   train_mean_acc3_2,
                                                                   time.time() - time1))
    step_t.append(epoch)  # 此步为更新迭代步数
    loss1_plt.append(train_mean_loss)

    try:
        loss1_lines.remove(loss1_lines[0])  # 移除上一步曲线
    except Exception:
        pass
    loss1_lines = plt.plot(step_t, loss1_plt, 'r', lw=1)  # lw为曲线宽度
    # loss2_lines = plt.plot(step_t, loss2_plt, 'b', lw=1)

    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylim(0, 2)
    plt.ylabel("loss")
    plt.legend(["loss"])
    # ,"loss2"])
    plt.pause(0.1)  # 图片停留0.1s
    plt.savefig("/home/yuy/code/transformer/analysis/dense/loss.png")

    if (epoch + 1) % 10 == 0:
        dict_test_list = []
        test_net.eval()
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

            _, _, pred3 = test_net(x)
            pred3 = pred3.max(dim=1)

            pred3 = pred3[1].cpu().data.numpy()
            pred3_2 = subseg2seg(pred3, trachea)

            tmp2 = np.tile(pred3_2.T, (pred3_2.shape[0], 1))
            tmp1 = tmp2.T
            dict_test_list.append(np.abs(tmp1 - tmp2))
        my_net.eval()
        for epoch_test in range(10):
            test_accuracy1 = []
            test_accuracy2 = []
            test_accuracy3 = []
            test_accuracy2_1 = []
            test_accuracy3_1 = []
            test_accuracy3_2 = []

            index_case_test = 0
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

                pred1, pred2, pred3 = my_net(x, torch.tensor(dict_test_list[index_case_test]).to(device))
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

                tmp2 = np.tile(pred3_2.T, (pred3_2.shape[0], 1))
                tmp1 = tmp2.T
                # print(pred3.shape,tmp1.shape,tmp2.shape)
                dict_test_list[index_case_test] = np.abs(tmp1 - tmp2)

                '''con1_3 = consistence(pred1, pred3_1, label1)  # wrong->right
                con2_3 = consistence(pred2, pred3_2, label2)
                con3_1 = consistence(pred3_1, pred1, label1)  # right->wrong
                con3_2 = consistence(pred3_2, pred2, label2)
                test_consist1_3.append(con1_3)
                test_consist2_3.append(con2_3)
                test_consist3_1.append(con3_1)
                test_consist3_2.append(con3_2)'''

                test_accuracy1.append(acc1)
                test_accuracy2.append(acc2)
                test_accuracy2_1.append(acc2_1)
                test_accuracy3.append(acc3)
                test_accuracy3_1.append(acc3_1)
                test_accuracy3_2.append(acc3_2)
                index_case_test += 1
            test_accuracy1 = np.array(test_accuracy1)
            test_accuracy2 = np.array(test_accuracy2)
            test_accuracy2_1 = np.array(test_accuracy2_1)
            test_accuracy3 = np.array(test_accuracy3)
            test_accuracy3_2 = np.array(test_accuracy3_2)
            test_accuracy3_1 = np.array(test_accuracy3_1)
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
            print("Accuracy of Test Samples:{}, {}({}), {}({}，{})".format(mean_acc1, mean_acc2,
                                                                          mean_acc2_1, mean_acc3,
                                                                          mean_acc3_1, mean_acc3_2, ))
        if mean_acc3 > max_acc:
            max_acc = mean_acc3
            state_dict = my_net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch + 1,
                'save_dir': save_dir,
                'state_dict': state_dict},
                os.path.join(save_dir, 'best.ckpt'))
            print("best!!!!!!!!!!!")

    if (epoch + 1) % 100 == 0:
        state_dict = my_net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'epoch': epoch + 1,
            'save_dir': save_dir,
            'state_dict': state_dict},
            os.path.join(save_dir, '%04d.ckpt' % (epoch + 1)))

'''
my_net.eval()

test_accuracy1=[]
test_accuracy2=[]
test_accuracy2_1=[]
test_accuracy3=[]
test_accuracy3_1=[]
test_accuracy3_2=[]
for case in test_loader_case:
    trachea = loc_trachea(case.x)
    edge = case.edge_index.to(device)
    #x = case.x.type(torch.DoubleTensor).to(device)
    x = case.x.to(device)
    edge_prop = case.edge_attr
    y_lobar = case.y_lobar.to(device)
    y_lobar = y_lobar.long()
    y_seg = case.y_seg.to(device)
    y_seg = y_seg.long()
    y_subseg = case.y_subseg.to(device)
    y_subseg = y_subseg.long()

    pred1,pred2,pred3 = my_net(x)
    pred1 = pred1.max(dim = 1)
    pred2 = pred2.max(dim = 1)
    pred3 = pred3.max(dim = 1)


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
    pred3_2 = subseg2seg(pred3,trachea)
    pred3_1 = seg2lobor(pred3_2)
    acc3 = np.sum((label3 == pred3).astype(np.uint8)) / (label3.shape[0])
    acc3_2 = np.sum((label2 == pred3_2).astype(np.uint8)) / (label2.shape[0])
    acc3_1 = np.sum((label1 == pred3_1).astype(np.uint8)) / (label1.shape[0])


    test_accuracy1.append(acc1)
    test_accuracy2.append(acc2)
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
mean_acc1 = test_accuracy1.mean()
mean_acc2 = test_accuracy2.mean()
mean_acc2_1 = test_accuracy2_1.mean()
mean_acc3 = test_accuracy3.mean()
mean_acc3_2 = test_accuracy3_2.mean()
mean_acc3_1 = test_accuracy3_1.mean()

#print("Accuracy of Test Samples:{}, {}({}), {}({}，{}) r->w({},{}) w->r({},{})".format(mean_acc1,mean_acc2,mean_acc2_1,mean_acc3,mean_acc3_1,mean_acc3_2,con_mean_3_1,con_mean_3_2,con_mean_1_3,con_mean_2_3))
print("Accuracy of Test Samples:{}, {}({}), {}({}，{})".format(mean_acc1, mean_acc2,
                                                                                      mean_acc2_1, mean_acc3,
                                                                                      mean_acc3_1, mean_acc3_2,))'''


