# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:26:37 2022

@author: Yu
"""

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
from transformer import AirwayFormer_jump, AirwayFormer_hierarchy, AirwayFormer_dense
import random

seed = 444
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random
import time
import shutil
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
sys.path.append("..")
from dataset import multitask_dataset
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


train_path2 = "/home/yuy/code/data/graph_ht_pred_train_v6_v3/"
test_path2 = "/home/yuy/code/data/graph_ht_pred_test_v6_v3/"
train_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_train/"
test_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_test_pred/"
epochs = 800

dataset1 = multitask_dataset(train_path2, train_path3)
dataset2 = multitask_dataset(test_path2, test_path3, test=True)
test_loader_case = DataLoader(dataset2, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

save_dir2 = "analysis/dense/"
if not os.path.exists(save_dir2):
    os.makedirs(save_dir2)


kfold = KFold(n_splits=15)
fold_select = random.sample(range(0, 14), 5)
fold_select.sort()
idx_fold = 0
idx_select = 0
for train_index, val_index in kfold.split(dataset1):
    if idx_select>5:
        break
    if idx_fold != fold_select[idx_select]:
        idx_fold += 1
        continue
    idx_select += 1
    print("!!!!!!!!!!!!!!!!!!kfold!!!!!!!!!!!!!!!!!!", idx_fold)
    data_train_fold =[dataset1[i]for i in train_index]
    data_val_fold = [dataset1[i]for i in val_index]

    train_loader_case = DataLoader(data_train_fold, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader_case = DataLoader(data_val_fold, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    max_acc = 0
    # torch.set_default_dtype(torch.float64)
    my_net = AirwayFormer_hierarchy(input_dim=23, num_classes1=6, num_classes2=20, num_classes3=127, dim=128, heads=4,
                                    mlp_dim=256, dim_head=32, dropout=0., )

    save_dir = "checkpoints/kfold15_{}hierarchy_seed{}_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/".format(idx_fold,seed)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    sys.stdout = Logger(logfile)
    pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
    for f in pyfiles:
        shutil.copy(f, os.path.join(save_dir, f))
    # checkpoint = torch.load(name)
    # my_net.load_state_dict(checkpoint['state_dict'])

    # my_net = AirwayFormer_hierarchy(input_dim=23, num_classes1=6,num_classes2=20,num_classes3=127, dim=128, depth=2, heads=4, mlp_dim=256, dim_head=32,dropout=0.0)
    step_t = []  # 用于存放横坐标
    loss1_plt = []  # 用于存放train_loss

    train_mean_loss = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_net = my_net.to(device)
    # optimizer = torch.optim.SGD(my_net.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(my_net.parameters(), lr=5e-4, eps=1e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,300], gamma=0.1)

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

            output1, output2, output3 = my_net(x)

            weights = case.weights.to(device)
            loss_function = LabelSmoothCrossEntropyLoss(weight=weights, smoothing=0.02)
            loss1 = loss_function(output1, y_lobar) + loss_function(output2, y_seg) + loss_function(output3, y_subseg)

            # loss = loss_function(output1, y_lobar) + loss_function(output2, y_seg) + loss_function(output3, y_subseg)


            loss = loss1

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
            train_loss1.append(loss1.item())
            train_loss.append(loss.item())

            optimizer.step()

        # scheduler.step()

        train_accuracy1 = np.array(train_accuracy1)
        train_accuracy2 = np.array(train_accuracy2)
        train_accuracy2_1 = np.array(train_accuracy2_1)
        train_accuracy3 = np.array(train_accuracy3)
        train_accuracy3_2 = np.array(train_accuracy3_2)
        train_accuracy3_1 = np.array(train_accuracy3_1)
        train_loss1 = np.array(train_loss1)
        train_loss = np.array(train_loss)
        train_mean_acc1 = np.mean(train_accuracy1)
        train_mean_acc2 = np.mean(train_accuracy2)
        train_mean_acc2_1 = np.mean(train_accuracy2_1)
        train_mean_acc3 = np.mean(train_accuracy3)
        train_mean_acc3_2 = np.mean(train_accuracy3_2)
        train_mean_acc3_1 = np.mean(train_accuracy3_1)
        train_mean_loss1 = np.mean(train_loss1)
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
            "epoch:{},loss:{}，acc:{}, {}({}), {}({}，{})time:{}".format(epoch + 1, train_mean_loss1,
                                                                       train_mean_acc1,
                                                                       train_mean_acc2,
                                                                       train_mean_acc2_1,
                                                                       train_mean_acc3,
                                                                       train_mean_acc3_1,
                                                                       train_mean_acc3_2,
                                                                       time.time() - time1))
        step_t.append(epoch)  # 此步为更新迭代步数
        loss1_plt.append(train_mean_loss1)


        try:
            loss1_lines.remove(loss1_lines[0])  # 移除上一步曲线
        except Exception:
            pass
        loss1_lines = plt.plot(step_t, loss1_plt, 'r', lw=1)  # lw为曲线宽度

        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylim(0, 2)
        plt.ylabel("loss")
        plt.legend(["loss1"])
        plt.pause(0.1)  # 图片停留0.1s
        plt.savefig("/home/yuy/code/transformer/analysis/dense/loss.png")

        if (epoch + 1) % 10 == 0:
            my_net.eval()
            for case in val_loader_case:
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

                pred1, pred2, pred3 = my_net(x)
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
            print("Accuracy of val Samples:{}, {}({}), {}({}，{})".format(mean_acc1, mean_acc2,
                                                                          mean_acc2_1, mean_acc3,
                                                                          mean_acc3_1, mean_acc3_2, ))
            if epoch>400 and mean_acc3 > max_acc:
                max_acc = mean_acc3
                state_dict = my_net.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].cpu()
                torch.save({
                    'epoch': epoch + 1,
                    'save_dir': save_dir,
                    'state_dict': state_dict},
                    os.path.join(save_dir, 'best.ckpt'))
                #print("!!!!!!!!!!!!!!!!!!!!!!best!!!!!!!!!!!!!!!!!!!!!!")


        if (epoch + 1) % 100 == 0:
            state_dict = my_net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch + 1,
                'save_dir': save_dir,
                'state_dict': state_dict},
                os.path.join(save_dir, '%04d.ckpt' % (epoch + 1)))
    idx_fold  += 1
    my_net.eval()

    test_accuracy1 = []
    test_accuracy2 = []
    test_accuracy2_1 = []
    test_accuracy3 = []
    test_accuracy3_1 = []
    test_accuracy3_2 = []
    checkpoint = torch.load(os.path.join(save_dir, 'best.ckpt'))
    my_net.load_state_dict(checkpoint['state_dict'])
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

        pred1, pred2, pred3 = my_net(x)
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
    print("!!!!!!!!!!!!!!!!!!!!!!test!!!!!!!!!!!!!!!!!!!!!!")
    # print("Accuracy of Test Samples:{}, {}({}), {}({}，{}) r->w({},{}) w->r({},{})".format(mean_acc1,mean_acc2,mean_acc2_1,mean_acc3,mean_acc3_1,mean_acc3_2,con_mean_3_1,con_mean_3_2,con_mean_1_3,con_mean_2_3))
    print("Accuracy of Test Samples of the kfold{}:{}, {}({}), {}({}，{})".format(idx_fold-1,mean_acc1, mean_acc2,
                                                                  mean_acc2_1, mean_acc3,
                                                               mean_acc3_1, mean_acc3_2, ))
    acc = np.array([mean_acc1, mean_acc2,
                                                                  mean_acc2_1, mean_acc3,
                                                                  mean_acc3_1, mean_acc3_2,])
    save_path = os.path.join(save_dir, 'acc_test.pkl')
    with open(save_path, 'wb') as fo:  # 将数据写入pkl文件
        pickle.dump(acc, fo)




