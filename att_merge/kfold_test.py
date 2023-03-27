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
from transformer_atten import AirwayFormer_att_new
from sklearn.model_selection import KFold
import random


seed = 666
alpha = 0.1
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
import time
import shutil
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("..")
from dataset import multitask_dataset
from loss_functions import LabelSmoothCrossEntropyLoss,DependenceLoss
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
def get_p(epoch):
    '''if epoch<100:
        p = (100-epoch)/100*0.2
    else:
        p = (epoch-100)/500*0.2'''
    p = (800-epoch)/800*0.1
    return p

train_path2 = "/home/yuy/code/data/graph_ht_pred_train_v6_v3/"
test_path2 = "/home/yuy/code/data/graph_ht_pred_test_v6_v3/"
train_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_train/"
test_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_test_pred/"
epochs = 800
dataset1 = multitask_dataset(train_path2, train_path3, train=True)

dataset2 = multitask_dataset(test_path2, test_path3, test=True)
test_loader_case = DataLoader(dataset2, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

my_net = AirwayFormer_att_new(input_dim=23, num_classes1=6, num_classes2=20, num_classes3=127, dim=128, heads=4,
                              mlp_dim=256, dim_head=32, dropout=0., emb_dropout=0., alpha=alpha)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_net = my_net.to(device)
# optimizer = torch.optim.SGD(my_net.parameters(), lr=1e-4, momentum=0.9)
optimizer = torch.optim.Adam(my_net.parameters(), lr=5e-4, eps=1e-4)

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
    print("!!!!!!!!!!!!!!!!!!kfold!!!!!!!!!!!!!!!!!!", idx_fold)
    idx_select += 1
    idx_fold += 1

    max_acc = 0

    data_train_fold =[dataset1[i]for i in train_index]
    data_val_fold = [dataset1[i]for i in val_index]
    print(len(data_train_fold),len(data_val_fold))
    train_loader_case = DataLoader(data_train_fold, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader_case = DataLoader(data_val_fold, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    # checkpoint = torch.load(name)
    # my_net.load_state_dict(checkpoint['state_dict'])

    # my_net = AirwayFormer_hierarchy(input_dim=23, num_classes1=6,num_classes2=20,num_classes3=127, dim=128, depth=2, heads=4, mlp_dim=256, dim_head=32,dropout=0.0)

    save_dir = "checkpoints/kfoldtest15_{}_att_2_softatt{}_seed{}_dense_headmask_0.1_1_1_1_1_1__transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/".format(
        idx_fold, alpha, seed)
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
    loss2_plt = []  # 用于存放train_loss
    train_mean_loss = 100

    step_acc = []
    acc_val_plt = []
    acc_test_plt = []


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_net = my_net.to(device)
    # optimizer = torch.optim.SGD(my_net.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(my_net.parameters(), lr=5e-4, eps=1e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,300], gamma=0.1)
    print("!!!!!!!!seed=!!!!!!!!",seed)
    print("!!!!!!!!alpha!!!!!!!!",alpha)

    for epoch in range(epochs):
        my_net.train()
        time1 = time.time()
        test_accuracy11 = []
        train_accuracy11 = []
        test_accuracy21 = []
        train_accuracy21= []
        test_accuracy31 = []
        train_accuracy31 = []
        train_accuracy3_11 = []
        train_accuracy3_21 = []
        train_accuracy2_11 = []
        test_accuracy3_11 = []
        test_accuracy3_21 = []
        test_accuracy2_11 = []

        test_accuracy12 = []
        train_accuracy12 = []
        test_accuracy22 = []
        train_accuracy22 = []
        test_accuracy32 = []
        train_accuracy32 = []
        train_accuracy3_12 = []
        train_accuracy3_22 = []
        train_accuracy2_12 = []
        test_accuracy3_12 = []
        test_accuracy3_22 = []
        test_accuracy2_12 = []
        '''train_consist1_3 = []
        train_consist2_3 = []
        train_consist3_2 = []
        train_consist3_1 = []
        test_consist1_3 = []
        test_consist2_3 = []
        test_consist3_2 = []
        test_consist3_1 = []'''
        train_loss1 = []
        train_loss2 = []
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
            #p = get_p(epoch)
            output11, output21, output31, output12,output22, output32 = my_net(x,0.1)

            weights = case.weights.to(device)
            loss_function = LabelSmoothCrossEntropyLoss(weight=weights, smoothing=0.02)
            d_loss = DependenceLoss(weight=None, reduction='mean', alpha=0.01, p_loss1=1, p_loss2=2)
            loss =  loss_function(output11, y_lobar) +  loss_function(output21, y_seg) + loss_function(output31, y_subseg)+\
                  +  loss_function(output12, y_lobar) + loss_function(output22, y_seg) + loss_function(output32, y_subseg)

            # loss = loss_function(output1, y_lobar) + loss_function(output2, y_seg) + loss_function(output3, y_subseg)



            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=my_net.parameters(), max_norm=10, norm_type=2)

            pred11 = output11.max(dim=1)
            label1 = y_lobar.cpu().data.numpy()
            pred11 = pred11[1].cpu().data.numpy()
            acc11 = np.sum((label1 == pred11).astype(np.uint8)) / (label1.shape[0])
            train_accuracy11.append(acc11)
            pred21 = output21.max(dim=1)
            label2 = y_seg.cpu().data.numpy()
            pred21 = pred21[1].cpu().data.numpy()
            pred2_11 = seg2lobor(pred21)
            acc21 = np.sum((label2 == pred21).astype(np.uint8)) / (label2.shape[0])
            acc2_11 = np.sum((label1 == pred2_11).astype(np.uint8)) / (label1.shape[0])
            train_accuracy21.append(acc21)
            train_accuracy2_11.append(acc2_11)
            pred31 = output31.max(dim=1)
            label3 = y_subseg.cpu().data.numpy()
            pred31 = pred31[1].cpu().data.numpy()
            pred3_21 = subseg2seg(pred31, trachea)
            pred3_11 = seg2lobor(pred3_21)



            acc31 = np.sum((label3 == pred31).astype(np.uint8)) / (label3.shape[0])
            acc3_21 = np.sum((label2 == pred3_21).astype(np.uint8)) / (label2.shape[0])
            acc3_11 = np.sum((label1 == pred3_11).astype(np.uint8)) / (label1.shape[0])
            train_accuracy31.append(acc31)
            train_accuracy3_11.append(acc3_11)
            train_accuracy3_21.append(acc3_21)

            pred12 = output12.max(dim=1)
            pred12 = pred12[1].cpu().data.numpy()
            acc12 = np.sum((label1 == pred12).astype(np.uint8)) / (label1.shape[0])
            train_accuracy12.append(acc12)

            pred22 = output22.max(dim=1)
            label2 = y_seg.cpu().data.numpy()
            pred22 = pred22[1].cpu().data.numpy()
            pred2_12 = seg2lobor(pred22)
            acc22 = np.sum((label2 == pred22).astype(np.uint8)) / (label2.shape[0])
            acc2_12 = np.sum((label1 == pred2_12).astype(np.uint8)) / (label1.shape[0])
            train_accuracy22.append(acc22)
            train_accuracy2_12.append(acc2_12)
            pred32 = output32.max(dim=1)
            label3 = y_subseg.cpu().data.numpy()
            pred32 = pred32[1].cpu().data.numpy()
            pred3_22 = subseg2seg(pred32, trachea)
            pred3_12 = seg2lobor(pred3_22)

            acc32 = np.sum((label3 == pred32).astype(np.uint8)) / (label3.shape[0])
            acc3_22 = np.sum((label2 == pred3_22).astype(np.uint8)) / (label2.shape[0])
            acc3_12 = np.sum((label1 == pred3_12).astype(np.uint8)) / (label1.shape[0])
            train_accuracy32.append(acc32)
            train_accuracy3_12.append(acc3_12)
            train_accuracy3_22.append(acc3_22)

            train_loss.append(loss.item())
            optimizer.step()

        # scheduler.step()

        train_accuracy11 = np.array(train_accuracy11)
        train_accuracy21 = np.array(train_accuracy21)
        train_accuracy2_11 = np.array(train_accuracy2_11)
        train_accuracy31 = np.array(train_accuracy31)
        train_accuracy3_21 = np.array(train_accuracy3_21)
        train_accuracy3_11 = np.array(train_accuracy3_11)
        train_loss1 = np.array(train_loss1)
        train_loss2 = np.array(train_loss2)
        train_loss = np.array(train_loss)
        train_mean_acc11 = np.mean(train_accuracy11)
        train_mean_acc21 = np.mean(train_accuracy21)
        train_mean_acc2_11 = np.mean(train_accuracy2_11)
        train_mean_acc31 = np.mean(train_accuracy31)
        train_mean_acc3_21 = np.mean(train_accuracy3_21)
        train_mean_acc3_11 = np.mean(train_accuracy3_11)

        train_accuracy12 = np.array(train_accuracy12)
        train_accuracy22 = np.array(train_accuracy22)
        train_accuracy2_12 = np.array(train_accuracy2_12)
        train_accuracy32 = np.array(train_accuracy32)
        train_accuracy3_22 = np.array(train_accuracy3_22)
        train_accuracy3_12 = np.array(train_accuracy3_12)

        train_mean_acc12 = np.mean(train_accuracy12)
        train_mean_acc22 = np.mean(train_accuracy22)
        train_mean_acc2_12 = np.mean(train_accuracy2_12)
        train_mean_acc32 = np.mean(train_accuracy32)
        train_mean_acc3_22 = np.mean(train_accuracy3_22)
        train_mean_acc3_12 = np.mean(train_accuracy3_12)

        train_mean_loss = np.mean(train_loss)



        print(
            "epoch:{},loss:{}，acc:{}, {}({}), {}({}，{}), {},{}({}), {}({}, {})time:{}".format(epoch + 1, train_mean_loss,
                                                                       train_mean_acc11,
                                                                       train_mean_acc21,
                                                                       train_mean_acc2_11,
                                                                       train_mean_acc31,
                                                                       train_mean_acc3_11,
                                                                       train_mean_acc3_21,
                                                                       train_mean_acc12,
                                                                       train_mean_acc22,
                                                                       train_mean_acc2_12,
                                                                       train_mean_acc32,
                                                                       train_mean_acc3_12,
                                                                       train_mean_acc3_22,
                                                                       time.time() - time1))

        step_t.append(epoch)  # 此步为更新迭代步数
        loss1_plt.append(train_mean_loss)
        save_path = os.path.join(save_dir, 'loss.pkl')
        with open(save_path, 'wb') as fo:  # 将数据写入pkl文件
            pickle.dump(loss1_plt, fo)
        # loss2_plt.append(train_mean_loss2)

        '''try:
            loss1_lines.remove(loss1_lines[0])  # 移除上一步曲线
            # loss2_lines.remove(loss2_lines[0])
        except Exception:
            pass
        loss1_lines = plt.plot(step_t, loss1_plt, 'r', lw=1)  # lw为曲线宽度
        # loss2_lines = plt.plot(step_t, loss2_plt, 'b', lw=1)

        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylim(0, 4)
        plt.ylabel("loss")
        plt.legend(["loss1"])
        # ,"loss2"])
        plt.pause(0.1)  # 图片停留0.1s
        loss_path = os.path.join(save_dir, 'loss.png')
        plt.savefig(loss_path)'''

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
                output11,output21,output31,output12,output22,output32 = my_net(x,alpha)
                pred11 = output11.max(dim=1)
                label1 = y_lobar.cpu().data.numpy()
                pred11 = pred11[1].cpu().data.numpy()

                acc11 = np.sum((label1 == pred11).astype(np.uint8)) / (label1.shape[0])
                test_accuracy11.append(acc11)
                pred21 = output21.max(dim=1)
                label2 = y_seg.cpu().data.numpy()
                pred21 = pred21[1].cpu().data.numpy()
                pred2_11 = seg2lobor(pred21)
                acc21 = np.sum((label2 == pred21).astype(np.uint8)) / (label2.shape[0])
                acc2_11 = np.sum((label1 == pred2_11).astype(np.uint8)) / (label1.shape[0])
                test_accuracy21.append(acc21)
                test_accuracy2_11.append(acc2_11)
                pred31 = output31.max(dim=1)
                label3 = y_subseg.cpu().data.numpy()
                pred31 = pred31[1].cpu().data.numpy()
                pred3_21 = subseg2seg(pred31, trachea)
                pred3_11 = seg2lobor(pred3_21)

                acc31 = np.sum((label3 == pred31).astype(np.uint8)) / (label3.shape[0])
                acc3_21 = np.sum((label2 == pred3_21).astype(np.uint8)) / (label2.shape[0])
                acc3_11 = np.sum((label1 == pred3_11).astype(np.uint8)) / (label1.shape[0])
                test_accuracy31.append(acc31)
                test_accuracy3_11.append(acc3_11)
                test_accuracy3_21.append(acc3_21)

                pred12 = output12.max(dim=1)
                pred12 = pred12[1].cpu().data.numpy()
                acc12 = np.sum((label1 == pred12).astype(np.uint8)) / (label1.shape[0])
                test_accuracy12.append(acc12)

                pred22 = output22.max(dim=1)
                label2 = y_seg.cpu().data.numpy()
                pred22 = pred22[1].cpu().data.numpy()
                pred2_12 = seg2lobor(pred22)
                acc22 = np.sum((label2 == pred22).astype(np.uint8)) / (label2.shape[0])
                acc2_12 = np.sum((label1 == pred2_12).astype(np.uint8)) / (label1.shape[0])
                test_accuracy22.append(acc22)
                test_accuracy2_12.append(acc2_12)
                pred32 = output32.max(dim=1)
                label3 = y_subseg.cpu().data.numpy()
                pred32 = pred32[1].cpu().data.numpy()
                pred3_22 = subseg2seg(pred32, trachea)
                pred3_12 = seg2lobor(pred3_22)

                acc32 = np.sum((label3 == pred32).astype(np.uint8)) / (label3.shape[0])
                acc3_22 = np.sum((label2 == pred3_22).astype(np.uint8)) / (label2.shape[0])
                acc3_12 = np.sum((label1 == pred3_12).astype(np.uint8)) / (label1.shape[0])
                test_accuracy32.append(acc32)
                test_accuracy3_12.append(acc3_12)
                test_accuracy3_22.append(acc3_22)



            test_accuracy11 = np.array(test_accuracy11)
            test_accuracy21 = np.array(test_accuracy21)
            test_accuracy2_11 = np.array(test_accuracy2_11)
            test_accuracy31 = np.array(test_accuracy31)
            test_accuracy3_21 = np.array(test_accuracy3_21)
            test_accuracy3_11 = np.array(test_accuracy3_11)
            test_mean_acc11 = np.mean(test_accuracy11)
            test_mean_acc21 = np.mean(test_accuracy21)
            test_mean_acc2_11 = np.mean(test_accuracy2_11)
            test_mean_acc31 = np.mean(test_accuracy31)
            test_mean_acc3_21 = np.mean(test_accuracy3_21)
            test_mean_acc3_11 = np.mean(test_accuracy3_11)

            test_accuracy12 = np.array(test_accuracy12)
            test_accuracy22 = np.array(test_accuracy22)
            test_accuracy2_12 = np.array(test_accuracy2_12)
            test_accuracy32 = np.array(test_accuracy32)
            test_accuracy3_22 = np.array(test_accuracy3_22)
            test_accuracy3_12 = np.array(test_accuracy3_12)
            test_mean_acc12 = np.mean(test_accuracy12)
            test_mean_acc22 = np.mean(test_accuracy22)
            test_mean_acc2_12 = np.mean(test_accuracy2_12)
            test_mean_acc32 = np.mean(test_accuracy32)
            test_mean_acc3_22 = np.mean(test_accuracy3_22)
            test_mean_acc3_12 = np.mean(test_accuracy3_12)

            if test_mean_acc32 > max_acc:
                max_acc = test_mean_acc32
                state_dict = my_net.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].cpu()
                torch.save({
                    'epoch': epoch + 1,
                    'save_dir': save_dir,
                    'state_dict': state_dict},
                    os.path.join(save_dir, 'best.ckpt'))

            '''val_acc = test_mean_acc32
            #plt.figure()
            step_acc.append(epoch)  # 此步为更新迭代步数
            acc_val_plt.append(val_acc)
            acc_test_plt.append(test_acc)

            save_path = os.path.join(save_dir, 'acc_test.pkl')
            with open(save_path, 'wb') as fo:  # 将数据写入pkl文件
                pickle.dump(acc_test_plt, fo)
            save_path = os.path.join(save_dir, 'acc_val.pkl')
            with open(save_path, 'wb') as fo:  # 将数据写入pkl文件
                pickle.dump(acc_val_plt, fo)

            # loss2_plt.append(train_mean_loss2)
    
            try:
                val_lines.remove(val_lines[0])  # 移除上一步曲线
                test_lines.remove(test_lines[0])  # 移除上一步曲线
                # loss2_lines.remove(loss2_lines[0])
            except Exception:
                pass
            val_lines = plt.plot(step_acc, acc_val_plt, 'r', lw=1)  # lw为曲线宽度
            test_lines = plt.plot(step_acc, acc_test_plt, 'g', lw=1)  # lw为曲线宽度

            # loss2_lines = plt.plot(step_t, loss2_plt, 'b', lw=1)
    
            plt.title("acc")
            plt.xlabel("epoch")
            plt.ylim(0.7, 1.0)
            plt.ylabel("acc")
            plt.legend(["acc_val","acc_test"])
            plt.pause(10)  # 图片停留1s
            acc_path = os.path.join(save_dir, 'acc.png')
            plt.savefig(acc_path)'''


            # print("Accuracy of Test Samples:{}, {}({}), {}({}，{}) r->w({},{}) w->r({},{})".format(mean_acc1,mean_acc2,mean_acc2_1,mean_acc3,mean_acc3_1,mean_acc3_2,con_mean_3_1,con_mean_3_2,con_mean_1_3,con_mean_2_3))
            '''print("Accuracy of Test Samples:{}, {}({}), {}({}，{}),{},{}({}), {}({}，{})".format(test_mean_acc11, test_mean_acc21,
                                                                          test_mean_acc2_11, test_mean_acc31,
                                                                          test_mean_acc3_11, test_mean_acc3_21,
                                                                                               test_mean_acc12,
                                                                                            test_mean_acc22,
                                                                                            test_mean_acc2_12,
                                                                                            test_mean_acc32,
                                                                                            test_mean_acc3_12,
                                                                                            test_mean_acc3_22,
                                                                                            ))'''
    checkpoint = torch.load(os.path.join(save_dir, 'best.ckpt'))
    my_net.load_state_dict(checkpoint['state_dict'])
    test_accuracy11 = []
    test_accuracy21 = []
    test_accuracy3_11 = []
    test_accuracy3_21 = []
    test_accuracy2_11 = []

    test_accuracy12 = []
    test_accuracy22 = []
    test_accuracy32 = []
    test_accuracy3_12 = []
    test_accuracy3_22 = []
    test_accuracy2_12 = []
    test_accuracy31 = []
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

        output11, output21, output31, output12, output22, output32 = my_net(x, alpha)
        pred11 = output11.max(dim=1)
        label1 = y_lobar.cpu().data.numpy()
        pred11 = pred11[1].cpu().data.numpy()

        acc11 = np.sum((label1 == pred11).astype(np.uint8)) / (label1.shape[0])
        test_accuracy11.append(acc11)
        pred21 = output21.max(dim=1)
        label2 = y_seg.cpu().data.numpy()
        pred21 = pred21[1].cpu().data.numpy()
        pred2_11 = seg2lobor(pred21)
        acc21 = np.sum((label2 == pred21).astype(np.uint8)) / (label2.shape[0])
        acc2_11 = np.sum((label1 == pred2_11).astype(np.uint8)) / (label1.shape[0])
        test_accuracy21.append(acc21)
        test_accuracy2_11.append(acc2_11)
        pred31 = output31.max(dim=1)
        label3 = y_subseg.cpu().data.numpy()
        pred31 = pred31[1].cpu().data.numpy()
        pred3_21 = subseg2seg(pred31, trachea)
        pred3_11 = seg2lobor(pred3_21)

        acc31 = np.sum((label3 == pred31).astype(np.uint8)) / (label3.shape[0])
        acc3_21 = np.sum((label2 == pred3_21).astype(np.uint8)) / (label2.shape[0])
        acc3_11 = np.sum((label1 == pred3_11).astype(np.uint8)) / (label1.shape[0])
        test_accuracy31.append(acc31)
        test_accuracy3_11.append(acc3_11)
        test_accuracy3_21.append(acc3_21)

        pred12 = output12.max(dim=1)
        pred12 = pred12[1].cpu().data.numpy()
        acc12 = np.sum((label1 == pred12).astype(np.uint8)) / (label1.shape[0])
        test_accuracy12.append(acc12)

        pred22 = output22.max(dim=1)
        label2 = y_seg.cpu().data.numpy()
        pred22 = pred22[1].cpu().data.numpy()
        pred2_12 = seg2lobor(pred22)
        acc22 = np.sum((label2 == pred22).astype(np.uint8)) / (label2.shape[0])
        acc2_12 = np.sum((label1 == pred2_12).astype(np.uint8)) / (label1.shape[0])
        test_accuracy22.append(acc22)
        test_accuracy2_12.append(acc2_12)
        pred32 = output32.max(dim=1)
        label3 = y_subseg.cpu().data.numpy()
        pred32 = pred32[1].cpu().data.numpy()
        pred3_22 = subseg2seg(pred32, trachea)
        pred3_12 = seg2lobor(pred3_22)

        acc32 = np.sum((label3 == pred32).astype(np.uint8)) / (label3.shape[0])
        acc3_22 = np.sum((label2 == pred3_22).astype(np.uint8)) / (label2.shape[0])
        acc3_12 = np.sum((label1 == pred3_12).astype(np.uint8)) / (label1.shape[0])
        test_accuracy32.append(acc32)
        test_accuracy3_12.append(acc3_12)
        test_accuracy3_22.append(acc3_22)

    test_accuracy11 = np.array(test_accuracy11)
    test_accuracy21 = np.array(test_accuracy21)
    test_accuracy2_11 = np.array(test_accuracy2_11)
    test_accuracy31 = np.array(test_accuracy31)
    test_accuracy3_21 = np.array(test_accuracy3_21)
    test_accuracy3_11 = np.array(test_accuracy3_11)
    test_mean_acc11 = np.mean(test_accuracy11)
    test_mean_acc21 = np.mean(test_accuracy21)
    test_mean_acc2_11 = np.mean(test_accuracy2_11)
    test_mean_acc31 = np.mean(test_accuracy31)
    test_mean_acc3_21 = np.mean(test_accuracy3_21)
    test_mean_acc3_11 = np.mean(test_accuracy3_11)

    test_accuracy12 = np.array(test_accuracy12)
    test_accuracy22 = np.array(test_accuracy22)
    test_accuracy2_12 = np.array(test_accuracy2_12)
    test_accuracy32 = np.array(test_accuracy32)
    test_accuracy3_22 = np.array(test_accuracy3_22)
    test_accuracy3_12 = np.array(test_accuracy3_12)
    test_mean_acc12 = np.mean(test_accuracy12)
    test_mean_acc22 = np.mean(test_accuracy22)
    test_mean_acc2_12 = np.mean(test_accuracy2_12)
    test_mean_acc32 = np.mean(test_accuracy32)
    test_mean_acc3_22 = np.mean(test_accuracy3_22)
    test_mean_acc3_12 = np.mean(test_accuracy3_12)

    acc_save = np.array([test_mean_acc11, test_mean_acc21,
                                                                              test_mean_acc2_11, test_mean_acc31,
                                                                              test_mean_acc3_11, test_mean_acc3_21,
                                                                                                test_mean_acc12,
                                                                                                test_mean_acc22,
                                                                                                test_mean_acc2_12,
                                                                                                test_mean_acc32,
                                                                                                test_mean_acc3_12,
                                                                                                test_mean_acc3_22])

    save_path = os.path.join(save_dir, 'acc_test.pkl')
    with open(save_path, 'wb') as fo:  # 将数据写入pkl文件
        pickle.dump(acc_save, fo)

    print("Accuracy of Test Samples of kfold{}:{}, {}({}), {}({}，{}),{},{}({}), {}({}，{})".format(idx_fold-1,test_mean_acc11, test_mean_acc21,
                                                                                       test_mean_acc2_11,
                                                                                       test_mean_acc31,
                                                                                       test_mean_acc3_11,
                                                                                       test_mean_acc3_21,
                                                                                       test_mean_acc12,
                                                                                       test_mean_acc22,
                                                                                       test_mean_acc2_12,
                                                                                       test_mean_acc32,
                                                                                       test_mean_acc3_12,
                                                                                       test_mean_acc3_22,
                                                                                       ))



