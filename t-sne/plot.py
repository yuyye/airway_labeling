import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
from dataset import multitask_dataset
from torch_geometric.loader import DataLoader
import os
import sys
import pickle
from transformer_base_spd import AirwayFormer_se
sys.path.append("..")
from utils import *



def plotlabels(S_lowDWeights, Trure_labels, name):
    file_color = open("a.txt", 'r')
    colors_file = []
    for line in file_color:
        colors_file.append(line.split())
    color_matrix = [[0 for _ in range(3)] for _ in range(len(colors_file) - 1)]
    for i in range(len(colors_file) - 1):
        color_matrix[i][0] = float(colors_file[i + 1][1]) / 255
        color_matrix[i][1] = float(colors_file[i + 1][2]) / 255
        color_matrix[i][2] = float(colors_file[i + 1][3]) / 255
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    for index in range(int(np.max(True_labels))+1):

        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, c=color_matrix[index], edgecolors=color_matrix[index], s=1, alpha=1)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=16, fontweight='normal', pad=20)
    #plt.savefig("t-sne/1.png")
    plt.show()
arr_x = np.load("checkpoint/base_spd/out3.npy")
arr_y = np.load("checkpoint/base_spd/y_sub.npy")
plotlabels(arr_x,arr_y,"the t-sne of out3 at sub level of base_spd")

'''arr_x = np.load("checkpoint/merge_base/out3.npy")
arr_y = np.load("checkpoint/merge_base/y_sub.npy")
plotlabels(arr_x,arr_y,"the t-sne of out3 at sub level of merge_base")'''
'''arr_x = np.load("checkpoint/base/out3.npy")
arr_y = np.load("checkpoint/base/y_sub.npy")
plotlabels(arr_x,arr_y,"the t-sne of out3 at sub level of base")'''




