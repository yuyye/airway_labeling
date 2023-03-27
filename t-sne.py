
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
from dataset import multitask_dataset
from torch_geometric.loader import DataLoader
import os
import pickle
from utils import *
sys.path.append("att_merge/")
from transformer_atten import AirwayFormer_att_Gres
'''sys.path.append("att_2_soft/")
from transformer_att import AirwayFormer_att_new'''
alpha = 0.1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=100,learning_rate = 100,n_iter = 5000,perplexity=40.)

    x_ts = ts.fit_transform(feat)

    #print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final

def plotlabels(S_lowDWeights, Trure_labels, name):
    file_color = open("../a.txt", 'r')
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
        print(X.shape)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=16, fontweight='normal', pad=20)
    #plt.savefig("t-sne/1.png")
    plt.show()

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

dataset1 = multitask_dataset(train_path2, train_path3, train=True)
train_loader_case = DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

dataset2 = multitask_dataset(test_path2, test_path3, test=True)
test_loader_case = DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
feat_train = []
feat_test = []
# t-SNE的最终结果的降维与可视化
save_dir = "t-sne/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

my_net = AirwayFormer_att_Gres(input_dim=23, num_classes1=6, num_classes2=20, num_classes3=127, dim=128, heads=4,
                                  mlp_dim=256, dim_head=32, dropout = 0., emb_dropout=0.,alpha = alpha)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_net = my_net.to(device)

module_path = "/home/yuy/code/transformer/att_merge/checkpoints/att_2_soft0.1_dense_headmask_0.1_1_1_1_1_1_seed222_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
checkpoint = torch.load(module_path)
my_net.load_state_dict(checkpoint['state_dict'])
idx = 0
'''for case in test_loader_case:
    idx += 1
    if idx != 3 :
        continue
    x = case.x.to(device)
    y_lobar = case.y_lobar
    y_seg = case.y_seg
    y_subseg = case.y_subseg

    x_raw,output11, output21, output31, output12, output22, output32 = my_net(x, 0.)
    x_raw = x_raw[0]
    output31 = output31[0]
    output32 = output32[0]
    x = visual(x.cpu().numpy())
    x_raw = visual(x_raw)
    output31 = visual(output31)
    output32 = visual(output32)'''

idx = 0
for case in train_loader_case:
    x = case.x.to(device)
    y = case.y_lobar
    edge = case.edge_index.to(device)
    A_hat = to_adj(edge)
    D_hat = to_degree(A_hat)
    A_norm = to_Anorm(A_hat, D_hat)
    x_raw, output11, output21, output31, output12, output22, output32 = my_net(x, A_norm.detach(),0.)
    #x_raw, output11, output21, output31, output12, output22, output32 = my_net(x, 0.)
    out = x_raw.detach().cpu().numpy()[0]
    idx += 1
    if idx ==1:
        arr_x = out
        arr_y = y
    else:
        arr_x = np.concatenate((out, arr_x), axis=0)
        arr_y = np.concatenate((y, arr_y), axis=0)

    y_lobar = case.y_lobar
    y_seg = case.y_seg
    y_subseg = case.y_subseg
plotlabels(visual(arr_x),arr_y,"the t-sne of x at lob level in trian-case")


'''label3 = y_subseg.cpu().data.numpy()
pred32 = output32.max(dim=1)
pred32 = pred32[1].cpu().data.numpy()

acc32 = np.sum((label3 == pred32).astype(np.uint8)) / (label3.shape[0])'''



'''x_trian = []
y_train = []
for case in train_loader_case:
    x_trian.append(visual(case.x))
    y_train.append([case.y_lobar,case.y_seg,case.y_subseg])'''


'''for case in test_loader_case:
    x = case.x
    feat_test.append(x)
feat_train = np.array(feat_train)
feat_test = np.array(feat_test)
print(feat_train.shape,feat_test.shape)
x_train = visual(feat_train)
x_test = visual(feat_test)'''

