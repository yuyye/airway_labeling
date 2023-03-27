import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np
import torch
from transformer_atten import AirwayFormer_att_new
import  torch.nn.functional as F
from torch_geometric.loader import DataLoader
import networkx as nx
import sys
sys.path.append("../Spatial\ Encoding/")
from dataset import multitask_dataset

def loc_trachea(x):
    idx = np.argmax(x[:, 13])
    return idx

save_dir= "analysis/att_final/"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''name1 = "checkpoints/att_1_1_1_1.5_1.5_seed222_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
name2 = "checkpoints/att_1_1_1_1.5_1.5_seed333_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
name3 = "checkpoints/att_1_1_1_1.5_1.5_seed444_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
name4 = "checkpoints/att_1_1_1_1.5_1.5_seed555_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
name5 = "checkpoints/att_1_1_1_1.5_1.5_seed666_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"'''


'''name1 = "checkpoints/att_2_dense_detach_headmask_0.1_1_1_1_1_1_seed222_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
name2 = "checkpoints/att_2_dense_detach_headmask_0.1_1_1_1_1_1_seed333_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
name3 = "checkpoints/att_2_dense_detach_headmask_0.1_1_1_1_1_1_seed444_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
name4 = "checkpoints/att_2_dense_detach_headmask_0.1_1_1_1_1_1_seed555_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
name5 = "checkpoints/att_2_dense_detach_headmask_0.1_1_1_1_1_1_seed666_transformer_6layer_dim128_heads4_hdim32_mlp256_postnorm_adam5e-4_eps_hierarchy222/best.ckpt"
name = [name1,name2,name3,name4,name5]'''
#my_net = AirwayFormer_att_new(input_dim=23, num_classes1=6,num_classes2=20,num_classes3=127, dim=128,  heads=4, mlp_dim=256, dim_head=32,dropout=0.0)
train_path2 = "/home/yuy/code/data/graph_ht_pred_train_v6_v3/"
test_path2 = "/home/yuy/code/data/graph_ht_pred_test_v6_v3/"
train_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_train/"
test_path3 = "/home/yuy/code/data/graph_data_n_third_level_v3_test_pred/"
spd_train = "/home/yuy/code/transformer/Spatial Encoding/spd_train/"
spd_test = "/home/yuy/code/transformer/Spatial Encoding/spd_test/"
epochs = 800
dataset1 = multitask_dataset(train_path2, train_path3,train=True)
train_loader_case = DataLoader(dataset1, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#my_net = my_net.to(device)

file_color = open("./a.txt", 'r')
colors_file = []
for line in file_color:
    colors_file.append(line.split())
color_matrix = [[0 for _ in range(3)] for _ in range(len(colors_file)-1)]
for i in range(len(colors_file)-1):
    color_matrix[i][0] = float(colors_file[i+1][1])/255
    color_matrix[i][1] = float(colors_file[i+1][2])/255
    color_matrix[i][2] = float(colors_file[i+1][3])/255

i = 0
# np.zeros(len(case.x),dtype= int)
for case in train_loader_case:
    if case.patient[0] == "2586574": #选中第一个case
        trachea = loc_trachea(case.x)
        edge = case.edge_index
        # x = case.x.type(torch.DoubleTensor).to(device)
        x = case.x
        print(case.patient)


    i += 1


G = nx.Graph()
G.add_nodes_from(range(x.shape[0]))
G_class = nx.Graph()
G_class.add_nodes_from(range(x.shape[0]))
for j in range(edge.shape[1]):
    node1 = edge[0, j].item()
    node2 = edge[1, j].item()
    G.add_edges_from([(node1, node2)])
    G_class.add_edges_from([(node1, node2)])

colors_class = []
'''for i in range(x.shape[0]):
    colors_class.append(color_matrix[int(hypergraph[5][i].item())])'''
for i in range(x.shape[0]):
    colors_class.append(color_matrix[0])
idx = 20
colors_class[0] = color_matrix[1]
colors_class[28] = color_matrix[2]
plt.figure()
nx.draw(G, pos=nx.kamada_kawai_layout(G), node_color=colors_class, with_labels=False, node_size=50, cmap=plt.cm.summer,
        font_size=10,
        width=2.0)
plt.show()
'''for j in range(5):
    checkpoint = torch.load(name[j])
    my_net.load_state_dict(checkpoint['state_dict'])
    _, output2_1, output3_1, _,output2_2, output3_2 = my_net(x,0)
    if j == 0:
        att_map_seg_1 = my_net.give.transformer[2].layers[1][0].attentionmap[0]#第二分支的seg
        att_map_seg_2 = my_net.accecpt.transformer[2].layers[1][0].attentionmap[0]  # 第二分支的seg
    else:
        att_map_seg_1 += my_net.give.transformer[2].layers[1][0].attentionmap[0]#第二分支的seg
        att_map_seg_2 += my_net.accecpt.transformer[2].layers[1][0].attentionmap[0]  # 第二分支的seg

att_map_seg_1 = att_map_seg_1/5
att_map_seg_2 = att_map_seg_2/5'''

'''checkpoint = torch.load(name[0])
my_net.load_state_dict(checkpoint['state_dict'])
_, output2_1, output3_1, output2_2, output3_2 = my_net(x,0)

att_map_seg_1 = my_net.give.transformer[0].layers[1][0].attentionmap[0]#第二分支的seg
att_map_seg_2 = my_net.accecpt.transformer[0].layers[1][0].attentionmap[0]  # 第二分支的seg
output2_1 = output2_1.max(dim=1)[1].cpu()
output2_2 = output2_2.max(dim=1)[1].cpu()

pred2_1 = torch.where(output2_1 == y_seg,1,0)

pred2_2 = torch.where(output2_2 == y_seg,1,0)'''


# nx.draw(G, pos=nx.kamada_kawai_layout(G), labels=node_attrs, node_color=colors, node_size=100, font_size=10)

'''for i in range(x.shape[0]):
    colors_class.append(color_matrix[int(output2_2[i].item())])


plt.figure()
nx.draw(G_class, pos=nx.kamada_kawai_layout(G), with_labels=False, node_color=colors_class, node_size=50,
        font_size=10,
        width=2.0)
plt_save_class = os.path.join(save_dir, "seg_2.png")
plt.savefig(plt_save_class)
plt.close()'''




'''if pred2_1[node_num] and pred2_2[node_num]:
    path_save = os.path.join(save_dir,"r2r/")
    num_r2r += 1

if pred2_1[node_num] and (not pred2_2[node_num]):
    path_save = os.path.join(save_dir,"r2w/")
    num_r2w += 1

if (not pred2_1[node_num]) and (not pred2_2[node_num]):
    path_save = os.path.join(save_dir,"w2w/")
    num_w2w += 1

if (not pred2_1[node_num]) and pred2_2[node_num]:
    path_save = os.path.join(save_dir,"w2r/")
    num_w2r += 1
for head in range(4):
    colors_1 = att_map_seg_1[head, node_num, :].detach().cpu().numpy()
    colors_2 = att_map_seg_2[head, node_num, :].detach().cpu().numpy()
    plt.figure()
    nx.draw(G, pos=nx.kamada_kawai_layout(G), node_color=colors_1, with_labels=False, node_size=50, cmap=plt.cm.summer,
            font_size=10,
            width=2.0)
    # nx.draw(G, pos=nx.kamada_kawai_layout(G), labels=node_attrs, node_color=colors, node_size=100, font_size=10)

    plt.title("{}_subseg1_head{}".format(node_num,head))

    plt_save_1 = os.path.join(save_dir, "{}_subseg1_head{}.png".format(node_num,head))
    plt.savefig(plt_save_1, bbox_inches='tight')
    plt.close()

    plt.figure()
    nx.draw(G, pos=nx.kamada_kawai_layout(G), node_color=colors_2, with_labels=False, node_size=50,
            cmap=plt.cm.summer,
            font_size=10,
            width=2.0)
    # nx.draw(G, pos=nx.kamada_kawai_layout(G), labels=node_attrs, node_color=colors, node_size=100, font_size=10)
    plt.title("{}_subseg2_head{}".format(node_num, head))
    plt_save_2 = os.path.join(save_dir, "{}_subseg2_head{}.png".format(node_num, head))
    plt.savefig(plt_save_2, bbox_inches='tight')
    plt.close()'''

'''colors_class = []
for i in range(x.shape[0]):
    #colors_class.append(color_matrix[int(y_subseg[i].item())])
    colors_class.append(color_matrix[9])

colors_class[node_num] = color_matrix[10]'''

'''plt.figure()
nx.draw(G_class, pos=nx.kamada_kawai_layout(G), with_labels=False, node_color=colors_class, node_size=50,
        font_size=10,
        width=2.0)
plt_save_class = os.path.join(save_dir, "{}_class.png".format(node_num))
plt.savefig(plt_save_class)
plt.close()'''

#print("r2r",num_r2r,"r2w",num_r2w,"w2w",num_w2w,"w2r",num_w2r)









































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































