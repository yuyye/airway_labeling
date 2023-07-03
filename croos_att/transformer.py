# -*- coding: utf-8 -*-
"""
Created on Thu Dec 09 16:36:29 2022

@author: yuy

the attention map : subseg -> seg
简化了transformer_att_new和transformer_att_Gres
"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn.functional as F


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention_spd(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, spd,p):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + spd
        #attn = self.attend(dots + self.attend(dots/10000).detach() * spd)
        #print(torch.median(self.attend(dots)))
        #attn = self.attend(dots + F.normalize(dots * spd, p=2, dim=-1) * torch.norm(dots, p=2, dim=-1, keepdim=True))
        #dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale).detach() * spd

        attn = self.attend(dots)

        self.attentionmap = attn
        attn = self.dropout(attn)

        mask = torch.ones_like(attn,requires_grad=False)
        a = np.random.binomial(1, 1-p, size=mask.shape[1])
        while np.sum(a) == 0:
            a = np.random.binomial(1, 1 - p, size=mask.shape[1])
        for i in range(mask.shape[1]):
            if a[i] == 0:
                mask[:,i,:,:] = 0

        attn = attn*mask*mask.shape[1]/np.sum(a) #normalization

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention_cross(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1,x2,spd,p):
        q = self.to_q(x1)
        q = rearrange(q,'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(x2).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + spd
        attn = self.attend(dots)
        attn = self.dropout(attn)


        mask = torch.ones_like(attn, requires_grad=False)
        a = np.random.binomial(1, 1 - p, size=mask.shape[1])
        while np.sum(a) == 0:
            a = np.random.binomial(1, 1 - p, size=mask.shape[1])
        for i in range(mask.shape[1]):
            if a[i] == 0:
                mask[:, i, :, :] = 0

        attn = attn * mask * mask.shape[1] / np.sum(a)  # normalization

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)





class Transformer_postnorm_cross_spd(nn.Module):#use 分支2接收att
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_spd(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        self.cross = Attention_cross(dim, heads=heads, dim_head=dim_head, dropout=dropout)

    def forward(self, x,spd,x2,p):
        for attn, ff in self.layers:
            x = attn(x,spd,p) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
            x = self.cross(x,x2,spd,p) + x
            x = self.norm(x)
        return x

class Transformer_postnorm_spd(nn.Module):#use 普通
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_spd(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x,spd,p):
        for attn, ff in self.layers:
            x = attn(x,spd,p) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x

class Transformer_postnorm_give_spd(nn.Module):#use 分支1
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth ):
            self.layers.append(nn.ModuleList([
                Attention_spd(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))


    def forward(self, x,spd,p):
        for attn, ff in self.layers:

            attn = attn(x,spd,p)
            x = attn + x

            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x




class AirwayFormer_give_spd(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()


        hierarchy = [2, 2, 2]
        self.transformer = nn.ModuleList([])
        self.dense_linear = nn.ModuleList(
            [nn.Linear((i+1) * dim, dim) for i in range(len(hierarchy))]
        )
        layer_num = 0
        for d in hierarchy:
            if layer_num != 0:
                self.transformer.append(
                    Transformer_postnorm_give_spd(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            else:
                self.transformer.append(
                    Transformer_postnorm_spd(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            layer_num += 1

    def forward(self,x,spd,p):

        x_ = []

        list = []
        list.append(x)
        give_list = []

        for i in range(len(self.transformer)):
            x = self.dense_linear[i](torch.cat(list, dim=-1))
            '''if i == 0:
                x = self.transformer[i](x, spd[i],p)
            else:'''
            x = self.transformer[i](x,spd[i],p)
            #give_list.append(give)#detach
            x_.append(x)
            list.append(x)

        return x_[0], x_[1],x_[2]


class AirwayFormer_accept_spd(nn.Module):#use
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()



        hierarchy = [2, 2,2]
        self.transformer = nn.ModuleList([])
        self.dense_linear = nn.ModuleList(
            [nn.Linear((i + 1) * dim, dim) for i in range(len(hierarchy))]
        )
        layer_num = 0
        for d in hierarchy:
            if layer_num == 2:
                self.transformer.append(
                    Transformer_postnorm_spd(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            else:
                self.transformer.append(
                    Transformer_postnorm_cross_spd(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            layer_num += 1


    def forward(self, x,spd,x2,p):
        x_ = []
        list = []

        list.append(x)
        for i in range(len(self.transformer)):
            x = self.dense_linear[i](torch.cat(list, dim=-1))
            if i == 2:
                x = self.transformer[i](x, spd[i],p)
            else:
                x = self.transformer[i](x, spd[i],x2[i], p)
            x_.append(x)
            list.append(x)

        return x_[0],x_[1],x_[2]
class AirwayFormer_att_se(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.,alpha=0.):
        super().__init__()

        self.accecpt = AirwayFormer_accept_spd(input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.)

        self.give = AirwayFormer_give_spd(input_dim, num_classes1, num_classes2, num_classes3, dim, heads,
                                           mlp_dim, dim_head=64,
                                           dropout=0., emb_dropout=0.,)
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )
        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes2)
        )
        self.mlp_head3 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes3)
        )
        self.to_embedding = nn.Sequential(nn.Linear(input_dim, dim))
        self.alpha = alpha
        self.spatial_pos_encoder1 = nn.Embedding(30, heads, padding_idx=0)
        self.spatial_pos_encoder2 = nn.Embedding(30, heads, padding_idx=0)
        self.spatial_pos_encoder3 = nn.Embedding(30, heads, padding_idx=0)
        '''self.spatial_pos_encoder4 = nn.Embedding(50, heads, padding_idx=0)
        self.spatial_pos_encoder5 = nn.Embedding(50, heads, padding_idx=0)
        self.spatial_pos_encoder6 = nn.Embedding(50, heads, padding_idx=0)'''



    def forward(self,x,spd,p):
        x = self.to_embedding(x)
        x = x.unsqueeze(0)
        dict = []
        spd1 = spd.unsqueeze(0)
        spd1 = self.spatial_pos_encoder1(spd1).permute(0, 3, 1, 2)
        dict.append(spd1)
        spd2 = spd.unsqueeze(0)
        spd2 = self.spatial_pos_encoder2(spd2).permute(0, 3, 1, 2)
        dict.append(spd2)
        spd3 = spd.unsqueeze(0)
        spd3 = self.spatial_pos_encoder3(spd3).permute(0, 3, 1, 2)
        dict.append(spd3)

        '''dict2 = []
        spd4 = spd.unsqueeze(0)
        spd4 = self.spatial_pos_encoder4(spd4).permute(0, 3, 1, 2)
        dict2.append(spd4)
        spd5 = spd.unsqueeze(0)
        spd5 = self.spatial_pos_encoder5(spd5).permute(0, 3, 1, 2)
        dict2.append(spd5)
        spd6 = spd.unsqueeze(0)
        spd6 = self.spatial_pos_encoder6(spd6).permute(0, 3, 1, 2)
        dict2.append(spd6)'''

        x1_1, x2_1, x3_1 = self.give(x,dict,p)
        x2 = [x2_1,x3_1]
        x1_2, x2_2, x3_2 = self.accecpt(x,dict,x2,p)

        x1_1 = self.mlp_head1(x1_1)
        x1_1 = x1_1.squeeze(0)
        x2_1 = self.mlp_head2(x2_1)
        x2_1 = x2_1.squeeze(0)
        x3_1 = self.mlp_head3(x3_1)
        x3_1= x3_1.squeeze(0)
        x1_2 = self.mlp_head1(x1_2)
        x1_2 = x1_2.squeeze(0)
        x2_2 = self.mlp_head2(x2_2)
        x2_2 = x2_2.squeeze(0)
        x3_2 = self.mlp_head3(x3_2)
        x3_2= x3_2.squeeze(0)
        return x1_1, x2_1, x3_1, x1_2,x2_2, x3_2







