# -*- coding: utf-8 -*-
"""
Created on Thu Dec 09 16:36:29 2022

@author: yuy

the attention map : subseg -> seg
"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


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


class Attention(nn.Module):
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

    def forward(self, x, p):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

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

class Attention_att(nn.Module):
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

    def forward(self, x, att,p,alpha):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        #dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + att
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        '''
        dots = torch.matmul(q, k.transpose(-1, -2))
        #矩阵归一化        
        dots = F.normalize(att, p=2, dim=-1)*torch.norm(dots, p=2, dim=-1,keepdim=True) + dots
        '''


        attn = alpha*self.attend(dots)+(1-alpha)*self.attend(att)

        self.attentionmap = attn
        attn = self.dropout(attn)

        mask = torch.ones_like(attn)
        a = np.random.binomial(1, p, size=mask.shape[1])
        for i in range(mask.shape[1]):
            if a[i] == 1:
                mask[:,i,:,:] = 0

        attn = attn*mask

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention_give(nn.Module):
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

    def forward(self, x, p):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        self.attentionmap = attn
        attn = self.dropout(attn)

        mask = torch.ones_like(attn)
        a = np.random.binomial(1, p, size=mask.shape[1])
        for i in range(mask.shape[1]):
            if a[i] == 1:
                mask[:,i,:,:] = 0

        attn = attn*mask

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out),dots




class Transformer_postnorm_att(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_att(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x,att,p,alpha):
        for attn, ff in self.layers:
            x = attn(x,att,p,alpha) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x

class Transformer_postnorm(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x,p):
        for attn, ff in self.layers:
            x = attn(x,p) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x

class Transformer_postnorm_give(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth ):
            self.layers.append(nn.ModuleList([
                Attention_give(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))


    def forward(self, x,p):
        for attn, ff in self.layers:

            attn,give = attn(x,p)
            x = attn + x

            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x, give


class AirwayFormer_give(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()


        hierarchy = [2, 2]
        self.transformer = nn.ModuleList([])
        self.dense_linear = nn.ModuleList(
            [nn.Linear((i+2) * dim, dim) for i in range(2)]
        )
        layer_num = 0
        for d in hierarchy:
            if layer_num == 1:
                self.transformer.append(
                    Transformer_postnorm_give(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            else:
                self.transformer.append(
                    Transformer_postnorm(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            layer_num += 1

    def forward(self,out_list,p):

        x_ = []

        list = []
        list.append(out_list[0])
        list.append(out_list[1])
        for i in range(len(self.transformer)):
            x = self.dense_linear[i](torch.cat(list, dim=-1))
            if i == 1:
                x, give = self.transformer[i](x,p)
            else:
                x = self.transformer[i](x,p)
            x_.append(x)
            list.append(x)

        return x_[0], x_[1],give.detach()

class AirwayFormer_accept(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()



        hierarchy = [2, 2]
        self.transformer = nn.ModuleList([])
        self.dense_linear = nn.ModuleList(
            [nn.Linear(dim + (i + 1) * dim, dim) for i in range(len(hierarchy))]
        )
        layer_num = 0
        for d in hierarchy:
            if layer_num == 0:
                self.transformer.append(
                    Transformer_postnorm_att(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            else:
                self.transformer.append(
                    Transformer_postnorm(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            layer_num += 1


    def forward(self, out_list,att,p):
        x_ = []
        list = []
        list.append(out_list[0])
        list.append(out_list[1])

        for i in range(len(self.transformer)):
            x = self.dense_linear[i](torch.cat(list, dim=-1))
            if i == 0:
                x = self.transformer[i](x,att,p)
            else:
                x = self.transformer[i](x,p)
            x_.append(x)
            list.append(x)



        #return x1, x2, x3

        return x_[0],x_[1]


class AirwayFormer_att(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()

        self.accecpt = AirwayFormer_accept(input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.)

        self.give = AirwayFormer_give(input_dim, num_classes1, num_classes2, num_classes3, dim, heads,
                                           mlp_dim, dim_head=64,
                                           dropout=0., emb_dropout=0.)
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
        self.lob = Transformer_postnorm(dim, 2, heads, dim_head, mlp_dim, dropout)
        self.to_embedding = nn.Sequential(nn.Linear(input_dim, dim))


    def forward(self, x,p):
        in_list = []
        x = self.to_embedding(x)
        x = x.unsqueeze(0)
        in_list.append(x)
        x = self.lob(x,p)
        in_list.append(x)
        x1_1 = x
        x2_1, x3_1, att = self.give(in_list,p)
        x2_2, x3_2 = self.accecpt(in_list,att,p)

        x1_1 = self.mlp_head1(x1_1)
        x1_1 = x1_1.squeeze(0)
        x2_1 = self.mlp_head2(x2_1)
        x2_1 = x2_1.squeeze(0)
        x3_1 = self.mlp_head3(x3_1)
        x3_1= x3_1.squeeze(0)

        x2_2 = self.mlp_head2(x2_2)
        x2_2 = x2_2.squeeze(0)
        x3_2 = self.mlp_head3(x3_2)
        x3_2= x3_2.squeeze(0)



        return x1_1, x2_1, x3_1, x2_2, x3_2



class AirwayFormer_give_new(nn.Module):
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
                    Transformer_postnorm_give(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            else:
                self.transformer.append(
                    Transformer_postnorm(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            layer_num += 1

    def forward(self,x,p):

        x_ = []

        list = []
        give_list = []
        list.append(x)

        for i in range(len(self.transformer)):
            x = self.dense_linear[i](torch.cat(list, dim=-1))
            if i == 0:
                x = self.transformer[i](x, p)
            else:
                x, give = self.transformer[i](x, p)
                give_list.append(give.detach())
            x_.append(x)
            list.append(x)

        return x_[0], x_[1],x_[2],give_list

class AirwayFormer_accept_new(nn.Module):
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
                    Transformer_postnorm(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            else:
                self.transformer.append(
                    Transformer_postnorm_att(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            layer_num += 1


    def forward(self, x,att,p,alpha):
        x_ = []
        list = []
        list.append(x)


        for i in range(len(self.transformer)):
            x = self.dense_linear[i](torch.cat(list, dim=-1))
            if i == 2:
                x = self.transformer[i](x, p)
            else:
                x = self.transformer[i](x, att[i], p,alpha)
            x_.append(x)
            list.append(x)



        #return x1, x2, x3

        return x_[0],x_[1],x_[2]


class AirwayFormer_att_new(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.,alpha=0.):
        super().__init__()

        self.accecpt = AirwayFormer_accept_new(input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.)

        self.give = AirwayFormer_give_new(input_dim, num_classes1, num_classes2, num_classes3, dim, heads,
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


    def forward(self, x,p):
        x = self.to_embedding(x)
        x = x.unsqueeze(0)


        x1_1, x2_1, x3_1, att = self.give(x,p)
        x1_2, x2_2, x3_2 = self.accecpt(x,att,p,self.alpha)
        print("logits!")
        '''x1_1 = self.mlp_head1(x1_1)
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
        x3_2= x3_2.squeeze(0)'''



        return x,x1_1, x2_1, x3_1, x1_2,x2_2, x3_2


