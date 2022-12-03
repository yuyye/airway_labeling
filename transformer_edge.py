# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:52:29 2022

@author: Yu
"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention_edgeinfo(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.dropout_dist = nn.Dropout(0.1)

    def forward(self, x, dist):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dist = self.dropout_dist(dist)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + dist

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer_postnorm_edgeinfo(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_edgeinfo(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, dist):
        for attn, ff in self.layers:
            x = attn(x, dist) + x
            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x

class AirwayFormer_codebook(nn.Module):
    def __init__(self, input_dim, num_classes1,num_classes2,num_classes3, dim, heads, mlp_dim, dim_head = 64, dropout = 0.):
        super().__init__()
        
        self.to_embedding = nn.Sequential(nn.Linear(input_dim, dim))

        hierarchy = [2, 2, 2]
        self.transformer = nn.ModuleList([])
        for d in hierarchy:
            self.transformer.append(
                Transformer_postnorm_edgeinfo(dim, d, heads, dim_head, mlp_dim, dropout)
                )

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
        
        self.spatial_pos_encoder = nn.Embedding(20, heads,padding_idx=0)  # 511

    def forward(self, x, dist):   # dist.shape: [num_nodes,] 
        x = self.to_embedding(x)
        x = x.unsqueeze(0)  
        
        dist = dist.unsqueeze(0).long()
        dist = self.spatial_pos_encoder(dist).permute(0, 3, 1, 2)
        dist_list = [torch.zeros_like(dist),dist,torch.zeros_like(dist)]

        
        x_ = []
        for i in range(len(self.transformer)):
            x = self.transformer[i](x, dist_list[i])
            x_.append(x)

        x1 = self.mlp_head1(x_[0])
        x1 = x1.squeeze(0)
        x2 = self.mlp_head2(x_[1])
        x2 = x2.squeeze(0)
        x3 = self.mlp_head3(x_[2])
        x3 = x3.squeeze(0)
        
        return x1,x2,x3