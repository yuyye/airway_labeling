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

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        self.attentionmap = attn
        attn = self.dropout(attn)

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

    def forward(self, x, att):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + att

        attn = self.attend(dots)

        self.attentionmap = attn
        attn = self.dropout(attn)

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

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        self.attentionmap = attn
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out),dots

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

    def forward(self, x, att):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + att

        attn = self.attend(dots)

        self.attentionmap = attn
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



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

    def forward(self, x,att):
        for attn, ff in self.layers:
            x = attn(x,att) + x
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

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
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


    def forward(self, x):
        for attn, ff in self.layers:

            attn,give = attn(x)
            x = attn + x

            x = self.norm(x)
            x = ff(x) + x
            x = self.norm(x)
        return x, give

class AirwayFormer_hierarchy(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, depth, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()

        self.to_embedding = nn.Sequential(nn.Linear(input_dim, dim))

        hierarchy = [2, 2, 2]
        self.transformer = nn.ModuleList([])
        for d in hierarchy:
            self.transformer.append(
                Transformer_postnorm(dim, d, heads, dim_head, mlp_dim, dropout)
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

    def forward(self, x):
        x = self.to_embedding(x)
        x = x.unsqueeze(0)

        x_ = []
        for former in self.transformer:
            x = former(x)
            x_.append(x)

        x1 = self.mlp_head1(x_[0])
        x1 = x1.squeeze(0)
        x2 = self.mlp_head2(x_[1])
        x2 = x2.squeeze(0)
        x3 = self.mlp_head3(x_[2])
        x3 = x3.squeeze(0)

        return x1, x2, x3

class AirwayFormer_give(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()

        self.to_embedding = nn.Sequential(nn.Linear(input_dim, dim))

        hierarchy = [2, 2, 2]
        self.transformer = nn.ModuleList([])
        layer_num = 0
        for d in hierarchy:
            if layer_num == 2:
                self.transformer.append(
                    Transformer_postnorm_give(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            else:
                self.transformer.append(
                    Transformer_postnorm(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            layer_num += 1
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

    def forward(self, x):
        x = self.to_embedding(x)
        x = x.unsqueeze(0)

        x_ = []
        layer_num = 0
        for former in self.transformer:
            if layer_num == 2:
                x, give = former(x)
            else:
                x = former(x)
            layer_num += 1
            x_.append(x)

        x1 = self.mlp_head1(x_[0])
        x1 = x1.squeeze(0)
        x2 = self.mlp_head2(x_[1])
        x2 = x2.squeeze(0)
        x3 = self.mlp_head3(x_[2])
        x3 = x3.squeeze(0)

        return x1, x2, x3,give

class AirwayFormer_accept(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()

        self.to_embedding = nn.Sequential(nn.Linear(input_dim, dim))

        hierarchy = [2, 2, 2]
        self.transformer = nn.ModuleList([])
        layer_num = 0
        for d in hierarchy:
            if layer_num == 1:
                self.transformer.append(
                    Transformer_postnorm_att(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            else:
                self.transformer.append(
                    Transformer_postnorm(dim, d, heads, dim_head, mlp_dim, dropout)
                )
            layer_num += 1

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

    def forward(self, x,att):
        x = self.to_embedding(x)
        x = x.unsqueeze(0)

        x_ = []
        layer_num = 0
        for former in self.transformer:
            if layer_num == 1:
                x = former(x,att)
            else:
                x = former(x)
            layer_num += 1
            x_.append(x)

        x1 = self.mlp_head1(x_[0])
        x1 = x1.squeeze(0)
        x2 = self.mlp_head2(x_[1])
        x2 = x2.squeeze(0)
        x3 = self.mlp_head3(x_[2])
        x3 = x3.squeeze(0)

        return x1, x2, x3

class AirwayFormer_att(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()

        self.accecpt = AirwayFormer_accept(input_dim, num_classes1, num_classes2, num_classes3, dim, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.)

        self.give = AirwayFormer_give(input_dim, num_classes1, num_classes2, num_classes3, dim, heads,
                                           mlp_dim, dim_head=64,
                                           dropout=0., emb_dropout=0.)


    def forward(self, x):
        x1_1, x2_1, x3_1, att = self.give(x)
        x1_2, x2_2, x3_2 = self.accecpt(x,att)



        return x1_1, x2_1, x3_1,x1_2, x2_2, x3_2





