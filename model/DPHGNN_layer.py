import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import dhg
from dhg.structure.graphs import Graph
from dhg.structure.hypergraphs import Hypergraph
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from utils.utils import random_noise, eigen_features
from dataset import CoRTO
from model.static_induction import HGSpectralNet, CGNet, SGNet, HypGNet
from model.layers.TAA_layer import TAA_msg
initializer = init.xavier_uniform_

class DPHGNNConv(nn.Module):
  def __init__(self,
               in_channels: 10,
               out_channels: 64,
               bias: bool=True,
               use_bn: bool=False,
               drop_rate: float=0.5,
               atten_neg_slope: float=0.2,
               is_last: bool=False,
               ):
    super(DPHGNNConv, self).__init__()
    self.is_last = is_last
    self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
    self.act = nn.ReLU(inplace=True)
    self.drop = nn.Dropout(drop_rate)
    self.theta_x = nn.Linear(in_channels, out_channels, bias=bias)
    self.theta_v2e_group = nn.Linear(in_channels, out_channels, bias=bias)
    self.theta_e2v = nn.Linear(74, out_channels, bias = bias)
    self.atten_dropout = nn.Dropout(drop_rate)
    self.atten_act = nn.LeakyReLU(atten_neg_slope)
    self.act = nn.ELU(inplace=True)
    self.theta_vertex = nn.Linear(in_channels, out_channels, bias=bias)
    self.atten_vertex = nn.Linear(out_channels, 1, bias=False)

  def forward(self, X, HG, S_features):
    X_init = self.theta_x(X)
    X_feat = self.theta_vertex(X)
    x_for_vertex = self.atten_vertex(X_feat)
    v2e_atten_score = x_for_vertex[HG.v2e_src]
    v2e_atten_score = self.atten_dropout(self.atten_act(v2e_atten_score).squeeze())
    X_feat_group = self.theta_v2e_group(X)
    if self.bn is not None:
      X_feat = self.bn(X_feat)
      X_feat_group = self.bn(X_feat_group)
    Y_v2e = self.act(HG.v2e(X_feat, aggr="softmax_then_sum", v2e_weight=v2e_atten_score))
    print(Y_v2e.shape)
    Y_v2e_group = self.act(HG.v2e_aggregation_of_group(group_name='main', X= X_feat_group, aggr="mean"))
    msg = torch.cat((Y_v2e, S_features), dim=1) # ([27528, 10], [27528, 64] -> [27528, 74])
    Y = self.theta_e2v(msg)
    X = HG.e2v(Y, aggr="mean") #[27528, 64]-> [66790, 64]
    if not self.is_last:
      X = self.drop(self.act(X))
    return X + X_init #[66790, 64]

class HypergraphConv(nn.Module):
    def __init__(
        self,
        in_channels: 64,
        out_channels: 2,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X, HG, S_features):
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        Y = hg.v2e(X, aggr="mean")
        _De = torch.zeros(hg.num_e, device=hg.device)
        _De = _De.scatter_reduce(0, index=hg.v2e_dst, src=hg.D_v.clone()._values()[hg.v2e_src], reduce="mean")
        _De = _De.pow(-0.5)
        _De[_De.isinf()] = 1
        Y = _De.view(-1, 1) * Y
        X = hg.e2v(Y, aggr="sum")
        X = torch.sparse.mm(hg.D_v_neg_1_2, X)
        if not self.is_last:
            X = self.drop(self.act(X))
        return X

class DPHGNN(nn.Module):
    def __init__(self, in_channels:int, hid_channels:int, out_channels: int, is_last: bool=False):
        super(DPHGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(DPHGNNConv(in_channels, hid_channels, is_last=False))
        self.layers.append(HypergraphConv(hid_channels, out_channels, is_last=True))

    def forward(self, X, HG, S_features):
        for layer in self.layers:
            X = layer(X, HG, S_features)
        return X



