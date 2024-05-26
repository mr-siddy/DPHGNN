import torch
import torch.nn as nn
import dhg 
from dhg.structure import Hypergraph


class Hypergcn_GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True, use_bn:bool=True, drop_rate: float=0.5):
        super(Hypergcn_GraphConv, self).__init__()
        #self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias = bias)

    def forward(self, X, G):
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        X = G.smoothing_with_GCN(X)
        
        X_ = self.drop(self.act(X))
        return X_

class Clique_GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggr: str="mean", bias: bool=True, drop_rate: float=0.5,):
        super(Clique_GraphConv, self).__init__()
        self.aggr = aggr
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        if aggr == "mean":
            self.theta = nn.Linear(in_channels*2, out_channels, bias=bias)
        else:
            raise NotImplementedError()

    def forward(self, X, G):
        if self.aggr == "mean":
            X_nbr = G.v2v(X, aggr="mean")
            X = torch.cat([X, X_nbr], dim=1)
        else:
            raise NotImplementedError()

        X_ = self.theta(X)
        X_ = self.drop(self.act(X_))

        return X_

class Star_GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggr: str="mean", bias: bool=True, drop_rate: float=0.5,):
        super(Star_GraphConv, self).__init__()
        self.aggr = aggr
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        if aggr == "mean":
            self.theta = nn.Linear(in_channels*2, out_channels, bias=bias)
        else:
            raise NotImplementedError()

    def forward(self, X, G):
        if self.aggr == "mean":
            X_nbr = G.v2v(X, aggr="mean")
            X = torch.cat([X, X_nbr], dim=1)
        else:
            raise NotImplementedError()

        X_ = self.theta(X)
        X_ = self.drop(self.act(X_))

        return X_