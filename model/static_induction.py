import torch
import torch.nn as nn
import dhg
from dhg.structure import Hypergraph
from model.layers.gcn_layer import Hypergcn_GraphConv, Clique_GraphConv, Star_GraphConv

class HGSpectralNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias=True, drop_rate: float=0.3):
        super(HGSpectralNet, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        #self.theta_in = nn.Linear(in_channels, out_channels, bias= bias)
        self.theta_out = nn.Linear(in_channels*2, out_channels, bias= bias)

    def smoothing_with_HGNN(self, X, HG):
        L_HGNN = HG.L_HGNN
        X_hgnn = L_HGNN.mm(X)
        return X_hgnn

    def smoothing_with_rw(self, X, HG):
        L_rw = HG.L_rw
        X_rw = L_rw.mm(X)
        return X_rw

    def smoothing_with_sym(self, X, HG):
        L_sym = HG.L_sym
        X_sym = L_sym.mm(X)
        return X_sym

    def forward(self, X, HG):
        #X = self.theta_in(X)
        X_hgnn = self.smoothing_with_HGNN(X, HG)
        X_sym = self.smoothing_with_sym(X, HG)
        X_rw = self.smoothing_with_rw(X, HG)
        X_sym_rw = (X_sym + X_rw)/2
        X_spectral = torch.cat([X_hgnn, X_sym_rw], dim=1)
        X_ = self.drop(self.act(self.theta_out(X_spectral)))

        return X_


class CGNet(nn.Module):
    def __init__(self, in_channels:int, hid_channels:int, out_channels: int, is_last: bool=False):
        super(CGNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Clique_GraphConv(in_channels, hid_channels))
        self.layers.append(Clique_GraphConv(hid_channels, out_channels))

    def forward(self, X, G):
        for layer in self.layers:
            X = layer(X, G)
        return X

class SGNet(nn.Module):
    def __init__(self, in_channels:int, hid_channels:int, out_channels: int, is_last: bool=False):
        super(SGNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Star_GraphConv(in_channels, hid_channels))
        self.layers.append(Star_GraphConv(hid_channels, out_channels))

    def forward(self, X, G):
        for layer in self.layers:
            X = layer(X, G)
        return X

class HypGNet(nn.Module):
    def __init__(self, in_channels:int, hid_channels:int, out_channels: int, is_last: bool=False):
        super(HypGNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Hypergcn_GraphConv(in_channels, hid_channels))
        self.layers.append(Hypergcn_GraphConv(hid_channels, out_channels))

    def forward(self, X, G):
        for layer in self.layers:
            X = layer(X, G)
        return X

