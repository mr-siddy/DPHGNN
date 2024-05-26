import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import wandb
import dhg
from dhg.structure import Graph, Hypergraph
from dataset import CoRTO
from model.layers.gcn_layer import Hypergcn_GraphConv, Clique_GraphConv, Star_GraphConv
from model.DPHGNN_layer import DPHGNNConv, HypergraphConv, DPHGNN
from model.static_induction import HGSpectralNet, CGNet, HypGNet, SGNet
from model.layers.TAA_layer import TAA_msg, TAA_Spectral, TAA_Spatial, GSpectralNet
from utils.utils import set_seed
from utils.process import _format_inputs
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

#from utils.benchmark_datasets import BenchmarkDatasets


set_seed(2023)
# run_coRTO = wandb.init(
#     project="DPHGNN",
#     config={
#         "architecture": "DPHGNN",
#         "dataset": "CoRTO",
#     }
# )

data = CoRTO(data_root="/home/siddy/DPHGNN/data_root/")

#device = torch.device("cuda" if torch.cuda.is_available else "cpu")
device = torch.device("cpu")
X, lbl = data["features"], data["labels"]
print(X.shape, lbl.shape)
HG = Hypergraph(data["num_vertices"], data["edge_list"])

train_mask = data["train_mask"]
test_mask = data["test_mask"]
val_mask = data["val_mask"]

spectral_layer = HGSpectralNet(
    in_channels=10,
    out_channels=64
).to(device)


dphgnn_layer = DPHGNN(
    in_channels=10,
    hid_channels= 64,
    out_channels=2
).to(device)

HypG_layer = HypGNet(in_channels=10, hid_channels= 64, out_channels= 64).to(device)
CG_layer = CGNet(in_channels= 10, hid_channels=  64, out_channels=64).to(device)
SG_layer = SGNet(in_channels= 10, hid_channels=64, out_channels=64).to(device)
TAA_Spatial_layer = TAA_Spatial(in_dim=10, num_heads=1).to(device)
TAA_Spectral_layer = TAA_Spectral(in_dim=10, num_heads=1).to(device)
GSpectralNet_layer = GSpectralNet(in_channels=10, out_channels=10).to(device)
FC_layer = nn.Linear(in_features=140, out_features=10).to(device)


optimizer = torch.optim.Adam(
    [
        {'params': dphgnn_layer.parameters()},
        {'params': spectral_layer.parameters(), 'lr':0.01, 'weight_decay':0.0005},
        {'params': HypG_layer.parameters(), 'lr':0.1, 'weight_decay':5e-5},
        {'params': CG_layer.parameters(), 'lr':0.1, 'weight_decay':5e-5},
        {'params': SG_layer.parameters(), 'lr':0.1, 'weight_decay':5e-5},
        {'params': TAA_Spatial_layer.parameters(), 'lr':0.001, 'weight_decay':0.001},
        {'params': TAA_Spectral_layer.parameters(), 'lr':0.001, 'weight_decay':0.001}

    ], lr=0.01, weight_decay=5e-4
)


def train(data, optimizer, epoch):
    X = data["features"]
    HG = Hypergraph(data["num_vertices"], data["edge_list"])
    train_idx = data["train_mask"]
    spectral_layer.train()
    dphgnn_layer.train()
    HypG_layer.train()
    CG_layer.train()
    SG_layer.train()
    TAA_Spatial_layer.train()
    TAA_Spectral_layer.train()
    GSpectralNet_layer.train()
    st = time.time()
    taa_features, _, S_features = TAA_msg(data)
    spectral_features = spectral_layer(X, HG) #[66790, 10]->[66790, 64]
    X = torch.cat([taa_features, spectral_features], dim=1) #[66790, 64+76=140]
    X = FC_layer(X)#[66790,140-> 10]
    outs = dphgnn_layer(X, HG, S_features) # [66790, 10]       S_star: [27528, 64]
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    #wandb.log({"loss": loss.item()})
    return loss.item()

@torch.no_grad()
def infer(data, test=False):
    X = data["features"]
    HG = Hypergraph(data["num_vertices"], data["edge_list"])
    idx = data["val_mask"]
    spectral_layer.eval()
    dphgnn_layer.eval()
    HypG_layer.eval()
    CG_layer.eval()
    SG_layer.eval()
    TAA_Spatial_layer.eval()
    TAA_Spectral_layer.eval()
    GSpectralNet_layer.eval()
    st = time.time()
    taa_features, _, S_features = TAA_msg(data)
    spectral_features = spectral_layer(X, HG) #[66790, 10]->[66790, 64]
    X = torch.cat([taa_features, spectral_features], dim=1) #[66790, 64+76=140]
    X = FC_layer(X)#[66790,140-> 10]
    outs = dphgnn_layer(X, HG, S_features) # [66790, 10]       S_star: [27528, 64]
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res, outs

if __name__ == "__main__":
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data = CoRTO(data_root="/data_root")
    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(400):
        loss = train(data, optimizer, epoch)
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res, _ = infer(data, test=False)
                #wandb.log({"val_res": val_res})
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                #wandb.log({"best_val": best_val})
                best_state = (net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    print("test...")
    net.load_state_dict(best_state)
    res, outs = infer(X, HG, lbl, test_mask, test=True)
    #wandb.log({"test_res": res})
    print(f"final result: epoch: {best_epoch}")
    print(res)