import time
import pandas as pd
import numpy as np

from typing import Optional
from functools import partial

import torch
import torch.optim as optim
import dhg
from dhg import Graph, Hypergraph
from dhg.models import UniGIN
from dhg.random import set_seed
from dhg.datapipe import (load_from_pickle, 
                          norm_ft, 
                          to_tensor, 
                          to_long_tensor, 
                          to_bool_tensor,
)
from dhg.data import BaseData
from dataset import CoRTO
from utils.utils import set_seed
from dhg.metrics import HyperGraphVertexClassificationEvaluator as Evaluator

def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()
    
    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st: .5f}s, Loss: {loss.item(): .5f}")
    wandb.log({"loss": loss.item()})
    return loss.item()

@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res


if __name__ == "__main__":
    set_seed(2023)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(
            [
                "accuracy",
                "f1_score",
                {"f1_score": {"average": "micro"}}
            ]
        )

    data = CoRTO(data_root="/data_root")
    X, lbl =  data["features"], data["labels"]
    G = Hypergraph(data["num_vertices"], data["edge_list"])

    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    net = UniGIN(X.shape[1], 32, data["num_classes"], use_bn=True, drop_rate=0.25)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay = 5e-4)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0

    for epoch in range(50):
            #train
        train(net, X, G, lbl, train_mask, optimizer, epoch)
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask)
                #wandb.log({"val_res": val_res})
            if val_res > best_val:
                print(f"update best: {val_res: .5f}")
                best_epoch = epoch
                best_val = val_res
                #wandb.log({"best_val": best_val})
                best_state = deepcopy(net.state_dict())

    print("\ntrain finished")
    print(f"best val: {best_val: .5f}")

    #test
    print("test..")
    net.load_state_dict(best_state)
    res = infer(net, X, G, lbl, test_mask, test=True)
    #wandb.log({"test_res": res})
    print(f"final result: epoch: {best_epoch}")
    print(res)
