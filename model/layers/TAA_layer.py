import torch
import torch.nn as nn
import dhg
from dhg.structure import Graph, Hypergraph
from model.static_induction import HypGNet, CGNet, SGNet
from utils.utils import random_noise
import torch
import torch.nn as nn

#device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

HypG_layer = HypGNet(in_channels=10, hid_channels= 64, out_channels= 64).to(device)
CG_layer = CGNet(in_channels= 10, hid_channels=  64, out_channels=64).to(device)
SG_layer = SGNet(in_channels= 10, hid_channels=64, out_channels=64).to(device)

class TAA_Spatial(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(TAA_Spatial, self).__init__()
        self.num_heads = num_heads
        self.hyp_proj = nn.Linear(in_dim, 10)
        self.star_proj = nn.Linear(in_dim, 10)
        self.clique_proj = nn.Linear(in_dim, 10)
        self.attention = nn.MultiheadAttention(10, num_heads)

    def forward(self, X_hyp, X_star, X_clique):
        hyp_proj = self.hyp_proj(X_hyp)
        star_proj = self.star_proj(X_star)
        clique_proj = self.clique_proj(X_clique)

        spatial_attn_output, _ = self.attention(hyp_proj, star_proj, clique_proj)
        return spatial_attn_output

class TAA_Spectral(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(TAA_Spectral, self).__init__()
        self.num_heads = num_heads
        self.hyp_proj = nn.Linear(in_dim, 10)
        self.star_proj = nn.Linear(in_dim, 10)
        self.clique_proj = nn.Linear(in_dim, 10)
        self.attention = nn.MultiheadAttention(10, num_heads)

    def forward(self, X_L_hyp, X_L_star, X_L_clique):
        hyp_proj = self.hyp_proj(X_L_hyp)
        star_proj = self.star_proj(X_L_star)
        clique_proj = self.clique_proj(X_L_clique)

        spectral_attn_output, _ = self.attention(hyp_proj, star_proj, clique_proj)
        return spectral_attn_output

class GSpectralNet(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GSpectralNet, self).__init__()
        self.theta_hyp = nn.Linear(in_channels, out_channels, bias = bias)
        self.theta_cg = nn.Linear(in_channels, out_channels, bias = bias)
        self.theta_sg = nn.Linear(in_channels, out_channels, bias = bias)

    def laplacian_smoothing(self, G, X):
        return G.smoothing_with_GCN(X)

    def forward(self, X_HypGNet, X_CGNet, X_SGNet, G_hyp, G_c, G_s):
        X_hyp = self.theta_hyp(X_HypGNet)
        X_cg = self.theta_cg(X_CGNet)
        X_sg = self.theta_sg(X_SGNet)
        X_hyp_s = self.laplacian_smoothing(G_hyp, X_hyp)
        X_cg_s = self.laplacian_smoothing(G_c, X_cg)
        X_sg_s = self.laplacian_smoothing(G_s, X_sg)
        return X_hyp_s, X_cg_s, X_sg_s

def get_star_features(X):
    s_init = torch.zeros(27528, 10).to(device)
    X_star = torch.cat((X, s_init), 0)
    return X_star

TAA_Spatial_layer = TAA_Spatial(in_dim=64, num_heads=1).to(device)
TAA_Spectral_layer = TAA_Spectral(in_dim=10, num_heads=1).to(device)
GSpectralNet_layer = GSpectralNet(in_channels=64, out_channels=10).to(device)

def TAA_msg(data): 
    HG = Hypergraph(data["num_vertices"], data["edge_list"])
    X, lbl = data["features"].to(device), data["labels"].to(device) # X: [66790, 10], lbl: [66790]
    G_clique = Graph.from_hypergraph_clique(HG)
    G_hypergcn = Graph.from_hypergraph_hypergcn(HG, X)
    G_star, v_mask = Graph.from_hypergraph_star(HG)
    X_star = get_star_features(X) # X_star: [94318, 10], 66790+27528
    clique_graph_features = CG_layer(X, G_clique) # 10->64 [66790, 64]
    star_graph_features = SG_layer(X_star, G_star) # 10->64, [94318, 64]
    X_star, S_star = star_graph_features[v_mask], star_graph_features[~v_mask] # X_star: [66790, 64], S_star: [27528, 64]
    hypergcn_graph_features = HypG_layer(X, G_hypergcn) # 10->64 [66790, 64]
    noise = random_noise(lbl).to(device) # [66790,2]
    X_hyp, X_cg, X_sg = GSpectralNet_layer(hypergcn_graph_features, clique_graph_features, X_star, G_hypergcn, G_clique, G_hypergcn) # [66790, 10]
    X_spatial = TAA_Spatial_layer(hypergcn_graph_features, clique_graph_features, X_star) # [66790, 64]
    X_spectral = TAA_Spectral_layer(X_hyp, X_cg, X_sg) # [66790, 10]
    #taa_features = X_spatial+X_spectral+noise # errr
    taa_features = torch.cat((X_spatial, X_spectral, noise), 1) # [66790, 76]
    return taa_features, X_star, S_star
