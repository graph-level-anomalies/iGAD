import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import graph_convolution_layer

class graph_convolution(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, graph_pooling_type, device):

        super(graph_convolution, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_pooling_type = graph_pooling_type
        self.device = device
        self.gc = graph_convolution_layer(self.input_dim, self.hidden_dim, self.output_dim, self.device)
        self.mlp_1 = nn.Linear(self.hidden_dim + self.output_dim*2, self.hidden_dim)
        self.mlp_2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, adj, features, graph_pool):

        h = self.gc(adj, features)
        h = self.mlp_2(F.relu(self.mlp_1(h)))
        H = torch.spmm(graph_pool, h)
        return H

