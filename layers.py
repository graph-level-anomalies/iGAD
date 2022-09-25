import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class graph_convolution_layer(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, device, bias=False):
        super(graph_convolution_layer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.device = device
        self.weight = Parameter(torch.FloatTensor(in_features, hidden_features))
        self.weight2 = Parameter(torch.FloatTensor(hidden_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(bias)
        self.mlp_layer_1 = nn.Linear(self.in_features, self.hidden_features, bias=True)
        self.mlp_layer_2 = nn.Linear(self.hidden_features, self.out_features, bias=True)
        self.relu = nn.ReLU()

    def reset_parameters(self, bias):

        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.weight2)
        if bias:
            self.bias.data.uniform_(-1, 1)

    def forward(self, adj, features):

        conv_layer_1_output = self.relu(torch.spmm(torch.spmm(adj, features), self.weight))
        conv_layer_2_output = torch.spmm(torch.spmm(adj, conv_layer_1_output), self.weight2)
        self_contribution_layer_output = self.mlp_layer_2(self.relu(self.mlp_layer_1(features)))
        outputs = torch.cat((self_contribution_layer_output, conv_layer_1_output, conv_layer_2_output), dim=1)

        return outputs