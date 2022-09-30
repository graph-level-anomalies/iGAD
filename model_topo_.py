import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class RW_GNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_subgraphs, size_subgraph, max_step, normalize, dropout, device):
        super(RW_GNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_subgraphs = n_subgraphs
        self.size_subgraphs = size_subgraph
        self.max_step = max_step
        self.device = device
        self.normalize = normalize


        self.Theta_matrix = Parameter(
            torch.FloatTensor(self.n_subgraphs, self.size_subgraphs * (self.size_subgraphs - 1) // 2, 1))  # random walk


        self.bn = nn.BatchNorm1d(self.n_subgraphs * (self.max_step-1))
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

        self.layer_1 = nn.Linear(self.n_subgraphs*(self.max_step-1), self.hidden_dim, bias= True)
        self.layer_2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.relu = nn.ReLU()


    def init_weights(self):
        #self.Theta_matrix.data.uniform_(-1, 1)
        nn.init.kaiming_normal_(self.Theta_matrix)

    def forward(self, adj, graph_indicator):
        sampled_matrix = self.relu(self.Theta_matrix)
        sampled_matrix = sampled_matrix[:, :, 0]

        # transform
        adj_sampled = torch.zeros(self.n_subgraphs, self.size_subgraphs, self.size_subgraphs).to(self.device)
        idx = torch.triu_indices(self.size_subgraphs, self.size_subgraphs, offset=1)
        adj_sampled[:, idx[0], idx[1]] = sampled_matrix
        adj_sampled = adj_sampled + torch.transpose(adj_sampled, 1, 2)

        # random walks
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.size(0)
        n_nodes = adj.shape[0]

        if self.normalize:
            norm = counts.unsqueeze(1).repeat(1, self.n_subgraphs) 

        E = torch.ones((self.n_subgraphs, self.size_subgraphs, n_nodes), device=self.device)

        I = torch.eye(n_nodes, device=self.device)
        adj_power = adj
        P_power_E = E
        random_walk_results = list()

        for i in range(1, self.max_step):

            I = torch.spmm(adj_power, I)
            P_power_E = torch.einsum("abc,acd->abd", (adj_sampled, P_power_E))
            res = torch.einsum("abc,cd->abd", (P_power_E, I))
            res = torch.mul(E, res)
            #the number of subgraph * the nodes number in subgraph * the number of nodes in this batch

            res = torch.zeros(res.size(0), res.size(1), n_graphs, device=self.device).index_add_(2, graph_indicator, res)
            res = torch.sum(res, dim=1)
            res = torch.transpose(res, 0, 1)

            if self.normalize:
                res /= norm
            random_walk_results.append(res)


        random_walk_results = torch.cat(random_walk_results, dim=1)
        random_walk_results = self.bn(random_walk_results)

        random_walk_results = self.layer_2(self.relu(self.layer_1(random_walk_results)))


        return random_walk_results
