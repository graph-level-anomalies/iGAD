import numpy as np
import torch
from math import ceil
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, precision_score,classification_report


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset):

    graph_indicator = np.loadtxt("datasets/%s/%s_graph_indicator.txt"%(dataset, dataset), dtype=np.int64)
    # the value in the i-th line is the graph_id of the node with node_id i
    edges = np.loadtxt("datasets/%s/%s_A.txt"%(dataset, dataset), dtype=np.int64, delimiter="," )
    # each line correspond to (row, col) resp. (node_id, node_id)
    edges -= 1
    _, graph_size = np.unique(graph_indicator, return_counts=True)
    # _:graph_idx, graph_size:the number of node of a graph
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(graph_indicator.size, graph_indicator.size))#.todense()
    X = np.loadtxt("datasets/%s/%s_node_labels.txt" % (dataset, dataset), dtype=np.int64).reshape(-1, 1)
    # the value in the i-th line corresponds to the node with node_id i
    enc = OneHotEncoder(sparse=False)
    X = enc.fit_transform(X)

    adj = []
    features = []
    start_idx = 0

    for i in range(graph_size.size):
        adj.append(A[start_idx:start_idx + graph_size[i], start_idx:start_idx + graph_size[i]])
        features.append(X[start_idx:start_idx + graph_size[i], :])
        start_idx += graph_size[i]

    class_labels = np.loadtxt("datasets/%s/%s_graph_labels.txt" % (dataset, dataset), dtype=np.int64)
    #print(class_labels.shape) (N,)

    class_labels = np.where(class_labels==-1, 0, class_labels)
    return adj, features, class_labels


def generate_batches_(adj, features, label, batch_size, graph_pooling_type, device, shuffle):
    N = len (label) # the number of graphs 
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32) # shape (84, )

    n_batches = ceil(N/batch_size)

    adj_lst = list()
    features_lst = list()
    graph_indicator_lst = list()
    label_lst = list()
    graph_pool_lst = list()

    for i in range(0, N, batch_size):
        n_graphs = min(i+batch_size, N)-i # the number of graphs in this batch (especially the last batch)
        n_nodes = sum([adj[index[j]].shape[0] for j in range(i, min(i+batch_size, N))]) # the number of nodes in this batch

        adj_batch = lil_matrix((n_nodes, n_nodes))
        features_batch = np.zeros((n_nodes, features[0].shape[1]))
        graph_indicator_batch = np.zeros(n_nodes)
        label_batch = np.zeros(n_graphs)
        graph_pool_batch =lil_matrix((n_graphs, n_nodes))

        idx = 0

        for j in range(i, min(i+batch_size, N)):
            n = adj[index[j]].shape[0] # the number of nodes in this graph
            adj_batch[idx:idx+n, idx:idx+n] = adj[index[j]]
            features_batch[idx:idx+n, :] = features[index[j]]
            graph_indicator_batch[idx:idx+n] = j-i
            label_batch[j-i] = label[index[j]]
            
            if graph_pooling_type == "average":
                graph_pool_batch[j-i, idx:idx+n] = 1./n
            else:
                graph_pool_batch[j-i, idx:idx+n] = 1

            idx += n

        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
        features_lst.append(torch.FloatTensor(features_batch).to(device))
        graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
        label_lst.append(torch.LongTensor(label_batch).to(device))
        graph_pool_lst.append(sparse_mx_to_torch_sparse_tensor(graph_pool_batch).to(device))

    return adj_lst, features_lst, graph_pool_lst, graph_indicator_lst, label_lst, n_batches

def compute_metrics(logits, labels):



    auc_ = roc_auc_score(labels.detach().cpu().numpy(), logits.detach().cpu().numpy()[:, 1])
    accuracy_ = accuracy_score(labels.detach().cpu().numpy(), logits.detach().cpu().numpy().argmax(axis=1))

    target_names = ['C0', 'C1']
    DICT = classification_report(labels.detach().cpu().numpy(), logits.detach().cpu().numpy().argmax(axis=1), target_names=target_names, output_dict=True)
    C0_preicison = DICT['C0']['precision']
    C0_recall = DICT['C0']['recall']
    C0_f1 = DICT['C0']['f1-score']

    C1_preicison = DICT['C1']['precision']
    C1_recall = DICT['C1']['recall']
    C1_f1 = DICT['C1']['f1-score']

    macro_precision = DICT['macro avg']['precision']
    macro_recall = DICT['macro avg']['recall']
    macro_f1 = DICT['macro avg']['f1-score']

    return auc_, accuracy_, macro_precision, macro_recall, macro_f1, C0_preicison, C0_recall, C0_f1, C1_preicison, C1_recall, C1_f1

def compute_priors(num1, num2, device):
    y_prior = torch.log(torch.tensor([num1+1e-8, num2+1e-8], requires_grad = False)).to(device)
    return y_prior