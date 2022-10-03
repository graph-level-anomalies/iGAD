import argparse
import time
import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
from utlis import load_data, generate_batches_, compute_metrics, compute_priors
from sklearn.model_selection import StratifiedShuffleSplit
from model import IGAD

parser = argparse.ArgumentParser(description='iGAD')

# dataset and model dependent args
parser.add_argument('--dataset', default='MCF-7', help='Dataset name')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')

# feature_part args
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in GNN')
parser.add_argument('--f_hidden_dim', type=int, default=64, help='hidden_dim (features)')
parser.add_argument('--f_output_dim', type=int, default=32, help='output_dim (features)')
parser.add_argument('--graph_pooling_type', type=str, default='average', choices=["sum", "average"], help='the type of graph pooling (sum/average)')

# topology_part args
parser.add_argument('--n_subgraphs', type=int, default=5, help='Number of subgraphs')
parser.add_argument('--size_subgraphs', type=int, default=10, help='Number of nodes of each subgraph')
parser.add_argument('--max_step', type=int, default=6, help='Max length of walks')
parser.add_argument('--hard', type=bool, default=False, help='whether to use ST Gumbel softmax')
parser.add_argument('--normalize', type=bool, default=True, help='whether to use ST Gumbel softmax')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--t_hidden_dim', type=int, default=32, help='hidden_dim (topology)')
parser.add_argument('--t_output_dim', type=int, default=16, help='hidden_dim (topology)')

# mlp args
parser.add_argument('--dim_1', type=int, default=32, help='hidden_dim (all)')

# other args
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--device', type=int, default=0, help='which gpu to use')
parser.add_argument('--n_split', type=int, default=5, help='cross validation')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
#os.environ['PYTHONHASHSEED'] = str(args.seed)
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True

print(device)
print(args.dataset)
performance_auc = []
performance_precision = []
performance_recall = []
performance_f1_score = []
performance_accuracy = []

performance_precision_C0 = []
performance_recall_C0 = []
performance_f1_score_C0 = []


performance_precision_C1 = []
performance_recall_C1 = []
performance_f1_score_C1 = []


adj_lst, feats_lst, graph_labels = load_data(args.dataset)
features_dim = feats_lst[0].shape[1]
skf = StratifiedShuffleSplit(args.n_split, test_size=0.2, train_size=0.8, random_state=args.seed)

Fold_idx = 1
for train_index, test_index in skf.split(np.zeros(len(adj_lst)), graph_labels):

	adj_train = [adj_lst[i] for i in train_index]
	feats_train = [feats_lst[i] for i in train_index]
	label_train = [graph_labels[i] for i in train_index]

	adj_test = [adj_lst[i] for i in test_index]
	feats_test = [feats_lst[i] for i in test_index]
	label_test = [graph_labels[i] for i in test_index]
	num_y_0 = label_train.count(0)
	num_y_1 = label_train.count(1)
	label_priors = compute_priors(num_y_0, num_y_1, device)

	model = IGAD(features_dim,
				 args.dim_1,
				 args.f_hidden_dim,
				 args.f_output_dim,
				 args.t_hidden_dim,
				 args.t_output_dim,
				 args.graph_pooling_type,
				 args.n_subgraphs,
				 args.size_subgraphs,
				 args.max_step,
				 args.normalize,
				 args.dropout,
				 device).to(device)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()


	def train(epoch, adj, feats, graph_pool, graph_indicator, labels):

		optimizer.zero_grad()
		outputs = model(adj, feats, graph_pool, graph_indicator)
		loss = criterion(outputs + label_priors, labels)
		#loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		return outputs, loss

	def test(epoch, adj, feats, graph_pool, graph_indicator, labels):

		outputs = model(adj, feats, graph_pool, graph_indicator)
		loss = criterion(outputs, labels)
		return outputs, loss

	print("----------------------------------------------starting training------------------------------------------------")

	performance_auc_epoch = []
	performance_precision_epoch = []
	performance_recall_epoch = []
	performance_f1_score_epoch = []
	performance_accuracy_epoch = []

	performance_precision_epoch_C0 = []
	performance_recall_epoch_C0 = []
	performance_f1_score_epoch_C0 = []

	performance_precision_epoch_C1 = []
	performance_recall_epoch_C1 = []
	performance_f1_score_epoch_C1 = []

	model.train()

	for epoch in range(1, args.epochs+1):

		adj_lst_train, feats_lst_train, graph_pool_lst_train, graph_indicator_lst_train, label_lst_train, n_train_batches = generate_batches_(
			adj_train, feats_train, label_train, args.batch_size, args.graph_pooling_type, device, shuffle=True)

		model.train()
		epoch_loss = 0
		epoch_time = 0
		for i in range(0, n_train_batches):
			start_time = time.time()
			outputs, loss = train(epoch,
								  adj_lst_train[i],
								  feats_lst_train[i],
								  graph_pool_lst_train[i],
								  graph_indicator_lst_train[i],
								  label_lst_train[i])
			end_time = time.time()
			epoch_time += end_time - start_time
			epoch_loss += loss.item()
			#print(f'outputs_of_a_training_batch:{outputs}') 

		print(f'Fold_idx:{Fold_idx}, Epoch: {epoch}, loss: {epoch_loss/ n_train_batches}, time: {epoch_time}s')


		logits_ = torch.Tensor().to(device)
		adj_lst_test, feats_lst_test, graph_pool_lst_test, graph_indicator_lst_test, label_lst_test, n_test_batches = generate_batches_(
			adj_test, feats_test, label_test, args.batch_size, args.graph_pooling_type, device, shuffle=False)

		model.eval()
		for j in range(0, n_test_batches):
			outputs, loss = test(epoch,
							 adj_lst_test[j],
							 feats_lst_test[j],
							 graph_pool_lst_test[j],
							 graph_indicator_lst_test[j],
							 label_lst_test[j])
			outputs = nn.functional.softmax(outputs, dim=1)
			if j ==0:
				logits_ = outputs
			else:
				logits_ = torch.cat((logits_, outputs), dim=0)

		labels_ = torch.cat(label_lst_test, dim=0)
		auc_test, accuracy_test, precision_test, recall_test, f1_score_test, \
		C0_precision_test, C0_recall_test, C0_f1_test, \
		C1_precision_test, C1_recall_test, C1_f1_test = compute_metrics(logits_, labels_)

		performance_auc_epoch.append(auc_test)
		performance_recall_epoch.append(recall_test)
		performance_f1_score_epoch.append(f1_score_test)
		performance_recall_epoch_C0.append(C0_recall_test)
		performance_f1_score_epoch_C0.append(C0_f1_test)
		performance_recall_epoch_C1.append(C1_recall_test)
		performance_f1_score_epoch_C1.append(C1_f1_test)



	Fold_idx += 1


	performance_auc.append(performance_auc_epoch)
	performance_recall.append(performance_recall_epoch)
	performance_f1_score.append(performance_f1_score_epoch)
	performance_recall_C0.append(performance_recall_epoch_C0)
	performance_f1_score_C0.append(performance_f1_score_epoch_C0)
	performance_recall_C1.append(performance_recall_epoch_C1)
	performance_f1_score_C1.append(performance_f1_score_epoch_C1)




auc_ = np.array(performance_auc)
recall_ = np.array(performance_recall)
f1_ = np.array(performance_f1_score)
accuracy_ = np.array(performance_accuracy)
recall_C0 = np.array(performance_recall_C0)
f1_C0 = np.array(performance_f1_score_C0)
recall_C1 = np.array(performance_recall_C1)
f1_C1 = np.array(performance_f1_score_C1)



auc_mean = np.mean(auc_, axis=0)
recall_mean = np.mean(recall_, axis=0)
f1_mean = np.mean(f1_, axis=0)
recall_C0_mean = np.mean(recall_C0, axis=0)
f1_C0_mean = np.mean(f1_C0, axis=0)
recall_C1_mean = np.mean(recall_C1, axis=0)
f1_C1_mean = np.mean(f1_C1, axis=0)



#__________________________________________________
idx = np.argmax(recall_mean)
best_auc_mean = auc_mean[idx]
best_recall_mean = recall_mean[idx]
best_f1_mean = f1_mean[idx]

auc_std = np.std(auc_[:, idx])
recall_std = np.std(recall_[:, idx])
f1_std = np.std(f1_[:, idx])

print(f'**under the situation of best recall, the best_idx:{idx}')
print('auc:%.4f +- %.4f' %(best_auc_mean, auc_std))
print('recall:%.4f +- %.4f' %(best_recall_mean, recall_std))
print('f1_score:%.4f +- %.4f' %(best_f1_mean, f1_std))
print('recall_C0:%.4f +- %.4f' %(recall_C0_mean[idx], np.std(recall_C0[:, idx])))
print('f1_score_C0:%.4f +- %.4f' %(f1_C0_mean[idx], np.std(f1_C0[:, idx])))
print('recall_C1:%.4f +- %.4f' %(recall_C1_mean[idx], np.std(recall_C1[:, idx])))
print('f1_score_C1:%.4f +- %.4f' %(f1_C1_mean[idx], np.std(f1_C1[:, idx])))
