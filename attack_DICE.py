import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCNJaccard, GCNSVD, RGCN
from scipy.sparse import csr_matrix

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', "uai", "acm"], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--defensemodel', type=str, default='GCN',  choices=['GCN', 'GCNJaccard', 'RGCN'])
parser.add_argument('--num_trials', type=int, default=10,  help='Number of trials')
parser.add_argument('--init', type=str, default='orthogonal',
                        choices=['orthogonal', 'uniform', 'xavier_uniform', 'kaiming_uniform', 'gaussian'])


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_trials = args.num_trials

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)


def test_gcn(adj,
            type_init ="uniform", mean_val=0.0, std_val=1.0, gain=1.0):

    if not adj.is_sparse:
        new_adj = csr_matrix(adj)

    else:
        new_adj = adj

    if not features.is_sparse:
        features_local = csr_matrix(features)

    else:
        features_local = features


    if args.defensemodel == "GCN":
        classifier = GCN(nfeat=features_local.shape[1],
                    with_bias=False, nhid=16, nclass=labels.max().item() + 1,
                     dropout=0.5, type_init=type_init, mean_val=mean_val,
                      std_val=std_val, gain=gain, device=device)
    elif args.defensemodel == "GCNJaccard":
        classifier = GCNJaccard(nfeat=features_local.shape[1],
                        nhid=16, nclass=labels.max().item() + 1,
                         dropout=0.5, type_init=type_init)

    elif args.defensemodel == "RGCN":
        classifier = RGCN(nnodes=adj.shape[0], nhid=16,
                    nfeat=features.shape[1], nclass=labels.max().item() + 1,
                            dropout=0.5, device=device, type_init=type_init)

    classifier = classifier.to(device)

    classifier.fit(features_local, new_adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val, idx_test=idx_test, verbose=False)

    classifier.eval()
    acc_val = classifier.test(idx_test)
    return acc_val


if __name__ == '__main__':


    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # Setup Attack Model
    model = DICE()

    n_perturbations = int(args.ptb_rate * (adj.sum()//2))

    model.attack(adj, labels, n_perturbations)
    modified_adj = model.modified_adj
    modified_adj = torch.FloatTensor(modified_adj.todense())

    #adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=False) #, device=device
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    l_acc_clean = []
    l_acc_attacked = []
    for exp in range(num_trials):
        clean_acc = test_gcn(adj,
                type_init=args.init)

        attacked_acc = test_gcn(modified_adj.cpu(),
                type_init=args.init)

        l_acc_clean.append(clean_acc)
        l_acc_attacked.append(attacked_acc)

    print('Using init - {} - Clean acc: {}' .format(args.init, np.mean(l_acc_clean)))
    print('Using init - {} - Attacked acc: {}' .format(args.init, np.mean(l_acc_attacked)))
