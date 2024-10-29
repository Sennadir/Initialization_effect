"""
Contains the complete implementation to reproduce the results of the "PGD"
Attack.


---
Note that I changed the code so as to use training example rather than in the test
case.
For the old case, refer to "attack_PGD_old.py"
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack.topology_attack import PGDAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from deeprobust.graph.defense import GCNJaccard, GCNSVD, RGCN
import scipy

from scipy.sparse import csr_matrix
import pickle


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora',
                'cora_ml', 'citeseer', 'polblogs', 'pubmed', "uai", "acm"], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,
                                                        help='pertubation rate')
parser.add_argument('--defensemodel', type=str, default='GCN',  choices=['GCN', 'GCNJaccard', 'RGCN'])
parser.add_argument('--num_trials', type=int, default=10,  help='Number of trials')
parser.add_argument('--init', type=str, default='orthogonal',
                        choices=['orthogonal', 'uniform', 'xavier_uniform', 'kaiming_uniform', 'gaussian'])


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_trials = args.num_trials

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)


def test_gcn(adj,
            type_init ="uniform", mean_val=0.0, std_val=1.0, gain=1.0):
    ''' test on GCN '''

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
    """
    Main function containing the PGD implementation.
    """

    adj, features, labels = data.adj, data.features, data.labels

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    n_perturbations = int(args.ptb_rate * (adj.sum()//2))

    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    sparse_adj, sparse_features = csr_matrix(adj), csr_matrix(features)

    # Target surrogate model
    target_gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5,
              device=device)

    target_gcn = target_gcn.to(device)
    target_gcn.fit(features, adj, labels, idx_train)

    # Setup Attack Model
    print('=== setup attack model ===')
    model = PGDAttack(model=target_gcn, nnodes=adj.shape[0], loss_type='CE',
                                                                device=device)
    model = model.to(device)

    model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbations)

    modified_adj = model.modified_adj

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
