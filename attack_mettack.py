import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN, GCNJaccard, GCNSVD, RGCN

from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.defense import *
from deeprobust.graph.data import Dataset
import argparse
from scipy.sparse import csr_matrix
import pickle
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
import scipy
import numpy as np
import os

import pickle
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float,     default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')

parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')

parser.add_argument('--defensemodel', type=str, default='GCN',  choices=['GCN', 'GCNJaccard', 'RGCN'])
parser.add_argument('--num_trials', type=int, default=10,  help='Number of trials')
parser.add_argument('--init', type=str, default='orthogonal',
                        choices=['orthogonal', 'uniform', 'xavier_uniform', 'kaiming_normal', 'gaussian'])

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_trials = args.num_trials

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)

adj, features, labels = data.adj, data.features, data.labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)


perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)


#1. to CSR sparse
adj, features = csr_matrix(adj), csr_matrix(features)


"""add undirected edges, orgn-arxiv is directed graph, we transfer it to undirected closely following
https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
"""
adj = adj + adj.T
adj[adj>1] = 1


# Setup GCN as the Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=False, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, train_iters=201)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)



def test_gcn(adj, type_init ="uniform", mean_val=0.0, std_val=1.0, gain=1.0):

    if args.defensemodel == "GCNJaccard":
        classifier = GCNJaccard(nfeat=features.shape[1],
                    nhid=16, nclass=labels.max().item() + 1,
                     dropout=0.5, type_init=type_init)

    elif args.defensemodel == "GCN":
        classifier = GCN(nfeat=features.shape[1],
                    with_bias=False, nhid=16, nclass=labels.max().item() + 1,
                     dropout=0.5, type_init=type_init, mean_val=mean_val,
                      std_val=std_val, gain=gain, device=device)

    elif args.defensemodel == "RGCN":
        classifier = RGCN(nnodes=adj.shape[0], nhid=16,
                nfeat=features.shape[1], nclass=labels.max().item() + 1,
                        dropout=0.5, device=device, type_init=type_init)


    classifier = classifier.to(device)

    classifier.fit(features, adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val, idx_test=idx_test, verbose=False)

    classifier.eval()
    val_acc = classifier.test(idx_test)

    return val_acc



if __name__ == '__main__':

    """Apply the attack"""
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations,
                                                            ll_constraint=False)
    modified_adj = model.modified_adj
    modified_adj_sparse = csr_matrix(modified_adj.cpu().numpy())

    # --- Train and test the model ---
    l_acc_clean = []
    l_acc_attacked = []
    for exp in range(num_trials):
        clean_acc = test_gcn(adj,
                type_init=args.init)

        attacked_acc = test_gcn(modified_adj_sparse,
                type_init=args.init)

        l_acc_clean.append(clean_acc)
        l_acc_attacked.append(attacked_acc)

    print('Using init - {} - Clean acc: {}' .format(args.init, np.mean(l_acc_clean)))
    print('Using init - {} - Attacked acc: {}' .format(args.init, np.mean(l_acc_attacked)))
