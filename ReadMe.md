# If You Want to Be Robust, Be Wary of Initialization

## Overview

This repository contains python codes and datasets necessary to reproduce the results of our paper: "If You Want to Be Robust, Be Wary of Initialization", accepted at Neurips 2024. The goal of our work is to show the existence of a connection between the initialization, number of epochs and the final model's underlying robustness.

## Requirements

Code is written in Python 3.6 and requires:

- PyTorch
- Torch Geometric
- NetworkX

## Datasets
For node classification, the used datasets are as follows:
- Cora
- CiteSeer
- ACM

All these datasets are part of the torch_geometric datasets and are directly downloaded when running the code.


## Training and Evaluation
To use our code, the user should first download the DeepRobust package (https://github.com/DSE-MSU/DeepRobust).

Since our aim was to investigate different initialization strategies, we adapted the code of the different models to make that possible. In this perspective, you need first to substitute the files in the folder "deeprobust/graph/defense" by the one provided in our implementation. Specifically, the file "gcn.py" (contains the GCN), "r_gcn.py" (contains the RGCN) and "gcn_preprocess.py" (contains the GCNJaccard)


To train and evaluate the model in the paper, the user should specify the following :

- Dataset : The dataset to be used
- hidden_dimension: The hidden dimension used in the model (if desired, otherwise default will be used)
- learning rate and epochs
- ptb_rate: The budget of the attack
- init: The initialization distribution ("orthogonal", "uniform" ...)
- defensemodel: The model to be used (GCN, RGCN, GCNJaccard)
- num_trials: The number of experiments (note that we use 10 for our results).


To run the code for the GCN subject to the Mettack approach using the Cora dataset and using the default parameters for a 10% budget with an orthogonal initialization:

```bash
python attack_mettack.py --dataset cora --ptb_rate 0.1 --init orthogonal -- defensemodel GCN
```

To run the code for the RGCN subject to the PGD approach using the CiteSeer dataset and using the default parameters for a 10% budget with an uniform initialization:

```bash
python attack_mettack.py --dataset citeseer --ptb_rate 0.1 --init orthogonal -- defensemodel RGCN
```

To run the code for the GCNJaccard subject to the DICE approach using the ACM dataset and using the default parameters for a 10% budget with the Kaiming initialization:

```bash
python attack_mettack.py --dataset acm --ptb_rate 0.1 --init kaiming_uniform -- defensemodel GCNJaccard
```


## Details

Note that for all our experimentations, we have used a 2-layers model. Please refer to the paper for additional experimental details.

## Citing
If you find our proposed analysis useful for your research, please consider citing our paper.

For any additional questions/suggestions you might have about the code and/or the proposed analysis, please contact: ennadir@kth.se.

## License
Our work is licensed under the MIT License.
