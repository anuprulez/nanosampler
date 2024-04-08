#import os
#import sys
#import time
#import pandas as pd
#import numpy as np
#import json
#import random
#import matplotlib.pyplot as plt
#import seaborn as sns

#import torch
#from torch_geometric.data import Data
#from torch.nn import Linear, BatchNorm1d, ReLU
#from torch_geometric.nn import GCNConv
#import torch.nn.functional as F

#import sklearn
#from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
#from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, roc_auc_score, RocCurveDisplay, average_precision_score, confusion_matrix

import preprocess_data
import train_model
import xai_explainer

'''n_epo = 1
k_folds = 5
batch_size = 32
num_classes = 5
gene_dim = 39
learning_rate = 0.001'''

config = {
    "n_edges": 1000,
    "n_epo": 1,
    "k_folds": 5,
    "batch_size": 32,
    "num_classes": 5,
    "gene_dim": 39,
    "learning_rate": 0.001,
    "plot_local_path": "../plots/",
    "data_local_path": "../naipu_processed_data/"
}


def run_training():
    compact_data, feature_n, mapped_f_name, out_genes = preprocess_data.read_files(config)
    trained_model, data = train_model.create_training_proc(compact_data, feature_n, mapped_f_name, out_genes, config)
    #xai_explainer.gnn_explainer(trained_model, data)


if __name__ == "__main__":
    run_training()