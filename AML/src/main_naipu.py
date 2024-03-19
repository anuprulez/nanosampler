import os
import sys
import time
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv

import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, roc_auc_score, RocCurveDisplay, average_precision_score, confusion_matrix


# matplotlib settings
#font = {'family': 'serif', 'size': 20}
#plt.rc('font', **font)

# data path
local_path = "nanodiag_datasets/GSE175758/naipu_processed_files/"

# neural network parameters
SEED = 32
n_epo = 100
k_folds = 5
batch_size = 32
num_classes = 5
gene_dim = 39
learning_rate = 0.001
n_edges = 100000

def create_edges():
    print("Probe genes relations")
    gene_relation_path = "significant_gene_relation_large.tsv"
    relations_probe_ids = pd.read_csv(local_path + gene_relation_path, sep="\t", header=None)
    relations_probe_ids = relations_probe_ids.sample(n=n_edges, random_state=1)
    print(relations_probe_ids)
    relations_probe_ids = relations_probe_ids[:n_edges]
    print(relations_probe_ids)
    feature_names = pd.read_csv(local_path + "df_feature_names.csv", sep="\t", header=None)
    feature_names.loc[:, 1] = feature_names.index
    print(feature_names)
    probe_gene_id_mapping = {index: i for i, index in enumerate(feature_names.loc[:, 0].unique())}
    print("Mapping relations to IDs...")
    s_time = time.time()
    relations_probe_ids = relations_probe_ids.replace({0: probe_gene_id_mapping})
    mapped_relations = relations_probe_ids.replace({1: probe_gene_id_mapping})
    e_time = time.time()
    print("Time elapsed for Id mapping for in-out links: {} seconds".format(e_time - s_time))
    print()
    print(mapped_relations)
    mapped_relations.to_csv(local_path + "mapped_significant_gene_relations.tsv", sep="\t", index=None)
    print("Edges created")


class GCN(torch.nn.Module):
    '''
    Neural network with graph convolution network (GCN)
    '''
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.conv1 = GCNConv(gene_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16) 
        self.conv4 = GCNConv(16, 8)
        self.classifier = Linear(8, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h

def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, header=None)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = df.iloc[:, 0:]
    return x, mapping

def save_mapping_json(lp, mapping_file):
    with open(lp, 'gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping_file))

def replace_name_by_ids(dataframe, col_index, mapper):
    names = dataframe.iloc[:, col_index]
    lst_names = names.tolist()
    ids = [mapper[mapper["name"] == name]["id"].values[0] for name in lst_names]
    dataframe.iloc[:, col_index] = ids
    return dataframe

def read_files():
    '''
    Read raw data files and create Pytorch dataset
    '''
    final_path = local_path
    print("Probe genes relations")
    gene_relation_path = "mapped_significant_gene_relations.tsv"
    relations_probe_ids = pd.read_csv(local_path + gene_relation_path, sep="\t", header=None)
    print(relations_probe_ids)
    print()
    print("NAIPU and DNAM features and labels")
    features_data_path = "df_nebit_dnam_features.csv"
    naipu_dnam_features = pd.read_csv(local_path + features_data_path, sep="\t", header=None)
    print(naipu_dnam_features)
    print()
    print("Feature names")
    feature_names = pd.read_csv(local_path + "df_feature_names.csv", sep="\t", header=None)
    print(feature_names)
    feature_names.loc[:, 1] = feature_names.index
    print(feature_names)
    #probe_gene_id_mapping = {index: i for i, index in enumerate(feature_names.loc[:, 0].unique())}
    print()
    print("Labels")
    labels = naipu_dnam_features.iloc[:, -1:]
    print(labels)
    print()
    print("Features without labels")
    feature_no_labels = naipu_dnam_features.iloc[:, :-1]
    print(feature_no_labels)
    print()
    print("Mapped feature names to ids")
    mapped_feature_names = feature_names.loc[:, 1]
    print(mapped_feature_names)    
    print("Mapped links")
    links_relation_probes = relations_probe_ids[:n_edges]
    print(links_relation_probes)
    print()
    print("Creating X and Y")
    x = feature_no_labels.iloc[:, 0:]
    y = torch.zeros(x.shape[0], dtype=torch.long)
    y = labels.iloc[:, 0]
    #shift labels from 1...5 to 0..4
    y = y - 1
    y = torch.tensor(y.to_numpy(), dtype=torch.long)
    # create data object
    x = torch.tensor(x.to_numpy(), dtype=torch.float)
    edge_index = torch.tensor(links_relation_probes.to_numpy(), dtype=torch.long)
    # set up Pytorch geometric dataset
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())
    # set up true labels
    compact_data.y = y

    return compact_data, feature_names, mapped_feature_names


def create_training_proc(compact_data, feature_n, mapped_f_name):
    '''
    Create network architecture and assign loss, optimizers ...
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Compact data")
    print(compact_data)
    print("Initialize model")
    model = GCN()
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    st_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
    tr_loss_epo = list()
    te_acc_epo = list()
    val_acc_epo = list()
    #mapped_f_name = mapped_f_name[:30]
    lst_mapped_f_name = np.array(mapped_f_name.index)
    complete_rand_index = [item for item in range(len(mapped_f_name.index))]
    tr_index, te_index = train_test_split(complete_rand_index, shuffle=True, test_size=0.33, random_state=42)
    tr_nodes = lst_mapped_f_name[tr_index]
    te_nodes = lst_mapped_f_name[te_index]
    print("tr_nodes: ", tr_nodes)
    print("te_nodes: ", te_nodes)
    print("intersection: ", list(set(tr_nodes).intersection(set(te_nodes))))
    compact_data.test_mask = create_masks(mapped_f_name, te_nodes)
    # loop over epochs
    print("Start epoch training...")
    for epoch in range(n_epo):
        tr_loss_fold = list()
        val_acc_fold = list()
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_index, val_index) in enumerate(kfold.split(tr_nodes)):
            val_node_ids = tr_nodes[val_index]
            train_nodes_ids = tr_nodes[train_index]
            compact_data.val_mask = create_masks(mapped_f_name, val_node_ids)
            n_batches = int((len(train_index) + 1) / float(batch_size))
            batch_tr_loss = list()
            # loop over batches
            print("Start fold training for epoch: {}, fold: {}...".format(epoch+1, fold+1))
            for bat in range(n_batches):
                batch_tr_node_ids = train_nodes_ids[bat * batch_size: (bat+1) * batch_size]
                compact_data.batch_train_mask = create_masks(mapped_f_name, batch_tr_node_ids)
                tr_loss, h = train(compact_data, optimizer, model, criterion)
                batch_tr_loss.append(tr_loss.detach().numpy())
            tr_loss_fold.append(np.mean(batch_tr_loss))
            # predict using trained model
            val_acc = predict_data_val(model, compact_data)
            print("Epoch {}/{}, fold {}/{} average training loss: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(np.mean(batch_tr_loss))))
            print("Epoch: {}/{}, Fold: {}/{}, val accuracy: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(val_acc)))
            val_acc_fold.append(val_acc)

        print("-------------------")
        te_acc, _, _ = predict_data_test(model, compact_data)
        te_acc_epo.append(te_acc)
        tr_loss_epo.append(np.mean(tr_loss_fold))
        val_acc_epo.append(np.mean(val_acc_fold))
        print()
        print("Epoch {}: Training Loss: {}".format(str(epoch+1), str(np.mean(tr_loss_fold))))
        print("Epoch {}: Val accuracy: {}".format(str(epoch+1), str(np.mean(val_acc_epo))))
        print("Epoch {}: Test accuracy: {}".format(str(epoch+1), str(np.mean(te_acc))))
        print()
    print("==============")
    plot_loss_acc(n_epo, tr_loss_epo, val_acc_epo, te_acc_epo)
    print("CV Training Loss after {} epochs: {}".format(str(n_epo), str(np.mean(tr_loss_epo))))
    print("CV Val acc after {} epochs: {}".format(str(n_epo), str(np.mean(val_acc_epo))))
    final_test_acc, pred_labels, true_labels = predict_data_test(model, compact_data)
    print("CV Test acc after {} epochs: {}".format(n_epo, final_test_acc))
    plot_confusion_matrix(true_labels, pred_labels, n_edges, n_epo)
    print("==============")


def create_masks(mapped_node_ids, mask_list):
    mask = mapped_node_ids.index.isin(mask_list)
    return torch.tensor(mask, dtype=torch.bool)


def train(data, optimizer, model, criterion):
    '''
    Training step
    '''
    # Clear gradients
    optimizer.zero_grad()
    # forward pass
    out, h = model(data.x, data.edge_index)
    # compute error using training mask
    loss = criterion(out[data.batch_train_mask], data.y[data.batch_train_mask])
    # compute gradients
    loss.backward()
    # optimize weights
    optimizer.step()
    return loss, h


def plot_confusion_matrix(true_labels, predicted_labels, edges, epo, classes=[0, 1, 2, 3, 4]):
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font scale for better readability
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    # Add labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Show plot
    plt.tight_layout()
    #plt.show()
    plt.grid(True)
    plt.savefig(local_path + "Confusion_matrix_NPPI_{}_NEpochs_{}.pdf".format(edges, epo), dpi=200)


def predict_data_val(model, compact_data):
    '''
    Predict using trained model and test data
    '''
    # predict on test fold
    model.eval()
    out = model(compact_data.x, compact_data.edge_index)
    pred = out[0].argmax(dim=1)
    val_correct = pred[compact_data.val_mask] == compact_data.y[compact_data.val_mask]
    val_acc = int(val_correct.sum()) / float(int(compact_data.val_mask.sum()))
    return val_acc


def predict_data_test(model, compact_data):
    '''
    Predict using trained model and test data
    '''
    # predict on test fold
    model.eval()
    out = model(compact_data.x, compact_data.edge_index)
    pred = out[0].argmax(dim=1)
    pred_labels = pred[compact_data.test_mask]
    true_labels = compact_data.y[compact_data.test_mask]
    test_correct = pred_labels == true_labels
    test_acc = int(test_correct.sum()) / float(int(compact_data.test_mask.sum()))
    return test_acc, pred_labels.numpy(), true_labels.numpy()
    

########################
# Plotting methods
########################
def plot_loss_acc(n_epo, tr_loss, val_acc, te_acc):
    # plot training loss
    plt.figure()
    x_val = np.arange(n_epo)
    plt.plot(x_val, tr_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("Training loss")
    plt.savefig(local_path + "{}_folds_CV_{}_links_{}_epochs_training_loss.pdf".format(k_folds, n_edges, n_epo), dpi=200)

    plt.figure()
    # plot accuracy on validation data
    x_val = np.arange(n_epo)
    plt.plot(x_val, val_acc)
    plt.plot(x_val, te_acc)
    plt.ylabel("Validation and Test accuracy")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.legend(["Validation", "Test"])
    plt.title("Validation and Test accuracy")
    plt.savefig(local_path + "{}_folds_CV_{}_links_{}_epochs_validation_test_accuracy.pdf".format(k_folds, n_edges, n_epo), dpi=200)
    plt.show()
    

if __name__ == "__main__":
    #create_edges()
    compact_data, feature_n, mapped_f_name = read_files()
    create_training_proc(compact_data, feature_n, mapped_f_name)
