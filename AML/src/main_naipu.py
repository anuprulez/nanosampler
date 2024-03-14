import os
import sys
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv

import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, roc_auc_score, RocCurveDisplay, average_precision_score


# matplotlib settings
#font = {'family': 'serif', 'size': 20}
#plt.rc('font', **font)

# data path
local_path = "nanodiag_datasets/GSE175758/naipu_processed_files/"

# neural network parameters
SEED = 32
n_epo = 2
k_folds = 5
batch_size = 2
num_classes = 5
gene_dim = 39
learning_rate = 0.001
n_edges = 10000

def create_edges():
    print("Probe genes relations")
    gene_relation_path = "significant_gene_relation_large.tsv"
    relations_probe_ids = pd.read_csv(local_path + gene_relation_path, sep="\t", header=None)
    print(relations_probe_ids)
    feature_names = pd.read_csv(local_path + "df_feature_names.csv", sep="\t", header=None)
    feature_names.loc[:, 1] = feature_names.index
    print(feature_names)
    probe_gene_id_mapping = {index: i for i, index in enumerate(feature_names.loc[:, 0].unique())}
    print("Mapping relations to IDs...")
    relations_probe_ids = relations_probe_ids.replace({0: probe_gene_id_mapping})
    mapped_relations = relations_probe_ids.replace({1: probe_gene_id_mapping})
    print()
    print(mapped_relations)
    mapped_relations.to_csv(local_path + "mapped_significant_gene_relations.csv", sep="\t", index=None)
    return mapped_relations


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
    #final_path = local_path
    #print("Probe genes relations")
    #gene_relation_path = "significant_gene_relation_large.tsv"
    #relations_probe_ids = pd.read_csv(local_path + gene_relation_path, sep="\t", header=None)
    #print(relations_probe_ids)
    print()
    print("NAIPU and DNAM features and labels")
    features_data_path = "df_nebit_dnam_features.csv"
    naipu_dnam_features = pd.read_csv(local_path + features_data_path, sep="\t", header=None)
    print(naipu_dnam_features)
    print()
    print("Feature names")
    #feature_names = pd.read_csv(local_path + "df_feature_names.csv", sep="\t", header=None)
    feature_names = pd.read_csv(local_path + "df_feature_names.csv", sep="\t", header=None)
    print(feature_names)
    #print(len(feature_names[0].tolist()))
    print()
    print("Labels")
    labels = naipu_dnam_features.iloc[:, -1:]
    print(labels)
    print()
    print("Features without labels")
    feature_no_labels = naipu_dnam_features.iloc[:, :-1]
    print(feature_no_labels)
    print()
    print("Probe name and ids mapping")
    probe_ids_mapping = pd.read_csv(local_path + "probe_genes_mapping_id.tsv", sep="\t")
    print(probe_ids_mapping)
    print()
    
    #print("Overall gene id mapping")
    #feature_names = feature_names[:10]
    #print("Mapping feature names to ids")
    #mapped_feature_names = replace_name_by_ids(feature_names, 0, probe_ids_mapping)
    #print(mapped_feature_names)
    #mapped_feature_names.to_csv(local_path + "mapped_feature_names.tsv", sep="\t", index=None)
    
    print("Mapped feature names to ids")
    mapped_feature_names = pd.read_csv(local_path + "mapped_feature_names.tsv", sep="\t")
    print(mapped_feature_names)
    
    # replace gene names with ids, only the first two columns
    #relations_probe_ids = relations_probe_ids[:n_edges]
    #print("Mapping links to ids")
    #relations_probe_ids = replace_name_by_ids(relations_probe_ids, 0, probe_ids_mapping)
    #links_relation_probes = replace_name_by_ids(relations_probe_ids, 1, probe_ids_mapping)
    #print("Mapped links")
    #print(links_relation_probes)
    #links_relation_probes = links_relation_probes.astype('int32')
    #links_relation_probes.to_csv(local_path + "links_relation_probes.tsv", sep="\t", index=None)
    
    print("Mapped links")
    links_relation_probes = pd.read_csv(local_path + "links_relation_probes.tsv", sep="\t")
    print(links_relation_probes)
    print()
    print("Creating X and Y")
    x = feature_no_labels.iloc[:, 0:]
    y = torch.zeros(x.shape[0], dtype=torch.long)
    y = labels.iloc[:, 0]
    y = y - 1 #shift labels from 1...5 to 0..4
    print(y)

    # create data object
    x = torch.tensor(x.to_numpy(), dtype=torch.float)
    edge_index = torch.tensor(links_relation_probes.to_numpy(), dtype=torch.long)
    # set up Pytorch geometric dataset
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())
    # set up true labels
    compact_data.y = y
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())
    # set up true labels
    compact_data.y = y

    return compact_data, probe_ids_mapping, feature_names, mapped_feature_names


def create_training_proc(compact_data, probe_gene_id_mapping, feature_n, mapped_f_name):
    '''
    Create network architecture and assign loss, optimizers ...
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Compact data")
    print(compact_data)
    #compact_data = compact_data.to(device)
    print("Initialize model")
    # initialize model
    model = GCN()
    print(model)
    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # create cross-validation (CV) fold object
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    st_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    tr_loss_epo = list()
    te_acc_epo = list()
    aml_cls_acc_epo = list()
    aml_prec_epo = list()
    aml_recall_epo = list()
    aml_fpr_epo = list()
    aml_tpr_epo = list()
    aml_auprc_epo = list()
    #mapped_f_name = mapped_f_name[:20]
    rand_index = [item for item in range(len(mapped_f_name.index))]
    print(mapped_f_name)
    lst_mapped_f_name = np.array(mapped_f_name.iloc[:, 0].tolist())
    print(lst_mapped_f_name)
    print("Random index")
    print(rand_index)
    # loop over epochs
    for epoch in range(n_epo):
        tr_loss_fold = list()
        te_acc_fold = list()
        aml_cls_acc_fold = list()
        aml_fpr = list()
        aml_tpr = list()
        aml_prec = list()
        aml_recall = list()
        aml_auprc_fold = list()
        # loop over folds
        # extract training and test for driver and passenger genes separately and then combine
        #for fold, (aml_tr, non_aml_tr) in enumerate(zip(kfold.split(aml_ids_list), kfold.split(non_aml_ids_list))):
        for fold, (train_test_index, val_index) in enumerate(kfold.split(rand_index)):
            
            print(train_test_index, len(train_test_index))
            print("Val index", val_index, len(val_index))
            print("Validation nodes")
            val_node_ids = lst_mapped_f_name[val_index]
            train_nodes_ids = lst_mapped_f_name[train_test_index]
            print("Train nodes: ", len(train_nodes_ids), "Val nodes:", len(val_node_ids))
            compact_data.test_mask = create_masks(mapped_f_name, val_node_ids)
            n_batches = int((len(train_test_index) + 1) / float(batch_size))
            batch_tr_loss = list()
            # loop over batches
            print("Start training...")
            for bat in range(n_batches):
                print(bat * batch_size, (bat+1) * batch_size)
                batch_tr_index = train_test_index[bat * batch_size: (bat+1) * batch_size]
                print(bat, batch_tr_index)
                tr_node_ids = lst_mapped_f_name[batch_tr_index]
                compact_data.train_mask = create_masks(mapped_f_name, tr_node_ids)
                tr_loss, h = train(compact_data, optimizer, model, criterion)
                batch_tr_loss.append(tr_loss.detach().numpy())
            tr_loss_fold.append(np.mean(batch_tr_loss))
            # predict using trained model
            test_acc, _, _, _, _, _ = predict_data(model, compact_data)
            #aml_auprc_fold.append(aml_auprc)
            #aml_cls_acc_fold.append(aml_cls_acc)
            print("Epoch {}/{}, fold {}/{} average training loss: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(np.mean(batch_tr_loss))))
            print("Epoch: {}/{}, Fold: {}/{}, test accuracy: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(test_acc)))
            #print("Epoch: {}/{}, Fold: {}/{}, test per class accuracy, AML: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(aml_cls_acc)))
            #print("Epoch: {}/{}, Fold: {}/{}, test AUPRC, AML: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(aml_auprc)))
            te_acc_fold.append(test_acc)

        print("-------------------")
        tr_loss_epo.append(np.mean(tr_loss_fold))
        te_acc_epo.append(np.mean(te_acc_fold))
        #aml_cls_acc_epo.append(np.mean(aml_cls_acc_fold))
        #aml_auprc_epo.append(np.mean(aml_auprc_fold))
        print()
        print("Epoch {}: Training Loss: {}".format(str(epoch+1), str(np.mean(tr_loss_fold))))
        print("Epoch {}: Test accuracy: {}".format(str(epoch+1), str(np.mean(te_acc_fold))))
        #print("Epoch {}: Test per class accuracy, AML: {}".format(str(epoch+1), str(np.mean(aml_cls_acc_fold))))
        #print("Epoch {}: Test AUPRC, AML: {}".format(str(epoch+1), str(np.mean(aml_auprc_fold))))
        print()
    print("==============")
    print("CV Training Loss after {} epochs: {}".format(str(n_epo), str(np.mean(tr_loss_epo))))
    print("CV Test acc after {} epochs: {}".format(str(n_epo), str(np.mean(te_acc_epo))))
    print("==============")
    #print("CV Test driver accuracy after {} epochs: {}".format(str(n_epo), str(np.mean(aml_cls_acc_epo))))
    #print("AML AUPRC after {} epochs: {}".format(str(n_epo), str(np.mean(aml_auprc_epo))))
    #plot_loss_acc(n_epo, tr_loss_epo, aml_cls_acc_epo)


def create_masks(mapped_node_ids, mask_list):
    print("In mask")
    print(mapped_node_ids)
    print(mask_list)
    mask = mapped_node_ids.iloc[:, 0].isin(mask_list)
    print(mask)
    return torch.tensor(mask, dtype=torch.bool)


def predict_data(model, compact_data):
    '''
    Predict using trained model and test data
    '''
    # predict on test fold
    model.eval()
    out = model(compact_data.x, compact_data.edge_index)
    pred = out[0].argmax(dim=1)
    test_correct = pred[compact_data.test_mask] == compact_data.y[compact_data.test_mask]
    test_acc = int(test_correct.sum()) / float(int(compact_data.test_mask.sum()))
    #dr_cls_acc, auprc_wt, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score = agg_per_class_acc(out[0], pred, compact_data, test_driver_genes, test_passenger_genes)
    #return dr_cls_acc, auprc_wt, test_acc, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score
    return test_acc

def agg_per_class_acc(prob_scores, pred, data, driver_ids, passenger_ids):
    '''
    Compute precision, recall, auc scores
    '''
    dr_tot = 0
    dr_corr = 0
    pass_tot = 0
    pass_corr = 0
    prob_scores = prob_scores.detach().numpy()
    # extract probability scores for driver (positive class, repr as 1) and passenger (negative class, repr as 0) genes 
    dr_prob_scores = prob_scores[driver_ids]
    pass_prob_scores = prob_scores[passenger_ids]
    #print("Prob scores: ", len(dr_prob_scores), len(pass_prob_scores))
    # compute precision for driver genes
    for driver_id in driver_ids:
        if data.test_mask[driver_id] == torch.tensor(True):
             dr_tot += 1
             if pred[driver_id] == torch.tensor(1):
                dr_corr += 1
    dr_pred_acc = dr_corr / float(dr_tot)
    # 1 as driver genes labels
    dr_label = torch.ones(dr_prob_scores.shape[0], dtype=torch.long)
    # 0 as passenger genes labels
    pass_label = torch.zeros(pass_prob_scores.shape[0], dtype=torch.long)
    # concatenate labels and their associated prob scores for driver and passenger genes
    dr_pass_label = torch.cat((dr_label, pass_label), 0)
    dr_pass_prob_scores = torch.cat((torch.tensor(dr_prob_scores), torch.tensor(pass_prob_scores)), 0)
    # compute precision, recall, tpr, fpr, auc scores
    dr_precision, dr_recall, _ = precision_recall_curve(dr_pass_label, dr_pass_prob_scores[:, 1], pos_label=1)
    dr_fpr, dr_tpr, _ = roc_curve(dr_pass_label, dr_pass_prob_scores[:, 1], pos_label=1)
    dr_roc_auc_score = roc_auc_score(dr_pass_label, dr_pass_prob_scores[:, 1])

    auprc_wt = average_precision_score(dr_pass_label, dr_pass_prob_scores[:, 1], average='weighted')
    print('AP', auprc_wt)
    
    print("AML prediction: Accuracy {}, # correctly predicted/total samples {}/{}".format(dr_pred_acc, dr_corr, dr_tot))
    print()
    return dr_pred_acc, auprc_wt, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score


def get_top_genes(model, compact_data, test_driver_genes, top_genes=10):
    '''
    Find top predicted driver genes
    '''
    corr_driver_genes = dict()
    model.eval()
    out = model(compact_data.x, compact_data.edge_index)
    #print(out, out[0].shape, out[1].shape)
    pred = out[0].argmax(dim=1)
    for driver_id in test_driver_genes:
        if compact_data.test_mask[driver_id] == torch.tensor(True):
             if pred[driver_id] == torch.tensor(1):
                 corr_driver_genes[driver_id] = out[0][:, 1][driver_id].detach().numpy()
    s_top_genes = {k: v for k, v in sorted(corr_driver_genes.items(), key=lambda item: item[1], reverse=True)}
    print("Top 10 Driver genes: ", list(s_top_genes.keys())[:top_genes])
    print("Find these ids in the gene_mapping.json file")
    

def train(data, optimizer, model, criterion):
    '''
    Training step
    '''
    # Clear gradients
    optimizer.zero_grad()
    # forward pass
    out, h = model(data.x, data.edge_index)
    # compute error using training mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    # compute gradients
    loss.backward()
    # optimize weights
    optimizer.step()
    return loss, h


########################
# Plotting methods
########################
def plot_aml_prec_recall(fpr_epo, tpr_epo, dr_prec_epo, dr_recall_epo, dr_roc_auc_score, auprc_wt):

    # plot precision, recall curve
    disp = PrecisionRecallDisplay(precision=dr_prec_epo, recall=dr_recall_epo, pos_label="AML").plot()
    plt.title("AML, Precision-Recall curve (all AML), AUPRC: {}".format(str(np.round(auprc_wt, 2))))
    plt.grid(True)
    plt.savefig(local_path + "AML_{}_prc_all_AML_links_{}_epo_{}.pdf".format(n_edges, n_epo), dpi=200)
    #plt.show()

    # plot ROC
    roc_auc = sklearn.metrics.auc(fpr_epo, tpr_epo)
    roc_display = RocCurveDisplay(fpr=fpr_epo, tpr=tpr_epo, roc_auc=roc_auc).plot()
    plt.title("ROC curve (all AML)")
    plt.grid(True)
    plt.savefig(local_path + "AML_roc_all_aml_links_{}_epo_{}.pdf".format("AML", n_edges, n_epo), dpi=200)
    #plt.show()


def plot_loss_acc(n_epo, tr_loss, te_acc):
    # plot training loss
    x_val = np.arange(n_epo)
    plt.plot(x_val, tr_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("{}, {} fold CV training loss vs epochs".format("AML", str(k_folds)))
    plt.savefig(local_path + "{}_{}_fold_CV_AML_training_loss_links_{}_epo_{}.pdf".format("AML", k_folds, n_edges, n_epo), dpi=200)
    #plt.show()

    # plot driver gene precision vs epochs
    x_val = np.arange(n_epo)
    plt.plot(x_val, te_acc)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("{}, {} fold CV AML pred accuracy vs epochs".format("AML", str(k_folds)))
    plt.savefig(local_path + "AML_{}_fold_CV_test_accuracy_links_{}_epo_{}.pdf".format(k_folds, n_edges, n_epo), dpi=200)
    #plt.show()


if __name__ == "__main__":
    create_edges()
    #compact_data, mapping, feature_n, mapped_f_name = read_files()
    #create_training_proc(compact_data, mapping, feature_n, mapped_f_name)