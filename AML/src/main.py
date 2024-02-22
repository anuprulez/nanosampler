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
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, roc_auc_score, RocCurveDisplay, average_precision_score


# matplotlib settings
#font = {'family': 'serif', 'size': 20}
#plt.rc('font', **font)

# data path
local_path = "nanodiag_datasets/GSE175758/"

# neural network parameters
SEED = 32
n_epo = 50
k_folds = 5
batch_size = 32
num_classes = 2
gene_dim = 12
learning_rate = 0.001
n_edges = 500 #267186


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


def read_files():
    '''
    Read raw data files and create Pytorch dataset
    '''
    final_path = local_path
    aml_probes_genes = pd.read_csv(local_path + "positive_probes_genes.tsv", sep="\t", header=None)
    print("Positive AML probes")
    aml_probes_genes = aml_probes_genes[0].tolist()[1:]
    print(len(aml_probes_genes))
    print()
    non_aml_probes_genes = pd.read_csv(local_path + "negative_probes_genes.tsv", sep="\t", header=None)
    print("Negative AML probes")
    non_aml_probes_genes = non_aml_probes_genes[0].tolist()[1:]
    print(len(non_aml_probes_genes))
    print()
    print("AML probe features")
    aml_probes_features = pd.read_csv(local_path + "positive_signals.tsv", sep="\t")
    print(aml_probes_features)
    print()
    print("Non-AML probe features")
    non_aml_probe_features = pd.read_csv(local_path + "balanced_negative_signals.tsv", sep="\t")
    print(non_aml_probe_features)
    print()
    probes_correlation_matrix = pd.read_csv(local_path + "df_pos_neg_probes_correlation_matrix.tsv", sep="\t")
    print(probes_correlation_matrix)
    print()
    probe_ids_mapping = pd.read_csv(local_path + "probe_genes_mapping_id.tsv", sep="\t")
    print(probe_ids_mapping)
    print()
    
    
    '''gene_features = pd.read_csv(final_path + "gene_features", header=None)
    links = pd.read_csv(final_path + "links", header=None)
    passengers = pd.read_csv(final_path + "passengers", header=None)
    print("AML genes")
    print(driver)
    print()
    print("----")
    print("Non-AML genes")
    print(passengers)
    print("----")
    print()
    print("Gene embeddings")
    print(gene_features)
    print("----")
    print()
    print("Gene links")
    print(links)
    print("----")
    

    driver_gene_list = driver[0].tolist()
    passenger_gene_list = passengers[0].tolist()

    x, mapping = load_node_csv(final_path + "gene_features", 0)
    y = torch.zeros(x.shape[0], dtype=torch.long)
    # assign all labels to -1
    y[:] = -1
    driver_ids = driver.replace({0: mapping})
    passenger_ids = passengers.replace({0: mapping})

    # driver = 1, passenger = 0
    y[driver_ids[0].tolist()] = 1
    y[passenger_ids[0].tolist()] = 0

    print("Saving mapping...")
    save_mapping_json(mapping)

    print("replacing gene ids")
    # set number of edges
    links = links[:n_edges]
    # replace gene names with ids, only the first two columns
    re_links = links.replace({0: mapping})
    re_links = re_links.replace({1: mapping})
    # create data object
    x = torch.tensor(x.loc[:, 1:].to_numpy(), dtype=torch.float)
    edge_index = torch.tensor(re_links.to_numpy(), dtype=torch.long)
    # set up Pytorch geometric dataset
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())
    # set up true labels
    compact_data.y = y
    return compact_data, driver_ids, passenger_ids, gene_features, mapping'''


def create_training_proc(compact_data, driver_ids, passenger_ids, gene_features, mapping):
    '''
    Create network architecture and assign loss, optimizers ...
    '''
    print(compact_data)
    # initialize model
    model = GCN()
    print(model)
    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # set integer ids to genes
    print("Replacing gene ids...")
    all_gene_ids = gene_features.replace({0: mapping})
    # balanced data size for each class
    equal_size = int(batch_size / float(num_classes))
    
    driver_ids_list = driver_ids[0].tolist()
    passenger_ids_list = passenger_ids[0].tolist()

    random.shuffle(driver_ids_list)
    random.shuffle(passenger_ids_list)

    driver_ids_list = np.reshape(driver_ids_list, (len(driver_ids_list), 1))
    passenger_ids_list = np.reshape(passenger_ids_list, (len(passenger_ids_list), 1))

    # create cross validation (CV) fold object
    kfold = KFold(n_splits=k_folds, shuffle=True)

    tr_loss_epo = list()
    te_acc_epo = list()
    dr_cls_acc_epo = list()
    dr_prec_epo = list()
    dr_recall_epo = list()
    dr_fpr_epo = list()
    dr_tpr_epo = list()
    dr_auprc_epo = list()
    # loop over epochs
    for epoch in range(n_epo):
        tr_loss_fold = list()
        te_acc_fold = list()
        dr_cls_acc_fold = list()
        dr_fpr = list()
        dr_tpr = list()
        dr_prec = list()
        dr_recall = list()
        dr_auprc_fold = list()
        # loop over folds
        # extract training and test for driver and passenger genes separately and then combine
        for fold, (dr_tr, pass_tr) in enumerate(zip(kfold.split(driver_ids_list), kfold.split(passenger_ids_list))):
            dr_tr_ids, dr_te_ids = dr_tr
            pass_tr_ids, pass_te_ids = pass_tr
            n_batches = int((len(dr_tr_ids) + len(pass_tr_ids) + 1) / float(batch_size))
            # combine te genes
            dr_te_ids = np.reshape(dr_te_ids, (dr_te_ids.shape[0]))
            pass_te_ids = np.reshape(pass_te_ids, (pass_te_ids.shape[0]))

            # extract gene ids
            dr_te_genes = driver_ids_list[dr_te_ids]
            pass_te_genes = passenger_ids_list[pass_te_ids]

            # reshape
            dr_te_genes = list(np.reshape(dr_te_genes, (dr_te_genes.shape[0])))
            pass_te_genes = list(np.reshape(pass_te_genes, (pass_te_genes.shape[0])))

            # create test masks
            # set test mask using test genes for drivers and passengers
            compact_data.test_mask = create_masks(all_gene_ids, dr_te_genes, pass_te_genes)

            batch_tr_loss = list()
            # loop over batches
            for bat in range(n_batches):
                random.shuffle(dr_tr_ids)
                random.shuffle(pass_tr_ids)
                batch_dr_tr_genes = driver_ids_list[dr_tr_ids]
                batch_dr_tr_genes = list(batch_dr_tr_genes.reshape((batch_dr_tr_genes.shape[0])))
                # balance train batches for both classes
                if len(batch_dr_tr_genes) < equal_size:
                    # oversample by choosing with replacement
                    batch_dr_tr_genes = list(np.random.choice(batch_dr_tr_genes, size=equal_size))
                else:
                    batch_dr_tr_genes = batch_dr_tr_genes[:int(batch_size / float(2))]
                batch_pass_tr_genes = passenger_ids_list[pass_tr_ids]
                batch_pass_tr_genes = batch_pass_tr_genes.reshape((batch_pass_tr_genes.shape[0]))
                if len(batch_pass_tr_genes) < equal_size:
                    # oversample by choosing with replacement
                    batch_pass_tr_genes = list(np.random.choice(batch_pass_tr_genes, size=equal_size))
                else:
                    batch_pass_tr_genes = batch_pass_tr_genes[:int(batch_size / float(2))]
                # set training mask using drivers and passengers genes with data balancing for each batch
                compact_data.train_mask = create_masks(all_gene_ids, batch_dr_tr_genes, batch_pass_tr_genes)
                # training for each batch
                tr_loss, h = train(compact_data, optimizer, model, criterion)
                batch_tr_loss.append(tr_loss.detach().numpy())
            tr_loss_fold.append(np.mean(batch_tr_loss))

            # create list of driver and passenger genes
            test_driver_genes = np.reshape(driver_ids_list[dr_te_ids], (len(driver_ids_list[dr_te_ids]))).tolist()
            test_passenger_genes = np.reshape(passenger_ids_list[pass_te_ids], (len(passenger_ids_list[pass_te_ids]))).tolist()
            # predict using trained model
            dr_cls_acc, dr_auprc, test_acc, _, _, _, _, _ = predict_data(model, compact_data, test_driver_genes, test_passenger_genes)
            dr_auprc_fold.append(dr_auprc)
            dr_cls_acc_fold.append(dr_cls_acc)
            print("Epoch {}/{}, fold {}/{} average training loss: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(np.mean(batch_tr_loss))))
            print("Epoch: {}/{}, Fold: {}/{}, test accuracy: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(test_acc)))
            print("Epoch: {}/{}, Fold: {}/{}, test per class accuracy, Driver: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(dr_cls_acc)))
            print("Epoch: {}/{}, Fold: {}/{}, test AUPRC, Driver: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(dr_auprc)))
            te_acc_fold.append(test_acc)

        print("-------------------")
        tr_loss_epo.append(np.mean(tr_loss_fold))
        te_acc_epo.append(np.mean(te_acc_fold))
        dr_cls_acc_epo.append(np.mean(dr_cls_acc_fold))
        dr_auprc_epo.append(np.mean(dr_auprc_fold))
        print()
        print("Epoch {}: Training Loss: {}".format(str(epoch+1), str(np.mean(tr_loss_fold))))
        print("Epoch {}: Test accuracy: {}".format(str(epoch+1), str(np.mean(te_acc_fold))))
        print("Epoch {}: Test per class accuracy, Driver: {}".format(str(epoch+1), str(np.mean(dr_cls_acc_fold))))
        print("Epoch {}: Test AUPRC, Driver: {}".format(str(epoch+1), str(np.mean(dr_auprc_fold))))
        print()

    # prepare test mask using all driver genes (# of driver genes are less)
    compact_data.test_mask = create_masks(all_gene_ids, driver_ids[0].tolist(), passenger_ids[0].tolist())

    # compute accuracy on all driver genes using trained model
    dr_com_acc, auprc_wt, te_acc, dr_com_fpr, dr_com_tpr, dr_com_prec, dr_com_rec, dr_roc_auc_score = predict_data(model, compact_data, driver_ids[0].tolist(), passenger_ids[0].tolist())

    print("CV Training Loss after {} epochs: {}".format(str(n_epo), str(np.mean(tr_loss_epo))))
    print("CV Test driver accuracy after {} epochs: {}".format(str(n_epo), str(np.mean(dr_cls_acc_epo))))
    print("Driver AUPRC after {} epochs: {}".format(str(n_epo), str(np.mean(dr_auprc_epo))))

    # create plots
    plot_dr_prec_recall(dr_com_fpr, dr_com_tpr, dr_com_prec, dr_com_rec, dr_roc_auc_score, auprc_wt)
    plot_loss_acc(n_epo, tr_loss_epo, dr_cls_acc_epo)

    # select top genes
    top_dr_gene_ids = get_top_genes(model, compact_data, driver_ids[0].tolist())

    # predict labels of unlabeled nodes
    # TODO:


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, header=None)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = df.iloc[:, 0:]
    return x, mapping


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
    
    print("Driver prediction: Accuracy {}, # correctly predicted/total samples {}/{}".format(dr_pred_acc, dr_corr, dr_tot))
    print()
    return dr_pred_acc, auprc_wt, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score


def create_masks(all_gene_ids, dr_te_genes, pass_te_genes):
    dr_te_genes.extend(pass_te_genes)
    te_mask = all_gene_ids[0].isin(dr_te_genes)
    te_mask = torch.tensor(te_mask, dtype=torch.bool)
    return te_mask


def predict_data(model, compact_data, test_driver_genes, test_passenger_genes):
    '''
    Predict using trained model and test data
    '''
    # predict on test fold
    model.eval()
    out = model(compact_data.x, compact_data.edge_index)
    pred = out[0].argmax(dim=1)
    test_correct = pred[compact_data.test_mask] == compact_data.y[compact_data.test_mask]
    test_acc = int(test_correct.sum()) / int(compact_data.test_mask.sum())
    dr_cls_acc, auprc_wt, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score = agg_per_class_acc(out[0], pred, compact_data, test_driver_genes, test_passenger_genes)
    return dr_cls_acc, auprc_wt, test_acc, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score


def save_mapping_json(mapping_file):
    with open('gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping_file))


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
    # [2609, 3501, 11659, 2641, 8759, 2502, 9911, 10645, 9067, 7315]
    # 2609: "CTNNB1". More details: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6354055/#:~:text=Mutations%20in%20the%20%CE%B2%2Dcatenin,status%20of%20cancer%2Drelated%20genes.
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
def plot_dr_prec_recall(fpr_epo, tpr_epo, dr_prec_epo, dr_recall_epo, dr_roc_auc_score, auprc_wt):

    # plot precision, recall curve
    disp = PrecisionRecallDisplay(precision=dr_prec_epo, recall=dr_recall_epo, pos_label="Driver").plot()
    plt.title("Cancer {}, Precision-Recall curve (all drivers), AUPRC: {}".format(cancer_type, str(np.round(auprc_wt, 2))))
    plt.grid(True)
    plt.savefig("plots/cancer_{}_prc_all_drivers_links_{}_epo_{}.pdf".format(cancer_type, n_edges, n_epo), dpi=200)
    plt.show()

    # plot ROC
    roc_auc = sklearn.metrics.auc(fpr_epo, tpr_epo)
    roc_display = RocCurveDisplay(fpr=fpr_epo, tpr=tpr_epo, roc_auc=roc_auc).plot()
    plt.title("Cancer {}, ROC curve (all drivers)".format(cancer_type))
    plt.grid(True)
    plt.savefig("plots/cancer_{}_roc_all_drivers_links_{}_epo_{}.pdf".format(cancer_type, n_edges, n_epo), dpi=200)
    plt.show()


def plot_loss_acc(n_epo, tr_loss, te_acc):
    # plot training loss
    x_val = np.arange(n_epo)
    plt.plot(x_val, tr_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("Cancer {}, {} fold CV training loss vs epochs".format(cancer_type, str(k_folds)))
    plt.savefig("plots/cancer_{}_{}_fold_CV_driver_training_loss_links_{}_epo_{}.pdf".format(cancer_type, k_folds, n_edges, n_epo), dpi=200)
    plt.show()


    # plot driver gene precision vs epochs
    x_val = np.arange(n_epo)
    plt.plot(x_val, te_acc)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("Cancer {}, {} fold CV Driver pred accuracy vs epochs".format(cancer_type, str(k_folds)))
    plt.savefig("plots/cancer_{}_{}_fold_CV_driver_test_accuracy_links_{}_epo_{}.pdf".format(cancer_type, k_folds, n_edges, n_epo), dpi=200)
    plt.show()


if __name__ == "__main__":
    compact_data, aml_genes_ids, non_aml_gene_ids, probe_gene_features, mapping = read_files()
    #create_training_proc(compact_data, driver_ids, passenger_ids, gene_features, mapping)