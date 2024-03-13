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
gene_dim = 34
learning_rate = 0.001
n_edges = 738000 #267186


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


def create_edges():
    relation_threshold = 0.8
    in_probe_relation = list()
    out_probe_relation = list()
    print("Loading Pearson correlation matrix...")
    corr_mat = pd.read_csv(local_path + "df_pos_neg_probes_correlation_matrix.tsv", sep="\t", index_col=None)
    print(corr_mat)
    print()
    del corr_mat[corr_mat.columns[0]]
    print(corr_mat)
    print()
    probe_gene_names = corr_mat.columns
    corr_mat_filtered = corr_mat > relation_threshold
    print(corr_mat_filtered)
    print()
    print("Creating probe-gene relations...")
    for index, col in enumerate(corr_mat_filtered.columns):
        for item_idx, item in enumerate(corr_mat_filtered[col]):
            if item == True and col != probe_gene_names[item_idx]:
                in_probe_relation.append(col)
                out_probe_relation.append(probe_gene_names[item_idx])
    df_significant_gene_relation = pd.DataFrame(zip(in_probe_relation, out_probe_relation), columns=["In", "Out"])
    df_significant_gene_relation.to_csv(local_path + "significant_gene_relation.tsv", sep="\t", index=False)
    print(df_significant_gene_relation)

def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, header=None)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = df.iloc[:, 0:]
    return x, mapping

def save_mapping_json(lp, mapping_file):
    with open(lp, 'gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping_file))

def read_files():
    '''
    Read raw data files and create Pytorch dataset
    '''
    
    final_path = local_path
    df_aml_probes_genes = pd.read_csv(local_path + "positive_probes_genes.tsv", sep="\t", header=None)
    df_aml_probes_genes = df_aml_probes_genes[1:]
    print("Positive AML probes")
    aml_probes_genes = df_aml_probes_genes[0].tolist()[1:]
    print(df_aml_probes_genes)
    print()
    df_non_aml_probes_genes = pd.read_csv(local_path + "negative_probes_genes.tsv", sep="\t", header=None)
    df_non_aml_probes_genes = df_non_aml_probes_genes[1:]
    print("Negative AML probes")
    non_aml_probes_genes = df_non_aml_probes_genes[0].tolist()[1:]
    print(df_non_aml_probes_genes)
    #print()
    #print("AML probe features")
    aml_probes_features = pd.read_csv(local_path + "positive_signals.tsv", sep="\t")
    #print(aml_probes_features)
    #print()
    aml_probes_features_T = aml_probes_features.transpose()
    #print(aml_probes_features_T)
    #print()
    #print("Non-AML probe features")
    non_aml_probe_features = pd.read_csv(local_path + "balanced_negative_signals.tsv", sep="\t")
    #print(non_aml_probe_features)
    #print()
    non_aml_probe_features_T = non_aml_probe_features.transpose()
    print(non_aml_probe_features_T)
    #print()
    #print("Probes correlation matrix")
    probes_correlation_matrix = pd.read_csv(local_path + "df_pos_neg_probes_correlation_matrix.tsv", sep="\t")
    #print(probes_correlation_matrix)
    #print()
    print("Probe name and ids mapping")
    probe_ids_mapping = pd.read_csv(local_path + "probe_genes_mapping_id.tsv", sep="\t")
    print(probe_ids_mapping)
    print()
    print("Probe genes relations")
    relations_probe_ids = pd.read_csv(local_path + "significant_gene_relation.tsv", sep="\t", header=None)
    relations_probe_ids = relations_probe_ids[1:]
    print(relations_probe_ids)
    print()
    
    #aml_gene_list = df_aml_probes_genes[0].tolist()
    #non_aml_gene_list = df_non_aml_probes_genes[0].tolist()

    #x, mapping = load_node_csv(local + "gene_features", 0)
    combine_aml_non_aml_probe_features = pd.concat([aml_probes_features_T, non_aml_probe_features_T], axis=0)
    print(combine_aml_non_aml_probe_features)
    
    #probe_gene_id_mapping = {probe_gene_name: probe_ids_mapping[probe_ids_mapping["name"] == probe_gene_name]["id"] for i, probe_gene_name in enumerate(combine_aml_non_aml_probe_features.index.unique())}

    probe_gene_id_mapping = {index: i for i, index in enumerate(combine_aml_non_aml_probe_features.index.unique())}
    x = combine_aml_non_aml_probe_features #.iloc[:, 0:]
    print(x.shape, x)
    
    y = torch.zeros(x.shape[0], dtype=torch.long)
    # assign all labels to -1
    y[:] = -1
    aml_probes_genes_ids = df_aml_probes_genes.replace({0: probe_gene_id_mapping})
    non_aml_probes_genes_ids = df_non_aml_probes_genes.replace({0: probe_gene_id_mapping})
    print("AML/Non AML probe gene ids")
    print()
    print(aml_probes_genes_ids)
    print()
    print(non_aml_probes_genes_ids)
    print()

    # aml = 1, non-aml = 0
    y[aml_probes_genes_ids[0].tolist()] = 1
    y[non_aml_probes_genes_ids[0].tolist()] = 0

    #print("Saving mapping...")
    #save_mapping_json(mapping)

    print("replacing gene ids")
    # set number of edges
    links = relations_probe_ids[:n_edges]
    print("Filtered relations:")
    print(links)
    print()
    
    # replace gene names with ids, only the first two columns
    links = links.replace({0: probe_gene_id_mapping})
    links_relation_probes = links.replace({1: probe_gene_id_mapping})

    print(links_relation_probes)
    
    # create data object
    x = torch.tensor(x.loc[:, 0:].to_numpy(), dtype=torch.float)
    edge_index = torch.tensor(links_relation_probes.to_numpy(), dtype=torch.long)
    # set up Pytorch geometric dataset
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())
    # set up true labels
    compact_data.y = y

    return compact_data, aml_probes_genes_ids, non_aml_probes_genes_ids, combine_aml_non_aml_probe_features, probe_gene_id_mapping


def create_training_proc(compact_data, aml_ids, non_aml_ids, combine_aml_non_aml_probe_features, probe_gene_id_mapping):
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
    # set integer ids to genes
    print(probe_gene_id_mapping)
    print("Replacing gene ids...")
    probe_ids = [probe_gene_id_mapping[i] for i in combine_aml_non_aml_probe_features.index.tolist()]
    combine_aml_non_aml_probe_features.insert(0, "probes", probe_ids)
    all_gene_ids = combine_aml_non_aml_probe_features
    print("All probes: ", all_gene_ids)
    print()
    # balanced data size for each class
    equal_size = int(batch_size / float(num_classes))
    
    aml_ids_list = aml_ids[0].tolist()
    non_aml_ids_list = non_aml_ids[0].tolist()

    #print("AML Ids", aml_ids_list)
    #print("Non-AML Ids", non_aml_ids_list)
    #print()

    random.shuffle(aml_ids_list)
    random.shuffle(non_aml_ids_list)

    aml_ids_list = np.reshape(aml_ids_list, (len(aml_ids_list), 1))
    non_aml_ids_list = np.reshape(non_aml_ids_list, (len(non_aml_ids_list), 1))

    #print("AML Ids", aml_ids_list)
    #print("Non-AML Ids", non_aml_ids_list)
    #print()

    # create cross-validation (CV) fold object
    kfold = KFold(n_splits=k_folds, shuffle=True)

    tr_loss_epo = list()
    te_acc_epo = list()
    aml_cls_acc_epo = list()
    aml_prec_epo = list()
    aml_recall_epo = list()
    aml_fpr_epo = list()
    aml_tpr_epo = list()
    aml_auprc_epo = list()
    
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
        for fold, (aml_tr, non_aml_tr) in enumerate(zip(kfold.split(aml_ids_list), kfold.split(non_aml_ids_list))):
            aml_tr_ids, aml_te_ids = aml_tr
            non_aml_tr_ids, non_aml_te_ids = non_aml_tr
            
            #print("AML TR/TE", aml_tr_ids, aml_te_ids)
            #print("Non-AML TR/TE", non_aml_tr_ids, non_aml_te_ids)
            
            n_batches = int((len(aml_tr_ids) + len(non_aml_tr_ids) + 1) / float(batch_size))
            
            # combine te genes
            aml_te_ids = np.reshape(aml_te_ids, (aml_te_ids.shape[0]))
            non_aml_te_ids = np.reshape(non_aml_te_ids, (non_aml_te_ids.shape[0]))

            # extract gene ids
            aml_te_genes = aml_ids_list[aml_te_ids]
            non_aml_te_genes = non_aml_ids_list[non_aml_te_ids]

            # reshape
            aml_te_genes = list(np.reshape(aml_te_genes, (aml_te_genes.shape[0])))
            non_aml_te_genes = list(np.reshape(non_aml_te_genes, (non_aml_te_genes.shape[0])))
            
            #print("AML Test probes", aml_te_genes)
            #print("Non AML Test probes", non_aml_te_genes)
            
            #print("AML probes: {}, Non-AML probes: {}".format(len(aml_te_genes), len(non_aml_te_genes)))
            #sys.exit()
            # create test masks
            # set test mask using test genes for drivers and passengers
            compact_data.test_mask = create_masks(all_gene_ids, aml_te_genes, non_aml_te_genes)

            batch_tr_loss = list()
            # loop over batches
            print("Start training...")
            for bat in range(n_batches):
                random.shuffle(aml_tr_ids)
                random.shuffle(non_aml_tr_ids)
                batch_aml_tr_genes = aml_ids_list[aml_tr_ids]
                batch_aml_tr_genes = list(batch_aml_tr_genes.reshape((batch_aml_tr_genes.shape[0])))
                # balance train batches for both classes
                if len(batch_aml_tr_genes) < equal_size:
                    # oversample by choosing with replacement
                    batch_aml_tr_genes = list(np.random.choice(batch_aml_tr_genes, size=equal_size))
                else:
                    batch_aml_tr_genes = batch_aml_tr_genes[:int(batch_size / float(2))]
                    
                batch_non_aml_tr_genes = non_aml_ids_list[non_aml_tr_ids]
                batch_non_aml_tr_genes = batch_non_aml_tr_genes.reshape((batch_non_aml_tr_genes.shape[0]))
                
                if len(batch_non_aml_tr_genes) < equal_size:
                    # oversample by choosing with replacement
                    batch_non_aml_tr_genes = list(np.random.choice(batch_non_aml_tr_genes, size=equal_size))
                else:
                    batch_non_aml_tr_genes = batch_non_aml_tr_genes[:int(batch_size / float(2))]
                # set training mask using drivers and passengers genes with data balancing for each batch
                compact_data.train_mask = create_masks(all_gene_ids, batch_aml_tr_genes, batch_non_aml_tr_genes)
                # training for each batch
                
                tr_loss, h = train(compact_data, optimizer, model, criterion)
                batch_tr_loss.append(tr_loss.detach().numpy())
            tr_loss_fold.append(np.mean(batch_tr_loss))

            # create list of driver and passenger genes
            test_aml_genes = np.reshape(aml_ids_list[aml_te_ids], (len(aml_ids_list[aml_te_ids]))).tolist()
            test_non_aml_genes = np.reshape(non_aml_ids_list[non_aml_te_ids], (len(non_aml_ids_list[non_aml_te_ids]))).tolist()
            #print("In predict data...")
            #print("Test probes: ", len(test_aml_genes), len(test_non_aml_genes))
            # predict using trained model
            aml_cls_acc, aml_auprc, test_acc, _, _, _, _, _ = predict_data(model, compact_data, test_aml_genes, test_non_aml_genes)
            aml_auprc_fold.append(aml_auprc)
            aml_cls_acc_fold.append(aml_cls_acc)
            print("Epoch {}/{}, fold {}/{} average training loss: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(np.mean(batch_tr_loss))))
            print("Epoch: {}/{}, Fold: {}/{}, test accuracy: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(test_acc)))
            print("Epoch: {}/{}, Fold: {}/{}, test per class accuracy, AML: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(aml_cls_acc)))
            print("Epoch: {}/{}, Fold: {}/{}, test AUPRC, AML: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(aml_auprc)))
            te_acc_fold.append(test_acc)

        print("-------------------")
        tr_loss_epo.append(np.mean(tr_loss_fold))
        te_acc_epo.append(np.mean(te_acc_fold))
        aml_cls_acc_epo.append(np.mean(aml_cls_acc_fold))
        aml_auprc_epo.append(np.mean(aml_auprc_fold))
        print()
        print("Epoch {}: Training Loss: {}".format(str(epoch+1), str(np.mean(tr_loss_fold))))
        print("Epoch {}: Test accuracy: {}".format(str(epoch+1), str(np.mean(te_acc_fold))))
        print("Epoch {}: Test per class accuracy, AML: {}".format(str(epoch+1), str(np.mean(aml_cls_acc_fold))))
        print("Epoch {}: Test AUPRC, AML: {}".format(str(epoch+1), str(np.mean(aml_auprc_fold))))
        print()

    print("CV Training Loss after {} epochs: {}".format(str(n_epo), str(np.mean(tr_loss_epo))))
    print("CV Test driver accuracy after {} epochs: {}".format(str(n_epo), str(np.mean(aml_cls_acc_epo))))
    print("AML AUPRC after {} epochs: {}".format(str(n_epo), str(np.mean(aml_auprc_epo))))
    plot_loss_acc(n_epo, tr_loss_epo, aml_cls_acc_epo)
    
    # prepare test mask using all driver genes (# of driver genes are less)
    #compact_data.test_mask = create_masks(all_gene_ids, aml_ids[0].tolist(), non_aml_ids[0].tolist())

    # compute accuracy on all driver genes using trained model
    #aml_com_acc, auprc_wt, te_acc, aml_com_fpr, aml_com_tpr, aml_com_prec, aml_com_rec, aml_roc_auc_score = predict_data(model, compact_data, aml_ids[0].tolist(), non_aml_ids[0].tolist())

    #print("CV Training Loss after {} epochs: {}".format(str(n_epo), str(np.mean(tr_loss_epo))))
    #print("CV Test driver accuracy after {} epochs: {}".format(str(n_epo), str(np.mean(aml_cls_acc_epo))))
    #print("AML AUPRC after {} epochs: {}".format(str(n_epo), str(np.mean(aml_auprc_epo))))

    # create plots
    #plot_aml_prec_recall(aml_com_fpr, aml_com_tpr, aml_com_prec, aml_com_rec, aml_roc_auc_score, auprc_wt)
    #plot_loss_acc(n_epo, tr_loss_epo, aml_cls_acc_epo)

    # select top genes
    #top_dr_gene_ids = get_top_genes(model, compact_data, driver_ids[0].tolist())

    # predict labels of unlabeled nodes
    # TODO:


def create_masks(all_gene_ids, dr_te_genes, pass_te_genes):
    dr_te_genes.extend(pass_te_genes)
    te_mask = all_gene_ids["probes"].isin(dr_te_genes)
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
    test_acc = int(test_correct.sum()) / float(int(compact_data.test_mask.sum()))
    dr_cls_acc, auprc_wt, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score = agg_per_class_acc(out[0], pred, compact_data, test_driver_genes, test_passenger_genes)
    return dr_cls_acc, auprc_wt, test_acc, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score


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
    compact_data, aml_genes_ids, non_aml_gene_ids, probe_gene_features, mapping = read_files()
    create_training_proc(compact_data, aml_genes_ids, non_aml_gene_ids, probe_gene_features, mapping)
    #### Run reference code
    #compact_data, driver_ids, passenger_ids, gene_features, mapping = read_files()
    #create_training_proc(compact_data, driver_ids, passenger_ids, gene_features, mapping)
