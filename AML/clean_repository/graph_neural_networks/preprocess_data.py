import torch
from torch_geometric.data import Data

import pandas as pd
import numpy as np


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


def read_files(config):
    '''
    Read raw data files and create Pytorch dataset
    '''
    data_local_path = config["data_local_path"]
    n_edges = config["n_edges"]
    print("Probe genes relations")
    gene_relation_path = config["out_links"] #"out_links_10000_20_03"
    relations_probe_ids = pd.read_csv(data_local_path + gene_relation_path, sep=" ", header=None)
    print(relations_probe_ids)
    #return relations_probe_ids
    print("Edges created")
    #relations_probe_ids = ppi
    print("NAIPU and DNAM features and labels")
    features_data_path = config["nedbit_dnam_features"]
    naipu_dnam_features = pd.read_csv(data_local_path + features_data_path, sep="\t", header=None)
    print(naipu_dnam_features)
    print()
    print("Labels")
    labels = naipu_dnam_features.iloc[:, -1:]
    print(labels)
    print()
    out_genes_path =  config["out_genes"]
    out_genes = pd.read_csv(data_local_path + out_genes_path, sep=" ", header=None)
    print("Out genes NIAPU")
    print(out_genes)
    print("Feature names")
    feature_names = out_genes.iloc[:, 1]
    print(feature_names)
    print()
    print("Features without labels")
    feature_no_labels = naipu_dnam_features.iloc[:, :-1]
    print(feature_no_labels)
    print()
    print("Mapped feature names to ids")
    mapped_feature_names = out_genes.loc[:, 0]
    print(mapped_feature_names)
    print()
    print("Mapped links before sampling")
    #links_relation_probes = relations_probe_ids[:n_edges]
    print(relations_probe_ids[:n_edges])
    print("Mapped links after sampling")
    links_relation_probes = relations_probe_ids.sample(n_edges)
    print(links_relation_probes)
    print("Add cg01550473_HSPA6 to links")
    cg01550473_HSPA6 = relations_probe_ids[(relations_probe_ids.loc[:, 0] == 10841) | (relations_probe_ids.loc[:, 1] == 10841)]
    cg01550473_HSPA6.reset_index(drop=True, inplace=True)
    links_relation_probes.reset_index(drop=True, inplace=True)
    print(cg01550473_HSPA6)
    links_relation_probes = pd.concat([links_relation_probes, cg01550473_HSPA6], axis=0, ignore_index=True)
    links_relation_probes = links_relation_probes.drop_duplicates()
    print(links_relation_probes)
    print()
    print("Creating X and Y")
    x = feature_no_labels.iloc[:, 0:]
    y = labels.iloc[:, 0]
    print(y)
    # shift labels from 1...5 to 0..4
    y = y - 1
    y = torch.tensor(y.to_numpy(), dtype=torch.long)
    # create data object
    x = torch.tensor(x.to_numpy(), dtype=torch.float)
    edge_index = torch.tensor(links_relation_probes.to_numpy(), dtype=torch.long)
    # set up Pytorch geometric dataset
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())
    # set up true labels
    compact_data.y = y
    return compact_data, feature_names, mapped_feature_names, out_genes