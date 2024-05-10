import preprocess_data
import train_model
import xai_explainer


config = {
    "SEED": 32,
    "n_edges": 1500000,
    "n_epo": 10,
    "k_folds": 5,
    "batch_size": 128,
    "num_classes": 5,
    "gene_dim": 39,
    "hidden_dim": 128,
    "learning_rate": 0.0001,
    "out_links": "out_links_only_positive_corr", # "out_links_10000_20_03"
    "out_genes": "out_genes_only_positive_corr", # "out_genes_10000_20_03"
    "nedbit_dnam_features": "df_nebit_dnam_features.csv", # "df_nebit_dnam_features.csv"
    "plot_local_path": "../plots/only_positive_corr_data/",
    "data_local_path": "../naipu_processed_data/only_positive_corr_data/",
    "model_local_path": "../models/only_positive_corr_data/"
}


def run_training():
    compact_data, feature_n, mapped_f_name, out_genes = preprocess_data.read_files(config)
    trained_model, data = train_model.create_training_proc(compact_data, feature_n, mapped_f_name, out_genes, config)
    #xai_explainer.gnn_explainer(trained_model, data, config)


if __name__ == "__main__":
    run_training()
