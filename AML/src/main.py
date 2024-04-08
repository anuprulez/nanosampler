import preprocess_data
import train_model
import xai_explainer


config = {
    "SEED": 32,
    "n_edges": 100000,
    "n_epo": 1,
    "k_folds": 5,
    "batch_size": 32,
    "num_classes": 5,
    "gene_dim": 39,
    "hidden_dim": 32,
    "learning_rate": 0.001,
    "plot_local_path": "../plots/",
    "data_local_path": "../naipu_processed_data/",
    "model_local_path": "../models"
}


def run_training():
    compact_data, feature_n, mapped_f_name, out_genes = preprocess_data.read_files(config)
    trained_model, data = train_model.create_training_proc(compact_data, feature_n, mapped_f_name, out_genes, config)
    xai_explainer.gnn_explainer(trained_model, data, config)


if __name__ == "__main__":
    run_training()