from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def gnn_explainer(model, data, config):
    plot_local_path = config["plot_local_path"]
    print("Running GNN explanation...")
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    node_index = 596
    explanation = explainer(data.x, data.edge_index, index=node_index)
    plt.figure(figsize=(8, 6))
    print(f'Generated explanations in {explanation.available_explanations}')
    path = plot_local_path + 'feature_importance.png'
    explanation.visualize_feature_importance(path, top_k=10)
    print(f"Feature importance plot has been saved to '{path}'")
    plt.figure(figsize=(8, 6))
    path = plot_local_path + 'subgraph.pdf'
    explanation.visualize_graph(path)
    print(f"Subgraph visualization plot has been saved to '{path}'")
    print(explanation.edge_mask, explanation.edge_mask.shape)
    print(explanation.node_mask, explanation.node_mask.shape)