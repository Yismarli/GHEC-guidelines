# main.py
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from ensemblebaselayer import train_base_ensemble
from decision_network_representation import build_decision_network, construct_node_features, load_and_preprocess_graph_data
from decision_work_pruning import clf_graph
from second_stage_strategy import comparative_trial_classification  # your existing comparison

if __name__ == '__main__':
    # Load data (replace with your actual path)
    data = pd.read_csv('your_data.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Base layer
    node_preds, node_test_preds, perf_metrics, node_count, val_y, val_X, classifiers = train_base_ensemble(
        X_train, X_test, y_train, y_test
    )

    y_val_tensor = torch.tensor(val_y.values)
    y_test_tensor = torch.tensor(y_test.values)

    # Decision network
    adj, edge_index = build_decision_network(perf_metrics, node_count)

    # Node features (no original features)
    construct_node_features(node_preds, node_test_preds, perf_metrics, adj, node_count)

    # Graph data
    train_graphs, test_graphs = load_and_preprocess_graph_data(node_count)

    # GNN pruning
    model, removed_nodes = clf_graph(train_graphs, test_graphs, y_val_tensor, y_test_tensor,
                                     edge_index, node_count, classifiers)

    print("Removed nodes:", removed_nodes)

    # Optional: compare with traditional ensembles
    comparative_trial_classification(X_train, X_test, y_train, y_test)