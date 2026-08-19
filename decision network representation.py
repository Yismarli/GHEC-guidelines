"""
Build decision network (adjacency matrix) and construct node feature files.
No original feature columns are used; only prediction, metrics, and degrees.
"""
import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import os


def build_decision_network(performance_metrics, node_count):
    """
    Build directed graph based on BalancedAccuracy (edge from higher to lower).
    Returns adjacency matrix and edge_index tensor.
    """
    print("================== Decision Network Construction ==================")
    adj = np.zeros((node_count, node_count), dtype=int)

    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                acc_i = performance_metrics.loc[performance_metrics['Node'] == i, 'BalancedAccuracy'].values[0]
                acc_j = performance_metrics.loc[performance_metrics['Node'] == j, 'BalancedAccuracy'].values[0]
                if acc_i > acc_j:
                    adj[i, j] = 1

    print("Adjacency matrix:\n", adj)

    coo = coo_matrix(adj)
    rows, cols = coo.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

    return adj, edge_index


def construct_node_features(node_predictions, node_test_predictions, performance_metrics,
                            adjacency_matrix, node_count):
    """
    Create CSV files for each node's training and testing features.
    Features: NodePredictions, BalancedAccuracy, TPR, TNR, OutDegree, InDegree.
    No original input features are included.
    """
    print("================== Node Feature Construction ==================")

    out_deg = np.sum(adjacency_matrix == 1, axis=1)
    in_deg = np.sum(adjacency_matrix == 1, axis=0)
    out_deg = out_deg / np.linalg.norm(out_deg) if np.linalg.norm(out_deg) > 0 else out_deg
    in_deg = in_deg / np.linalg.norm(in_deg) if np.linalg.norm(in_deg) > 0 else in_deg

    for node_idx in range(node_count):
        print(f"Processing node {node_idx}")
        metrics = performance_metrics.iloc[node_idx]

        # Training features
        train_df = pd.DataFrame({
            'NodePred': node_predictions.iloc[:, node_idx],
            'BalancedAcc': metrics['BalancedAccuracy'],
            'TPR': metrics['TPR'],
            'TNR': metrics['TNR'],
            'OutDeg': out_deg[node_idx],
            'InDeg': in_deg[node_idx]
        })
        train_df.to_csv(f'node_training_features_{node_idx}.csv', index=False)

        # Testing features
        test_df = pd.DataFrame({
            'NodePred': node_test_predictions.iloc[:, node_idx],
            'BalancedAcc': metrics['BalancedAccuracy'],
            'TPR': metrics['TPR'],
            'TNR': metrics['TNR'],
            'OutDeg': out_deg[node_idx],
            'InDeg': in_deg[node_idx]
        })
        test_df.to_csv(f'node_testing_features_{node_idx}.csv', index=False)

    print("Node feature files saved.")


def load_and_preprocess_graph_data(node_count):
    """
    Load node CSVs and combine into per-sample graph tensors.
    """
    print("================== Graph Data Loading ==================")
    train_files = [f"node_training_features_{i}.csv" for i in range(node_count)]
    test_files = [f"node_testing_features_{i}.csv" for i in range(node_count)]

    train_tensors = load_node_files(train_files, node_count)
    test_tensors = load_node_files(test_files, node_count)

    # Cleanup
    for f in train_files + test_files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    return train_tensors, test_tensors


def load_node_files(file_list, node_count):
    """
    Read CSVs, combine row-wise across nodes into sample tensors.
    """
    dataframes = [pd.read_csv(f) for f in file_list]
    num_samples = len(dataframes[0])
    tensors = {}
    for sample_idx in range(num_samples):
        row_data = [df.iloc[sample_idx] for df in dataframes]
        merged = pd.concat(row_data, axis=1, ignore_index=True)
        tensors[f"Sample_{sample_idx+1}"] = torch.tensor(
            merged.transpose().values, dtype=torch.float32
        )
    return tensors