import networkx as nx
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import os


def create_decision_network(performance_metrics, node_count):
    """
    Create decision network based on model performance metrics.
    """
    print("================== Decision Network Construction Start ==================")

    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((node_count, node_count))

    # Create directed edges based on balanced accuracy comparison
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                # Get balanced accuracy for both nodes
                acc_i = performance_metrics.loc[
                    performance_metrics['Node'] == i, 'BalancedAccuracy'
                ].values[0]
                acc_j = performance_metrics.loc[
                    performance_metrics['Node'] == j, 'BalancedAccuracy'
                ].values[0]

                # Create directed edge from better to worse performer
                if acc_i > acc_j:
                    adjacency_matrix[i, j] = 1

    print("Adjacency matrix created:")
    print(adjacency_matrix)

    # Convert to sparse matrix and edge index tensor
    sparse_matrix = coo_matrix(adjacency_matrix)
    rows, cols = sparse_matrix.nonzero()

    edge_index_np = np.array([rows, cols])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    print("Edge index tensor created")

    print("================== Decision Network Construction Complete ==================")

    return adjacency_matrix, edge_index


def construct_node_features(node_predictions, node_test_predictions, performance_metrics,
                            adjacency_matrix, node_count, validation_data, test_data,
                            feature_importance_dict, classifiers, top_n=10):
    """
    Construct node feature matrices for graph neural network.
    """
    print("================== Node Feature Construction Start ==================")

    # Calculate node degrees
    out_degrees = np.sum(adjacency_matrix == 1, axis=1)
    in_degrees = np.sum(adjacency_matrix == 1, axis=0)

    # Normalize degrees
    out_degrees = out_degrees / np.linalg.norm(out_degrees) if np.linalg.norm(out_degrees) > 0 else out_degrees
    in_degrees = in_degrees / np.linalg.norm(in_degrees) if np.linalg.norm(in_degrees) > 0 else in_degrees

    # Process each node
    for node_idx in range(node_count):
        print(f"Processing node {node_idx}")

        # Get classifier information
        model_name, classifier = classifiers[node_idx]
        features_info = feature_importance_dict[model_name]

        # Extract important features
        if 'feature_names' in features_info:
            selected_features = validation_data[features_info['feature_names'][:top_n]]
        else:
            print(f"No feature importance available for model {model_name}")
            selected_features = pd.DataFrame()

        # Construct training node features
        node_predictions_col = node_predictions.iloc[:, node_idx]
        metrics_row = performance_metrics.iloc[node_idx]

        training_features = pd.DataFrame({
            'NodePredictions': node_predictions_col,
            'BalancedAccuracy': metrics_row['BalancedAccuracy'],
            'TPR': metrics_row['TPR'],
            'TNR': metrics_row['TNR'],
            'OutDegree': out_degrees[node_idx],
            'InDegree': in_degrees[node_idx]
        })

        # Combine with important features
        selected_features_reset = selected_features.reset_index(drop=True)
        selected_features_reset.columns = list(range(top_n))

        training_node_data = pd.concat([training_features, selected_features_reset], axis=1)

        # Save training node data
        training_filename = f"node_training_features_{node_idx}.csv"
        training_node_data.to_csv(training_filename, index=False)
        print(f"Training features saved to {training_filename}")

        # Construct testing node features
        test_predictions_col = node_test_predictions.iloc[:, node_idx]

        if 'feature_names' in features_info:
            test_selected_features = test_data[features_info['feature_names'][:top_n]]
        else:
            test_selected_features = pd.DataFrame()

        testing_features = pd.DataFrame({
            'NodePredictions': test_predictions_col,
            'BalancedAccuracy': metrics_row['BalancedAccuracy'],
            'TPR': metrics_row['TPR'],
            'TNR': metrics_row['TNR'],
            'OutDegree': out_degrees[node_idx],
            'InDegree': in_degrees[node_idx]
        })

        test_selected_features_reset = test_selected_features.reset_index(drop=True)
        test_selected_features_reset.columns = list(range(top_n))

        testing_node_data = pd.concat([testing_features, test_selected_features_reset], axis=1)

        # Save testing node data
        testing_filename = f"node_testing_features_{node_idx}.csv"
        testing_node_data.to_csv(testing_filename, index=False)
        print(f"Testing features saved to {testing_filename}")

    print("================== Node Feature Construction Complete =================="")

    return True


def load_and_preprocess_graph_data(node_count):
    """
    Load and preprocess node feature files into graph data tensors.
    """
    print("================== Graph Data Loading Start ==================")

    # Load training data
    training_files = [f"node_training_features_{i}.csv" for i in range(node_count)]
    training_tensors = load_node_files(training_files, node_count)

    # Load testing data
    testing_files = [f"node_testing_features_{i}.csv" for i in range(node_count)]
    testing_tensors = load_node_files(testing_files, node_count)

    # Clean up temporary files
    cleanup_files(training_files + testing_files)

    print("================== Graph Data Loading Complete ==================")

    return training_tensors, testing_tensors


def load_node_files(file_list, node_count):
    """
    Load multiple node feature files and combine into tensor format.
    """
    loaded_data = {}

    for idx, filename in enumerate(file_list):
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            loaded_data[f"Node_{idx}"] = df

    # Merge data across nodes
    merged_tensors = {}
    num_samples = len(loaded_data['Node_0']) if loaded_data else 0

    for sample_idx in range(num_samples):
        sample_data = []

        for node_idx in range(node_count):
            node_key = f"Node_{node_idx}"
            if node_key in loaded_data:
                sample_data.append(loaded_data[node_key].iloc[sample_idx])

        if sample_data:
            merged_df = pd.concat(sample_data, axis=1, ignore_index=True)
            merged_tensors[f"Sample_{sample_idx + 1}"] = torch.tensor(
                merged_df.transpose().values, dtype=torch.float32
            )

    return merged_tensors


def cleanup_files(file_list):
    """
    Remove temporary files.
    """
    for filename in file_list:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Removed temporary file: {filename}")
        except Exception as e:
            print(f"Error removing file {filename}: {e}")


