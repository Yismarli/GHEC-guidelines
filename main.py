import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Import modules
from ensemblebaselayer import ensemble_base_layer_training
from decision network representation import (
    create_decision_network,
    construct_node_features,
    load_and_preprocess_graph_data
)
from decision network pruning import train_graph_model
from second stage strategy import comparative_trial_classification


def load_and_prepare_data(data_path, target_column=-1, test_size=0.3):
    """
    Load and prepare dataset for training and testing.
    """
    print(f"Loading data from: {data_path}")

    # Load data
    data = pd.read_csv(data_path)

    # Separate features and target
    y = data.iloc[:, target_column]
    X = data.iloc[:, :target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Display class distribution
    class_counts = y.value_counts()
    print("Class distribution:")
    for class_label, count in class_counts.items():
        percentage = count / len(y) * 100
        print(f"  Class {class_label}: {count} samples ({percentage:.2f}%)")

    return X_train, X_test, y_train, y_test


def main():
    """
    Main execution function for Hierarchical Ensemble Classifier.
    """
    print("============ Hierarchical Ensemble Classifier Start ============")

    # Configuration
    DATA_PATH = "D:\\datasets\\classification\\apple_quality\\processed_apple_quality.csv"

    # Step 1: Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(DATA_PATH)

    # Step 2: Base Layer Training
    print("\n--- Step 1: Base Layer Training ---")
    (
        node_predictions,
        node_test_predictions,
        performance_metrics,
        node_count,
        validation_labels,
        validation_features,
        feature_importance_dict,
        classifiers,
        important_features_train,
        important_features_test
    ) = ensemble_base_layer_training(X_train, X_test, y_train, y_test)

    # Convert labels to tensors
    y_train_tensor = torch.tensor(validation_labels.values)
    y_test_tensor = torch.tensor(y_test.values)

    # Step 3: Decision Network Construction
    print("\n--- Step 2: Decision Network Construction ---")
    adjacency_matrix, edge_index = create_decision_network(
        performance_metrics, node_count
    )

    # Step 4: Node Feature Construction
    print("\n--- Step 3: Node Feature Construction ---")
    construct_node_features(
        node_predictions,
        node_test_predictions,
        performance_metrics,
        adjacency_matrix,
        node_count,
        validation_features,
        X_test,
        feature_importance_dict,
        classifiers
    )

    # Step 5: Load Graph Data
    print("\n--- Step 4: Graph Data Preparation ---")
    training_graphs, testing_graphs = load_and_preprocess_graph_data(node_count)

    # Step 6: Graph Neural Network Training
    print("\n--- Step 5: Graph Neural Network Training ---")
    trained_model, removed_nodes = train_graph_model(
        training_graphs,
        testing_graphs,
        y_train_tensor,
        y_test_tensor,
        edge_index,
        node_count,
        classifiers
    )

    print(f"Final removed nodes: {removed_nodes}")

    # Step 7: Comparative Analysis
    print("\n--- Step 6: Comparative Analysis ---")
    comparative_trial_classification(X_train, X_test, y_train, y_test)

    print("============ Hierarchical Ensemble Classifier Complete ============")


if __name__ == "__main__":
    main()


