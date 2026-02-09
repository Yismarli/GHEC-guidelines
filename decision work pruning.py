import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.neural_network import MLPClassifier
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for ensemble decision learning.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim * 2)
        self.projection = nn.Linear(input_dim, hidden_dim * 2)
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Graph convolution
        conv_output = self.conv1(x, edge_index)

        # Feature projection
        projected_input = self.projection(x)

        # Combine and activate
        combined = F.relu(conv_output + projected_input)

        # Final output
        output = self.output_layer(combined)

        return output.squeeze()


class GraphDataset:
    """
    Dataset for graph neural network training and testing.
    """

    def __init__(self, graph_data, labels, edge_index):
        self.graph_data = graph_data
        self.labels = labels
        self.edge_index = edge_index

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        sample_key = f"Sample_{idx + 1}"
        if sample_key in self.graph_data:
            x = self.graph_data[sample_key]
            y = torch.tensor([self.labels[idx]], dtype=torch.float64)

            return Data(x=x, edge_index=self.edge_index, y=y)
        else:
            raise IndexError(f"Sample {idx} not found in graph data")


def filter_edges_by_nodes(edge_index, node_count, nodes_to_remove):
    """
    Remove edges connected to specified nodes.
    """
    if not nodes_to_remove:
        return edge_index

    # Create mask to keep edges
    mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

    for node in nodes_to_remove:
        # Find edges connected to this node
        connected_mask = (
                (edge_index[0] % node_count == node) |
                (edge_index[1] % node_count == node)
        )
        mask &= ~connected_mask

    # Apply mask
    filtered_edges = edge_index[:, mask]

    return filtered_edges


def extract_graph_features(data_frame, removed_nodes, node_count):
    """
    Extract features from graph data frame.
    """
    # Get columns to keep
    columns_to_keep = []
    for col in range(30):  # Assuming 30 columns in the data frame
        if col not in removed_nodes and (col + 15) not in removed_nodes:
            columns_to_keep.append(col)

    # Extract features (using columns > 14 as per original logic)
    feature_columns = [col for col in columns_to_keep if col > 14]

    if feature_columns:
        features = data_frame.iloc[:, feature_columns]
    else:
        features = pd.DataFrame()

    # Extract labels and predictions
    labels = data_frame.iloc[:, -1]
    predictions = data_frame.iloc[:, -2]

    return features, labels, predictions


def train_graph_model(training_graphs, testing_graphs, train_labels, test_labels,
                      edge_index, node_count, classifiers, epochs=200):
    """
    Train and evaluate graph neural network.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup model
    node_features = training_graphs['Sample_1'].shape[1]
    model = GraphNeuralNetwork(node_features, 128, 1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Create datasets
    train_dataset = GraphDataset(training_graphs, train_labels, edge_index)
    test_dataset = GraphDataset(testing_graphs, test_labels, edge_index)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100)

    # Meta-model for final predictions
    meta_model = MLPClassifier(max_iter=500, solver='adam', hidden_layer_sizes=128)

    # Track nodes to remove
    removed_nodes = []

    # Training loop
    for removal_iteration in range(node_count - 1):
        print(f"Removal iteration {removal_iteration + 1}")

        # Train GNN
        train_gnn(model, optimizer, criterion, train_loader, device,
                  node_count, removed_nodes, epochs)

        # Evaluate and select next node to remove
        next_node_to_remove = evaluate_and_select_node(
            model, test_loader, device, node_count, removed_nodes,
            train_labels, test_labels, meta_model
        )

        if next_node_to_remove is not None:
            removed_nodes.append(next_node_to_remove)
            print(f"Added node {next_node_to_remove} to removal list")

        torch.cuda.empty_cache()

    return model, removed_nodes


def train_gnn(model, optimizer, criterion, data_loader, device,
              node_count, removed_nodes, epochs):
    """
    Train the graph neural network.
    """
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0

        for batch in data_loader:
            optimizer.zero_grad()

            # Move data to device
            x = batch.x.to(device)
            edge_idx = batch.edge_index.to(device)
            y = batch.y.to(device)

            # Filter edges if nodes are removed
            if removed_nodes:
                edge_idx = filter_edges_by_nodes(edge_idx, node_count, removed_nodes)

            # Forward pass
            output = model(x, edge_idx)

            # Pool outputs across nodes
            batch_size = y.size(0)
            pooled_outputs = []

            for i in range(batch_size):
                start_idx = i * node_count
                end_idx = start_idx + node_count
                node_outputs = output[start_idx:end_idx]
                pooled = node_outputs.mean(dim=0, keepdim=True)
                pooled_outputs.append(pooled)

            pooled_tensor = torch.cat(pooled_outputs, dim=0).to(device)
            pooled_tensor = pooled_tensor.squeeze()

            # Calculate loss
            loss = criterion(pooled_tensor, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")


def evaluate_and_select_node(model, data_loader, device, node_count, removed_nodes,
                             train_labels, test_labels, meta_model):
    """
    Evaluate GNN predictions and select worst performing node.
    """
    model.eval()

    # Collect predictions
    train_features = []
    test_features = []

    with torch.no_grad():
        # Process training data
        for batch in data_loader:
            x = batch.x.to(device)
            edge_idx = batch.edge_index.to(device)
            y = batch.y

            if removed_nodes:
                edge_idx = filter_edges_by_nodes(edge_idx, node_count, removed_nodes)

            output = model(x, edge_idx)

            # Reshape and extract features
            batch_size = y.size(0)
            x_reshaped = x.view(batch_size, node_count, x.size(1))
            x_data = x_reshaped[:, :, 0]

            output_reshaped = output.view(batch_size, node_count)

            # Pool outputs
            pooled_outputs = []
            for i in range(batch_size):
                node_outputs = output_reshaped[i]
                pooled = node_outputs.mean(dim=0, keepdim=True)
                pooled_outputs.append(pooled)

            pooled_tensor = torch.cat(pooled_outputs, dim=0)

            # Combine features
            combined = torch.cat([
                x_data.cpu(),
                output_reshaped.cpu(),
                pooled_tensor.cpu(),
                y.unsqueeze(1).cpu()
            ], dim=1)

            if data_loader.dataset == train_labels:
                train_features.append(combined)
            else:
                test_features.append(combined)

    # Process features
    if train_features:
        train_tensor = torch.cat(train_features, dim=0)
        train_df = pd.DataFrame(train_tensor.numpy())

        # Extract features for meta-model
        train_X, train_y, _ = extract_graph_features(train_df, removed_nodes, node_count)

        if not train_X.empty:
            meta_model.fit(train_X, train_y)

    if test_features:
        test_tensor = torch.cat(test_features, dim=0)
        test_df = pd.DataFrame(test_tensor.numpy())

        # Extract features for meta-model
        test_X, test_y, _ = extract_graph_features(test_df, removed_nodes, node_count)

        if not test_X.empty and hasattr(meta_model, 'predict'):
            predictions = meta_model.predict(test_X)

            # Calculate metrics
            accuracy = accuracy_score(test_y, predictions)
            precision = precision_score(test_y, predictions)
            f1 = f1_score(test_y, predictions)
            balanced_acc = balanced_accuracy_score(test_y, predictions)

            # Confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
            tpr = tp / (tp + fn)
            tnr = tn / (fp + tn)

            print(f"Test Metrics - F1: {f1:.6f}, Balanced Acc: {balanced_acc:.6f}, "
                  f"Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, "
                  f"TPR: {tpr:.6f}, TNR: {tnr:.6f}")

    # Select worst performing node (simplified logic)
    # In practice, you would analyze node contributions here
    if removed_nodes and len(removed_nodes) < node_count - 1:
        return len(removed_nodes)  # Simple sequential removal

    return None