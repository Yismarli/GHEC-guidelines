"""
Graph Neural Network training and iterative node pruning based on test-set correlation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops


class GCN(nn.Module):
    """Simple GCN with residual connection for node-level regression."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim * 2)
        self.proj = nn.Linear(input_dim, hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        conv_out = self.conv1(x, edge_index)
        proj_out = self.proj(x)
        combined = F.relu(conv_out + proj_out)
        output = self.linear(combined)
        return output.squeeze()


def filter_edges(edge_index, node_count, nodes_to_remove):
    """Remove edges incident to given nodes."""
    if not nodes_to_remove:
        return edge_index
    mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    for node in nodes_to_remove:
        connected = (edge_index[0] % node_count == node) | (edge_index[1] % node_count == node)
        mask &= ~connected
    return edge_index[:, mask]


def extract_node_features_from_df(df, node_count, removed_nodes):
    """
    From a DataFrame with structure:
        cols 0..node_count-1: node original feature (x_data)
        cols node_count..2*node_count-1: GNN node outputs
        col -2: pooled prediction
        col -1: true label
    Return:
        - features: only GNN outputs of remaining nodes (columns node_count..2*node_count-1 excluding removed)
        - labels: true labels
        - pooled_preds: pooled predictions (unused here)
    """
    # Build list of columns to keep from the GNN output block
    all_gnn_cols = list(range(node_count, 2 * node_count))
    # Remove columns corresponding to already removed nodes
    keep_cols = [col for col in all_gnn_cols if (col - node_count) not in removed_nodes]
    features = df.iloc[:, keep_cols]
    labels = df.iloc[:, -1]
    pooled = df.iloc[:, -2]
    return features, labels, pooled


def train_and_test(epochs, device, optimizer, node_count, removed_nodes,
                   train_loader, test_loader, model, criterion, meta_model):
    """
    Train GNN for epochs, then evaluate on test set and select next node to remove.
    Returns the next node to remove (int).
    """
    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch.x.to(device)
            edge_idx = batch.edge_index.to(device)
            y = batch.y.to(device)

            if removed_nodes:
                edge_idx = filter_edges(edge_idx, node_count, removed_nodes)

            output = model(x, edge_idx)  # shape: (batch_size * node_count,)

            # Pool over nodes
            batch_size = y.size(0)
            pooled = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                start = i * node_count
                end = start + node_count
                pooled[i] = output[start:end].mean()

            loss = criterion(pooled, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

    # Collect training data for meta-model (from last epoch, but we don't have it stored)
    # The original code used the last epoch's `last_xgnn` to fit meta-model.
    # To replicate, we must rerun a forward pass without gradients to get the node outputs.
    # However, the original `train_and_test` used `last_xgnn` which was built during training.
    # To keep it simple, we can collect it during the last epoch or after training.
    # For now, we follow the original pattern: we'll re-evaluate on train set after training.
    model.eval()
    train_features_list = []
    train_labels_list = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch.x.to(device)
            edge_idx = batch.edge_index.to(device)
            y = batch.y.to(device)
            if removed_nodes:
                edge_idx = filter_edges(edge_idx, node_count, removed_nodes)
            output = model(x, edge_idx)
            batch_size = y.size(0)
            # Build the combined matrix: x_data + output_reshape + pooled + y
            x_reshaped = x.view(batch_size, node_count, -1)
            x_data = x_reshaped[:, :, 0]  # first feature per node
            output_reshaped = output.view(batch_size, node_count)
            pooled = output_reshaped.mean(dim=1, keepdim=True)
            y_reshaped = y.unsqueeze(1)
            combined = torch.cat([x_data, output_reshaped, pooled, y_reshaped], dim=1)
            train_features_list.append(combined.cpu())
            train_labels_list.append(y.cpu())

    train_combined = torch.cat(train_features_list, dim=0)
    train_df = pd.DataFrame(train_combined.numpy())
    # Extract features (GNN outputs of remaining nodes)
    train_X, train_y, _ = extract_node_features_from_df(train_df, node_count, removed_nodes)
    meta_model.fit(train_X, train_y)

    # Test set evaluation and node selection
    model.eval()
    test_features_list = []
    test_labels_list = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch.x.to(device)
            edge_idx = batch.edge_index.to(device)
            y = batch.y.to(device)
            if removed_nodes:
                edge_idx = filter_edges(edge_idx, node_count, removed_nodes)
            output = model(x, edge_idx)
            batch_size = y.size(0)
            x_reshaped = x.view(batch_size, node_count, -1)
            x_data = x_reshaped[:, :, 0]
            output_reshaped = output.view(batch_size, node_count)
            pooled = output_reshaped.mean(dim=1, keepdim=True)
            y_reshaped = y.unsqueeze(1)
            combined = torch.cat([x_data, output_reshaped, pooled, y_reshaped], dim=1)
            test_features_list.append(combined.cpu())
            test_labels_list.append(y.cpu())

    test_combined = torch.cat(test_features_list, dim=0)
    test_df = pd.DataFrame(test_combined.numpy())
    test_X, test_y, _ = extract_node_features_from_df(test_df, node_count, removed_nodes)

    # Predict using meta-model
    y_pred = meta_model.predict(test_X)

    # Compute metrics
    acc = accuracy_score(test_y, y_pred)
    prec = precision_score(test_y, y_pred, zero_division=0)
    f1 = f1_score(test_y, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(test_y, y_pred)
    tn, fp, fn, tp = confusion_matrix(test_y, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (fp + tn) if (fp + tn) > 0 else 0
    print(f"Test - F1: {f1:.4f}, BalAcc: {bal_acc:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, TPR: {tpr:.4f}, TNR: {tnr:.4f}")

    # Compute correlation of each node's output (GNN outputs) with the meta-model prediction
    # In original, they correlated each node's GNN output with y_pred (meta prediction)
    # We need node outputs from test set for remaining nodes.
    # We have test_X which contains only GNN outputs of remaining nodes.
    # To correlate with y_pred, we use the same order.
    # But original used xgnn_test_x which was the GNN outputs, and y_pre2 (meta prediction).
    # So we compute corr for each column in test_X with y_pred.
    corr_list = []
    for col in test_X.columns:
        node_vals = test_X[col].values
        corr, _ = pearsonr(node_vals, y_pred)
        if np.isnan(corr):
            corr = 0.0
        corr_list.append((col, corr))

    # Filter out already removed nodes (but col names are already only remaining nodes)
    # Original also excluded nodes in `wait_del_test_node_all` which was removed_nodes + node_nums? Actually original had:
    #   wait_del_test_node_all = [value + node_nums for value in nodes_to_remove_all]
    # This was wrong because col names are already in the range node_nums..2*node_nums-1.
    # We correctly handle it: col names are actual column indices, and removed_nodes are indices < node_nums.
    # So we filter columns whose (col - node_nums) is in removed_nodes. But we already filtered them out.
    # So we just pick the column with smallest correlation.
    if not corr_list:
        print("No columns to evaluate.")
        return None

    # Select column with minimal correlation
    worst_col, min_corr = min(corr_list, key=lambda x: x[1])
    node_to_remove = int(worst_col) - node_count  # map back to node index (0..node_count-1)
    print(f"Worst node: {node_to_remove} (corr={min_corr:.4f})")
    return node_to_remove


def clf_graph(training_graphs, testing_graphs, y_train, y_test, edge_index, node_count, classifiers):
    """
    Main function to run GNN training and iterative node pruning.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine input feature dimension
    sample_key = next(iter(training_graphs))
    input_dim = training_graphs[sample_key].shape[1]

    model = GCN(input_dim, 128, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Prepare datasets
    def create_dataset(graph_dict, labels):
        graph_list = []
        for idx, (key, value) in enumerate(graph_dict.items()):
            x = value
            y = torch.tensor([labels[idx]], dtype=torch.float64)
            graph_list.append(Data(x=x, edge_index=edge_index, y=y))
        return graph_list

    train_data = create_dataset(training_graphs, y_train)
    test_data = create_dataset(testing_graphs, y_test)

    batch_size = 200
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    meta_model = MLPClassifier(max_iter=200, solver='adam', hidden_layer_sizes=128)

    removed_nodes = []
    epochs = 200

    for iteration in range(node_count - 1):
        print(f"Removal iteration {iteration+1}/{node_count-1}, removed: {removed_nodes}")
        next_node = train_and_test(epochs, device, optimizer, node_count, removed_nodes,
                                   train_loader, test_loader, model, criterion, meta_model)
        if next_node is not None:
            removed_nodes.append(next_node)
        else:
            break
        torch.cuda.empty_cache()

    print(f"Final removed nodes: {removed_nodes}")
    return model, removed_nodes