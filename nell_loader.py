import torch
from torch_geometric.datasets import NELL
from synthetic_part_ref import partition_refinement_BE

# Define the dataset path where you want to store the data
dataset_path = './data/NELL'

# Load the NELL dataset and normalize the node features
dataset = NELL(root=dataset_path)

# Access the first graph in the dataset (since NELL is a single graph dataset)
data = dataset[0]

# If the feature matrix is sparse, convert it to dense
data.x = data.x.to_dense()

# Set the number of nodes manually, if it's not already set
if data.num_nodes is None:
    data.num_nodes = data.x.size(0)  # Set based on the number of rows in the feature matrix

# Print some basic information about the dataset
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of features per node: {data.num_node_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Train mask size: {data.train_mask.sum().item()}")
print(f"Validation mask size: {data.val_mask.sum().item()}")
print(f"Test mask size: {data.test_mask.sum().item()}")

# Example: accessing node features and edge index
#print("Node feature matrix:", data.x)
#print("Edge index matrix:", data.edge_index)

coarsest_part = partition_refinement_BE(data, 'NELL' + '.ode', 
                                          'NELL', 'labels')
    