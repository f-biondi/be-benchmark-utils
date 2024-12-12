import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.transforms import Constant
import synthetic_part_ref
import random

# Function to combine all graphs in the dataset into a single big graph
def combine_all_graphs_to_single(dataset, train_indices, val_indices, test_indices):
    # Define a constant transform for adding synthetic features if needed
    constant_transform = Constant(value=1.0) 
    combined_edge_index = []
    combined_x = []
    combined_node_labels = []
    
    # Masks for training, validation, and test sets
    train_mask = []
    val_mask = []
    test_mask = []

    # Keep track of node index shifts
    node_offset = 0
    
    for i, data in enumerate(dataset):
        # Apply Constant if no node features are present
        if data.x is None:
            data = constant_transform(data)
      
        # Shift the node indices in the edge_index
        edge_index_shifted = data.edge_index + node_offset
        
        # Append the shifted edge_index and the node features
        combined_edge_index.append(edge_index_shifted)
        combined_x.append(data.x)
        
        # Create node labels based on the graph label
        node_labels = torch.full((data.num_nodes,), data.y.item(), dtype=torch.long)
        combined_node_labels.append(node_labels)
        
        # Determine the mask for the current graph's nodes
        if i in train_indices:
            train_mask.extend([True] * data.num_nodes)
            val_mask.extend([False] * data.num_nodes)
            test_mask.extend([False] * data.num_nodes)
        elif i in val_indices:
            train_mask.extend([False] * data.num_nodes)
            val_mask.extend([True] * data.num_nodes)
            test_mask.extend([False] * data.num_nodes)
        else:
            train_mask.extend([False] * data.num_nodes)
            val_mask.extend([False] * data.num_nodes)
            test_mask.extend([True] * data.num_nodes)
        
        # Update the node offset for the next graph
        node_offset += data.num_nodes

    # Combine the list of edge indices, node features, and node labels into single tensors
    combined_edge_index = torch.cat(combined_edge_index, dim=1)
    combined_x = torch.cat(combined_x, dim=0)
    combined_node_labels = torch.cat(combined_node_labels, dim=0)
    
    # Create a new Data object for the combined graph with node labels and masks
    combined_data = Data(
        x=combined_x, 
        edge_index=combined_edge_index, 
        y=combined_node_labels,
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool)
    )
    
    return combined_data

# GNNBenchmarkDataset needs to be treated differently because it allows
# you to lead splits in terms of train, test and validation graphs.
# graph tasks discussed here https://arxiv.org/pdf/2003.00982

# check out OGB https://ogb.stanford.edu/docs/graphprop/#ogbg-mol

# *** TU Dataset ***   NO PREPARTITIONING
# Check that there are no node labels, edge labels, edge attributes
# COLLAB  372474, 158273 
# DD, 334925, 327813
# IMDB-BINARY 19773, 3595
# IMDB-MULTI 19502, 2111
# PROTEINS 43471,  37830
# REDDIT-BINARY 859254, 336507
# REDDIT-MULTI-5K, 2542071, 1159998
# github_stargazers, 1448038, 896431
# deezer_ego_nets, 226213, 138711
# twitch_egos, TODO!

def process_graph_classification_dataset(dataset):
    # Split the dataset into train, val, and test sets
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Define the split ratios
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    
    # Calculate split indices
    train_split = int(train_ratio * len(indices))
    val_split = train_split + int(val_ratio * len(indices))
    
    # Split indices for training, validation, and testing
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    # Combine all graphs into a single graph
    combined_graph = combine_all_graphs_to_single(dataset, train_indices, val_indices, test_indices)
    return combined_graph

if __name__ == '__main__':
    name = 'IMDB-MULTI'
    dataset = TUDataset(root='/tmp/'+name, name=name)
    
    #from torch_geometric.datasets import QM7b
    #dataset = QM7b(root='/path/to/dataset')
    
    # Combine all graphs into a single graph
    combined_graph = process_graph_classification_dataset(dataset=dataset)
    
    # Output the combined graph
    print('Combined Graph:')
    print(f'Number of nodes: {combined_graph.num_nodes}')
    print(f'Number of edges: {combined_graph.num_edges}')
    print(f'Node labels: {combined_graph.y}')
    print(f"Unique node labels: {torch.unique(combined_graph.y)}")
    print(f'Train mask sum: {combined_graph.train_mask.sum()}')
    print(f'Val mask sum: {combined_graph.val_mask.sum()}')
    print(f'Test mask sum: {combined_graph.test_mask.sum()}')
    
    # this is the maximal partitioning 
    prepartition = 'labels'
    coarsest_part = synthetic_part_ref.partition_refinement_BE(
        combined_graph, name + ".ode", name, prepartition=prepartition)
    reduced_size = len(torch.unique(coarsest_part))
    print(f"Reduced size: {reduced_size}")