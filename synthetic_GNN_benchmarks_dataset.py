import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import Data
import synthetic_part_ref
from torch_geometric.transforms import Constant

# Function to combine all graphs in the dataset into a single big graph and set masks
def combine_all_graphs_to_single(train_dataset, val_dataset, test_dataset):
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
    
    # Helper function to process each split
    def process_split(data_list, mask):
        nonlocal node_offset
        for data in data_list:
            # Apply Constant if no node features are present
            if data.x is None:
                data = constant_transform(data)
    
            # Shift the node indices in the edge_index
            edge_index_shifted = data.edge_index + node_offset

            # Append the shifted edge_index and the node features
            combined_edge_index.append(edge_index_shifted)
            combined_x.append(data.x)

            # Inherit node labels from the graph label
            node_labels = torch.full((data.num_nodes,), data.y.item(), dtype=torch.long)
            combined_node_labels.append(node_labels)

            # Extend the mask for the current graph's nodes
            mask.extend([True] * data.num_nodes)
            node_offset += data.num_nodes
    
    # Process each split
    process_split(train_dataset, train_mask)
    process_split(val_dataset, val_mask)
    process_split(test_dataset, test_mask)

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

if __name__ == '__main__':
    # Load the GNNBenchmarkDataset for the 'PATTERN' collection
    collection_name = 'CYCLES'  # Replace with 'CLUSTER' or others as needed

    # Load each split separately
    train_dataset = GNNBenchmarkDataset(root='/tmp/' + collection_name, name=collection_name, split='train')
    val_dataset = GNNBenchmarkDataset(root='/tmp/' + collection_name, name=collection_name, split='val')
    test_dataset = GNNBenchmarkDataset(root='/tmp/' + collection_name, name=collection_name, split='test')
    
    # Combine all graphs into a single graph
    combined_graph = combine_all_graphs_to_single(train_dataset, val_dataset, test_dataset)
    
    # Output the combined graph
    print('Combined Graph:')
    print(f'Number of nodes: {combined_graph.num_nodes}')
    print(f'Number of edges: {combined_graph.num_edges}')
    print(f'Node labels: {combined_graph.y}')
    print(f"Unique node labels: {torch.unique(combined_graph.y)}")
    print(f'Train mask sum: {combined_graph.train_mask.sum()}')
    print(f'Val mask sum: {combined_graph.val_mask.sum()}')
    print(f'Test mask sum: {combined_graph.test_mask.sum()}')
    coarsest_part = synthetic_part_ref.partition_refinement_BE(combined_graph, collection_name + ".ode", collection_name, prepartition='none')
    reduced_size = len(torch.unique(coarsest_part))
    print(f"Reduced size: {reduced_size}")