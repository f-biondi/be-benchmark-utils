from torch_geometric.data import Data
from synthetic_part_ref import partition_refinement_BE

import sys
import os

# Add the directory containing the package to sys.path
package_dir = '../lynkx'
sys.path.append(package_dir)

# Now you can import the package
import dataset
import homophily

def transform_pokec_to_pyg(pokec_dataset):
    """
    Transforms the output of load_pokec_mat into a PyTorch Geometric data object.
    
    Args:
        pokec_dataset (NCDataset): The dataset object returned by load_pokec_mat.
        
    Returns:
        data (Data): A PyTorch Geometric data object.
    """
    # Extract necessary components from the pokec_dataset
    edge_index = pokec_dataset.graph['edge_index']
    node_feat = pokec_dataset.graph['node_feat']
    label = pokec_dataset.label

    # Create PyTorch Geometric Data object
    pyg_data = Data(x=node_feat, edge_index=edge_index, y=label)

    return pyg_data

if __name__ == '__main__':
    # Assuming load_pokec_mat() is already defined and returns a dataset object
    pokec_dataset = dataset.load_genius()

    # Transform to PyTorch Geometric data object
    pyg_data = transform_pokec_to_pyg(pokec_dataset)
    
    # Print the PyTorch Geometric data object
    print(pyg_data)
    print(f'Number of nodes: {pyg_data.num_nodes}')
    print(f'Number of edges: {pyg_data.num_edges}')
    print(f'Number of features: {pyg_data.num_node_features}')

    measure = homophily.our_measure(pyg_data.edge_index, pyg_data.y)
    #print(pyg_data.edge_index)
    #print(pyg_data.y)
    print(measure)
    #coarsest_part = partition_refinement_BE(pyg_data, 'pokec' + '.ode', 
    #                                      'pokec', 'none')