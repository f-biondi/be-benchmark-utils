import torch
import numpy as np
from synthetic_erode import erode, erodeHandler
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
from datetime import datetime

# Interface has changed: now we can prepartition by 'labels', initial 'features', or
# 'none'    
def partition_refinement_BE(data, fileOut, modelName, prepartition="labels"):
    """Converts a Data object into a linear ERODE model that is written to
    fileOut

    It is converted as a directed graph if forced to do so or if data.is_directed()

    prepartition performs a pre-partitioning of the node set based on its labels
    as found in the PyTorch Data object 'data'

    the coarsest partition is returned as torch tensor that acts as a partition mask
    from nodeidx -> partition label
    """
    print("New version that takes into account train, validation and test masks as boolean masks")
    #assert hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask') and \
    #    data.train_mask.dtype == torch.bool and data.val_mask.dtype == torch.bool and \
    #        data.test_mask.dtype == torch.bool
    
    if data.is_directed():
        print("""Graph is directed. Treating it as such for the partition refinement. Direction
of edges is inferred from the list of edges of the data object.""")
    else:
        print("Graph is undirected.")

    with open(fileOut, "w") as fout:
        fout.write("begin model " + modelName + "\n")
        fout.write("begin init\n")
        fout.writelines(["S{0}\n".format(s) for s in range(data.x.shape[0])])
        fout.write("end init\n")
        # PyERODE does not read the prepartition block!
        fout.write("begin reactions\n")
        # iterates through columns in the edge-index format
        # each element is a tuple
        # ONLY WRITES WHAT IS IN THE EDGE INDEX
        # IF DIRECTED IT NATURALLY WRITES OUT BOTH DIRECTIONS
        for r in zip(*data.edge_index):
            fout.write( "S{1} -> S{0} + S{1}, 1\n".format(r[0],r[1]))
        fout.write("end reactions\n")
        fout.write("end model")
    erode.loadModel(fileOut)
    print(f'Number of variables (original): {erode.getVariablesNum()}')
    if prepartition == 'labels' or prepartition == 'features':
      partition = compute_prepartition(data, prepartition)
      # Put it all together
      partition = partition.tolist()
    elif prepartition == 'none':
      partition = [1]*data.x.shape[0] # basic prepartition with all elements in the same block
    else:
      raise Exception('Prepartition method ' + prepartition + ' not supported')
    print(f'Results with prepartition: {prepartition}')
    start = datetime.now()
    obtained = erode.computeBE(erodeHandler.py_to_j_list(partition))
    end = datetime.now()
    n_blocks = erode.getNumberOfBlocks(obtained)
    print(f'Number of variables (reduced): {n_blocks}')

    coarsest_part = erodeHandler.j_to_py_list(obtained)
    coarsest_part = torch.tensor(coarsest_part)
    # coarsest_part = vettore (len = numero nodi) entry sono le label che indicano il macronodo a cui appartine il nodo considerato
    return ((end - start).total_seconds(), coarsest_part)

# compute prepartition 
# if method is labels then the prepartition is computed based on unique node labels
# if method is features then the prepartition considers initial blocks with equal 
# initial features 
def compute_prepartition(data, method="labels"):
    if method == 'labels':
        return compute_prepartition_labels(data)
    if method == 'features':
        return compute_prepartition_features(data)
    raise Exception('Method' + method + ' not supported')

def compute_prepartition_labels(data):
    print('Computing prepartition based on node labels')
    labels =  torch.unique(data.y)
    partition = torch.zeros(data.x.shape[0], dtype=int)
    num_labels = len(labels)
    num_val = data.val_mask.sum().item()
    num_test = data.test_mask.sum().item()
    # Cannot assume that data.y is a label of integers
    # So I am converting that into a partition mask with integers (starting from 1)
    # 1 - Give same initial block to all train nodes with the same label
    for (k,label) in enumerate(labels):
        partition[(data.y == label) & data.train_mask] = k+1 # this is the value the mask is getting
      # 2 - Give pairwise distinct labels to nodes in validation mask
    start_val = num_labels + 1
    stop_val = num_labels + 1 + num_val
    partition[data.val_mask] = torch.tensor(list(range(start_val,stop_val)), dtype=int)
      # 3 - Give pairwise distinct labels to nodes in train mask
    start_test = stop_val
    stop_test = start_test + num_test
    partition[data.test_mask] = torch.tensor(list(range(start_test,stop_test)), dtype=int)
    return partition

def compute_prepartition_features(data):
    print('Computing prepartition based on node features')
    print('This ignores train, test, and validation masks')
    _, partition = torch.unique(data.x, dim=0, return_inverse=True)
    return partition

def mapping_masks(data, red_data, coarsest_part):
  print(f"MAPPING MASKS---------------------------------------------------")
  print(f"  Original data masks")
  print('     Size of train mask (sum(data.train_mask)): ', torch.sum(data.train_mask))
  print('       Size of val mask (sum(data.val_mask)): ', torch.sum(data.val_mask))
  print('       Size of test mask (sum(data.test_mask)): ', torch.sum(data.test_mask))
  # Extract the number of nodes in the full graph
  num_nodes_full = data.num_nodes
  # Get unique group IDs in the reduced graph
  unique_groups_red = torch.unique(coarsest_part)
  # Create a mapping from original nodes to reduced nodes
  mapping_dict = {group_id.item(): i for i, group_id in enumerate(unique_groups_red)}
  # Initialize masks for the reduced graph
  red_train_mask = torch.zeros(red_data.num_nodes, dtype=torch.bool)
  red_test_mask = torch.zeros(red_data.num_nodes, dtype=torch.bool)
  red_val_mask = torch.zeros(red_data.num_nodes, dtype=torch.bool)
  # Map original nodes to reduced nodes and set the mask values
  for original_node, reduced_node in mapping_dict.items():
      red_train_mask[reduced_node] = data.train_mask[coarsest_part == original_node].any()
      red_test_mask[reduced_node] = data.test_mask[coarsest_part == original_node].any()
      red_val_mask[reduced_node] = data.val_mask[coarsest_part == original_node].any()

  # Create a new red_data with the computed masks
  red_data_with_masks = Data(
      x=red_data.x,
      edge_index=red_data.edge_index,
      edge_weight=red_data.edge_weight,
      y=red_data.y,
      train_mask=red_train_mask,
      test_mask=red_test_mask,
      val_mask=red_val_mask
  )

  print(f"  Reduced data masks")
  print('     Size of train mask (sum(red_data_with_masks.train_mask)): ', torch.sum(red_data_with_masks.train_mask))
  print('       Size of val mask (sum(red_data_with_masks.val_mask)): ', torch.sum(red_data_with_masks.val_mask))
  print('       Size of test mask (sum(red_data_with_masks.test_mask)): ', torch.sum(red_data_with_masks.test_mask))
  print(f"END MAPPING MASKS---------------------------------------------------")

  return red_data_with_masks


def old_create_BE_reduced_network(data, partition, aggregation_func=np.mean):
  assert aggregation_func == np.mean
  """ Creates a reduced network from an original Data object graph with the given
  partition of blocks.

  The partition is given as a mask node index -> partition label.

  By default, it sums across features in the same block.

  It returns the Data object and the partition map, mapping a partition label
  to its representative macronode index (block)
  """
  # Here the assumption is that in the original graph, nodes are 0, ..., n - 1
  # In the reduced graph, macronodes have to be also in the same style, 0, ..., M - 1
  # where M is the number of blocks
  labels, idxs = np.unique(partition, return_index=True)
  M = len(labels)

  # 2024/05/24: This was broken because it may identify fewer nodes if there are isolated nodes
  # 2024/05/24: This IS broken because it may not work with weighted graphs
  A = to_scipy_sparse_matrix(data.edge_index, num_nodes=len(partition)).tocsr()
  original_features = data.x.numpy()
  red_features = None
  red_edge_list = [[],[]]
  red_weight = []
  for src in range(M):
    # macroblocks are created in the same order as partition labels are found
    src_label = labels[src]
    # create reduced features
    src_red_features =  aggregation_func(original_features[partition == src_label, :], axis = 0)
    if red_features is None:
        red_features = src_red_features
    else:
        red_features = np.vstack((red_features, src_red_features))

    # idxs has the representatives of the partition blocks.

    for tgt in range(M):
        # A[idxs[src], partition == labels[tgt]]
        # represents the outgoing rates toward all the elements of the target partition block
        # so np.sum(A[idxs[src], partition == labels[tgt]], axis=1) represents the total outgoing rate

        out_flux = np.sum(A[idxs[src], partition == labels[tgt]], axis=1)
        if out_flux: # need to specify both dimensions
            red_edge_list[0].append(src)
            red_edge_list[1].append(tgt)
            red_weight.append(out_flux.item())
            #if out_flux > 1:
            #    print(f'Adding from {src} to {tgt} with weight {out_flux}')

  red_edge_list = np.array(red_edge_list) # TODO create a tensor directly
  red_weight = torch.tensor(red_weight, dtype=torch.float).squeeze()
  red_y = data.y[idxs] # this is the reduced labels
  x = torch.tensor(red_features, dtype=torch.float)
  edge_index = torch.tensor(red_edge_list, dtype=torch.long)
  red_data = Data(x=x, edge_index=edge_index, edge_weight=red_weight, y=red_y)
  #print('reduced edge weights', red_data.edge_weight)
  #print('Warning: mapping masks not active')
  return mapping_masks(data, red_data, partition)


def create_BE_reduced_network(data, partition, aggregation_func=np.mean):
  assert aggregation_func == np.mean
  """ Creates a reduced network from an original Data object graph with the given
  partition of blocks.

  The partition is given as a mask node index -> partition label.

  By default, it sums across features in the same block.

  It returns the Data object and the partition map, mapping a partition label
  to its representative macronode index (block)
  """
  # red data will be filled with data.edge_list and data.edge_attr
  red_data = compute_lumped_matrix(data, partition)
  # now we need to fill it with data.x  
  red_data.x = aggregate_features_by_partition(data, partition_mask=partition)
  # and data.y
  _, idxs = np.unique(partition, return_index = True)
  red_data.y = data.y[idxs]
  return mapping_masks(data, red_data, partition)

def aggregate_features_by_partition(data, partition_mask):
    """
    Sum features across nodes within the same block defined by the partition mask.

    Args:
    data (Data): PyTorch Geometric data object containing node features x.
    partition_mask (torch.Tensor): 1D tensor containing integer values indicating the partition of nodes.

    Returns:
    torch.Tensor: Tensor containing the summed features for each partition.
    """
    # Ensure partition mask is a tensor
    partition_mask = torch.tensor(partition_mask, dtype=torch.long)
    
    # Map non-contiguous partition mask values to contiguous range
    unique_partitions = np.unique(partition_mask)
    num_partitions = len(unique_partitions)
    
    # Node features
    x = data.x
    summed_features = torch.zeros(num_partitions, x.shape[1])
    for (i,label) in enumerate(unique_partitions):
        summed_features[i] = torch.mean(x[partition_mask==label], 0) 
    return summed_features

def compute_lumped_matrix(data, partition_mask):
    """
    Compute the lumped matrix given a PyTorch Geometric data object and a partition mask.

    Args:
    data (Data): PyTorch Geometric data object containing matrix A.
    partition_mask (torch.Tensor): 1D tensor containing integer values indicating the partition of nodes.

    Returns:
    Data: PyTorch Geometric data object containing the lumped matrix L.
    """
    # Ensure partition mask is a tensor
    partition_mask = torch.tensor(partition_mask, dtype=torch.long)
    
    # Map non-contiguous partition mask values to contiguous range
    unique_partitions, inverse_indices = torch.unique(partition_mask, sorted=True, return_inverse=True)
    num_partitions = unique_partitions.size(0)
    
    # Get the adjacency matrix and other relevant information from the data object
    edge_index = data.edge_index
    edge_weight = data.edge_weight if data.edge_weight is not None else torch.ones(edge_index.size(1))

    row, col = edge_index
    row_partition = inverse_indices[row]
    col_partition = inverse_indices[col]
    
    # Find representative elements for each block
    representative_elements = {}
    for partition in range(num_partitions):
        representative_elements[partition] = (partition_mask == unique_partitions[partition]).nonzero(as_tuple=True)[0][0].item()

    # Initialize the lumped matrix
    lumped_matrix = torch.zeros((num_partitions, num_partitions), dtype=edge_weight.dtype)
    
    # Create a mask for each representative node
    for i in range(num_partitions):
        representative_node = representative_elements[i]
        mask = (row == representative_node)
        masked_col_partition = col_partition[mask]
        masked_edge_weight = edge_weight[mask]
        
        # Aggregate weights for each partition
        aggregated_values = scatter(masked_edge_weight, masked_col_partition, dim=0, dim_size=num_partitions, reduce='sum')
        lumped_matrix[i] = aggregated_values
    
    # Convert the lumped matrix into a PyTorch Geometric data object
    edge_index = torch.nonzero(lumped_matrix, as_tuple=False).t().contiguous()
    edge_weight = lumped_matrix[edge_index[0], edge_index[1]]
    
    lumped_data = Data(edge_index=edge_index, edge_weight=edge_weight)
    
    return lumped_data


def replace_synthetic_matrix(mat, partition):
    """Replaces a matrix of shape (num_nodes, num_features) according to the partition"""
    # Ensure partition is a torch Tensor
    partition = torch.tensor(partition)
    # Initialize the new feature matrix z with zeros having the same shape as data.x
    z = torch.zeros_like(mat)
    # Iterate through each unique partition block
    for block in partition.unique():
        # Find the indices of rows belonging to the current partition block
        indices = (partition == block).nonzero(as_tuple=True)[0]
        # Calculate the mean of the feature vectors for these rows
        mean_vector = torch.mean(mat[indices], dim=0)
        # Assign this mean vector to the corresponding rows in z
        z[indices] = mean_vector
    return z


def replace_synthetic_features(data, partition):
    # Data is the original data object
    print(f"CREATING SYNTHETIC FEATURES ---------------------------------------------")
    print(f"    Original data: {data}")
    z = replace_synthetic_matrix(data.x, partition)
    # Create a new Data object with the new feature matrix z
    new_data = Data(x=z, edge_index=data.edge_index, y=data.y, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
    print(f"    Synthetic data: {new_data}")
    print(f"END CREATING SYNTHETIC FEATURES ---------------------------------------------")
    return new_data

def pretty_print(msg, array):
    print(msg)
    array = array.detach().numpy()
    # Pretty print the NumPy array
    print(np.array2string(array, formatter={'float_kind': lambda x: "%.2f" % x}))
        
if __name__ == '__main__':
    from synthetic_graph import generate_graph
    import time
    
    for i in range(30):
        data = generate_graph(N = 200, E=3, M=2) 
        
        coarsest_part = partition_refinement_BE(data, "synth_data.ode", "synth", prepartition="labels")
        
        a = time.time()
        reduced_data_old = old_create_BE_reduced_network(data, coarsest_part)
        b = time.time() - a
        print(b)
        
        c = time.time()
        reduced_data_new = create_BE_reduced_network(data, coarsest_part)
        d = time.time() - c
        print(d)
        assert torch.equal(reduced_data_old.edge_index, reduced_data_new.edge_index)
        assert torch.equal(reduced_data_old.edge_weight,reduced_data_new.edge_weight)
        assert torch.equal(reduced_data_old.x,reduced_data_new.x)
        assert torch.equal(reduced_data_old.y,reduced_data_new.y)
        print('Test passed', b/d)
    
    

    
