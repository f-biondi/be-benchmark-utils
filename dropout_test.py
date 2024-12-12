import torch
import torch_geometric
import torch_geometric.transforms as T
from synthetic_part_ref import partition_refinement_BE
from datasets import Chameleon,Genius,Squirrel
# Example: Applying edge dropout in PyTorch Geometric
from torch_geometric.utils import dropout_edge

dropout_rate = 0.0  # Dropout probability
N = 5

def get_data():
    data = Squirrel(root='./data/wikipedia_network').get_data()
    return data

full_p = partition_refinement_BE(get_data(), "temp.ode", "temp", prepartition="none")
size_p = 0
for i in range(N):
    # Assume edge_index is the edge list of the graph
    # Apply edge dropout during training
    data = get_data()
    data.edge_index, _ = dropout_edge(data.edge_index, p=dropout_rate)
    drop_p = partition_refinement_BE(data, "temp.ode", "temp", prepartition="none")
    size_p += len(torch.unique(drop_p))

print(f'Full: {len(torch.unique(full_p))}, Dropped: {1.*size_p/N}')
