from torch_geometric.data import Data
import json
from synthetic_part_ref import partition_refinement_BE, compute_prepartition
import os
import numpy as np
from datetime import datetime
from synthetic_wl import partition_refinement_WL
import torch


def get_geometric_data(file_name,start):
  out = ""
  with open(file_name, "r") as f:
    lines = f.read().split("\n")[:-1]
    head = lines[2].split(" ")
    nodes = int(head[2])
    edges = int(head[4])
    out = out + f"{nodes}\n"
    out = out + f"{edges}\n"
    max_node = 0
    edge_index = [[],[]]
    for l in lines[4:]:
      l = l.split('\t')
      node_start = int(l[0]) - start
      node_end = int(l[0]) - start
      out = out + f"{node_start} {node_end}\n"
      edge_index[0].append(node_start)
      edge_index[1].append(node_end)
      max_node = max(max_node, node_start)
      max_node = max(max_node, node_end)
    print(f"Nodes: {nodes} Max Node: {max_node}")
    f = open("graph.txt","w")
    f.write(out)
    f.close()
    return Data(
        edge_index=torch.tensor(edge_index),
        x = torch.tensor([[] for _ in range(nodes)]),
        num_nodes=nodes
        )

"""
from datasets import Questions, Chameleon, Crocodile
import torch_geometric.transforms as T

data = None
with open("graph.txt", "r") as f:
    lines = f.read().split("\n")[:-2]
    print(lines)
    nodes = int(lines[0])
    edges = int(lines[1])
    edge_index = [[], []]
    for l in lines[2:]:
        l = l.split(" ")
        start = int(l[0])
        end = int(l[1])
        edge_index[0].append(start)
        edge_index[1].append(end)
    data = Data(
        edge_index=torch.tensor(edge_index),
        x = torch.tensor([[] for _ in range(nodes)]),
        y = torch.tensor([[] for _ in range(nodes)]),
        num_nodes=nodes
    )


#data = Chameleon(root='./data/wikipedia_network/').get_data()
#data = Crocodile(root='./data/wikipedia_network/').get_data(-1)
#prepartition = torch.zeros(data.num_nodes, dtype=torch.long)
#wl_infty = partition_refinement_WL(data, prepartition)
print(data)

java_time,java = partition_refinement_BE(data, "temp.ode", "temp", prepartition="none")
print("Java:", len(np.unique(java)))
#print("WL:", torch.unique(wl_infty).shape[0])            
"""
get_geometric_data("snap.txt", 0)
