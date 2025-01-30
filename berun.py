from synthetic_part_ref import partition_refinement_BE, compute_prepartition
from torch_geometric.data import Data
import torch
import numpy as np
import multiprocessing
import time
import os

graph_file = open("graph.txt","r").read().split("\n")[:-1]
partition_file = open("partition.txt","r").read().split("\n")[:-1]

edge_index = [[],[]]
nodes = int(graph_file[0])

for l in graph_file[2:]:
    l = l.split(" ")
    edge_index[0].append(int(l[0]))
    edge_index[1].append(int(l[1]))

features = []
for l in partition_file:
    features.append([int(l)])

data = Data(
      edge_index=torch.tensor(edge_index),
      x = torch.tensor(features),
      num_nodes=nodes
)

java_time,java = partition_refinement_BE(data, "temp.ode", "temp", prepartition="features")

with open("res.txt", "w") as f:
    f.write(f"{len(np.unique(java))}\n")
    f.write(f"{java_time}")

