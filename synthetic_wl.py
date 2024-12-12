from torch_geometric.nn import WLConv
import torch
import numpy as np
from synthetic_part_ref import partition_refinement_BE, compute_prepartition
from torch_geometric.data import Data

    
def normalize_labels(labels):
    label_map = {}
    normalized_labels = torch.zeros_like(labels)
    next_label = 0
    for label in labels:
        if label not in label_map:
            label_map[label] = next_label
            next_label += 1
        normalized_labels[labels == label] = label_map[label]
    return normalized_labels

def partition_refinement_WL(data, prepartition, iters=float('inf')):
    """Performs partition refinement using 1-WL.
    Returns the coarsest stable refinement of prepartition or the
    refinement after iters steps"""
    wl = WLConv()
    x = prepartition
    x_norm = normalize_labels(x)
    #print('x', x)
    k = 0
    while True:
        x_new = wl(x, data.edge_index)
        k += 1
        x_new_norm = normalize_labels(x_new)
        if k==iters or torch.equal(x_norm,x_new_norm):
            print(f'k = {k} iterations')
            return x_new_norm
        else:
            # reset the hashmap not to accumulate ids in memory
            wl.reset_parameters() 
            x = x_new
            x_norm = x_new_norm

def test_compare_wl_synthetic():
    """
    Compare WL-2 and WL-infty (BE) on synthetic graphs
    """
    from synthetic_graph import generate_graph
    import time 
    N = 10000
    speed_up = []
    for i in range(3):
        data = generate_graph(N = N, E=3, M=2) 
        prepartition = compute_prepartition(data) #torch.ones(data.x.shape[0], dtype=torch.long)
        
        start_wl = time.time()
        wl_infty = normalize_labels(partition_refinement_WL(data, prepartition)) 
        stop_wl = time.time() - start_wl 
        print(stop_wl)
        
        wl_2 = normalize_labels(partition_refinement_WL(data, prepartition,iters=2))
        print(f"Size: {len(prepartition)},WL-2: {len(np.unique(wl_2))}, WL-infty: {len(np.unique(wl_infty))}")

        start_be = time.time()
        be = normalize_labels(partition_refinement_BE(data, "synth_data.ode", "synth", prepartition="labels"))
        stop_be = time.time() - start_be 
        print(stop_be)
        
        if i > 0:
            speed_up.append(stop_be/stop_wl)
        assert torch.equal(wl_infty, be)
    print(speed_up)
    print(np.mean(speed_up))
    

def test_compare_wl_benchmarks():
  from datasets import (Actor,
      Chameleon, Squirrel, Crocodile,
        CiteSeer,
        Questions,
        AttributedCora, AttributedCiteSeer, AttributedPubMed,NELL
  )

  from synthetic_part_ref import compute_prepartition
  benchmarks = []
  public_split = True
  
  # RESULTS WHEN THE WL-TEST *DOES NOT* USE THE TRANSPOSED MATRIX
  # Family: WikipediaNetwork, Name: chameleon, Directed: True, Public: True, Size: 2277, WL-2: 1566, WL-infty: 1566, BE: 1718
  # Family: WikipediaNetwork, Name: squirrel, Directed: True, Public: True, Size: 5201, WL-2: 3736, WL-infty: 3736, BE: 4164
  # Family: WikipediaNetwork, Name: crocodile, Directed: False, Public: False, Size: 11631, WL-2: 6414, WL-infty: 6414, BE: 6414
  # Family: HeterophilousGraph, Name: Questions, Directed: False, Public: True, Size: 48921, WL-2: 39672, WL-infty: 39801, BE: 39801
  # Family: AttributedGraph, Name: Cora, Directed: True, Public: False, Size: 2708, WL-2: 1850, WL-infty: 1950, BE: 1436
  # Family: AttributedGraph, Name: CiteSeer, Directed: True, Public: False, Size: 3312, WL-2: 1722, WL-infty: 1758, BE: 1550
  # Family: AttributedGraph, Name: PubMed, Directed: True, Public: False, Size: 19717, WL-2: 9917, WL-infty: 10300, BE: 6752
  
  # RESULTS WHEN THE WL-TEST *DOES USE* THE TRANSPOSED MATRIX
  #Family: WikipediaNetwork, Name: chameleon, Directed: True, Public: True, Size: 2277, WL-2: 1718, WL-infty: 1718, BE: 1718
  #Family: WikipediaNetwork, Name: squirrel, Directed: True, Public: True, Size: 5201, WL-2: 4164, WL-infty: 4164, BE: 4164
  #Family: WikipediaNetwork, Name: crocodile, Directed: False, Public: False, Size: 11631, WL-2: 6414, WL-infty: 6414, BE: 6414
  #Family: HeterophilousGraph, Name: Questions, Directed: False, Public: True, Size: 48921, WL-2: 39672, WL-infty: 39801, BE: 39801
  #Family: AttributedGraph, Name: Cora, Directed: True, Public: False, Size: 2708, WL-2: 1390, WL-infty: 1436, BE: 1436
  #Family: AttributedGraph, Name: CiteSeer, Directed: True, Public: False, Size: 3312, WL-2: 1521, WL-infty: 1550, BE: 1550
  #Family: AttributedGraph, Name: PubMed, Directed: True, Public: False, Size: 19717, WL-2: 6751, WL-infty: 6752, BE: 6752



  #Family: WikipediaNetwork, Name: chameleon, Directed: True, Public: True, WL-0: 1190, WL-1: 1711, WL-2: 1718, WL-infty: 1718, BE: 1718
  #Family: AttributedGraph, Name: CiteSeer, Directed: True, Public: False, WL-0: 669, WL-1: 1316, WL-2: 1521, WL-infty: 1550, BE: 1550
  #Family: WikipediaNetwork, Name: squirrel, Directed: True, Public: True, WL-0: 2710, WL-1: 4128, WL-2: 4164, WL-infty: 4164, BE: 4164 
  #Family: AttributedGraph, Name: Cora, Directed: True, Public: False, WL-0: 549, WL-1: 1176, WL-2: 1390, WL-infty: 1436, BE: 1436
  #Family: AttributedGraph, Name: PubMed, Directed: True, Public: False, WL-0: 3947, WL-1: 6525, WL-2: 6751, WL-infty: 6752, BE: 6752
  #Family: HeterophilousGraph, Name: Questions, Directed: False, Public: True, WL-0: 24463, WL-1: 35632, WL-2: 39672, WL-infty: 39801, BE: 39801
  
  #benchmarks.append((Chameleon(root='./data/wikipedia_network/'), public_split))
  #benchmarks.append((AttributedCiteSeer(root='./data/attributed/'), not public_split))
  #benchmarks.append((Squirrel(root='./data/wikipedia_network/'), public_split))
  #benchmarks.append((AttributedCora(root='./data/attributed/'), not public_split))
  #benchmarks.append((AttributedPubMed(root='./data/attributed/'), not public_split))
  #benchmarks.append((Questions(root='./data/questions/'), public_split))
  #benchmarks.append((NELL(root='./data/nell/'), public_split))
  #benchmarks.append((Crocodile(root='./data/wikipedia_network/'), not public_split))
  #benchmarks.append((Actor(root='./data/'),  public_split))

  for (benchmark, public_split) in benchmarks:
    print(f'Analysing {benchmark.name}')
    if public_split:
        data = benchmark.get_data()
    else:
        data = benchmark.get_data(-1)
    
    # compute be on the normal data
    be = normalize_labels(
        partition_refinement_BE(data, "temp.ode", "temp", prepartition="labels"))
    # if it is directed, we need to transpose the matrix
    if data.is_directed():
        print(f"Family: {benchmark.family}, Name: {benchmark.name}, is directed. Working with transpose.")
        # here we are copying, but we can probably just swap edge indices
        data = transpose_data(data)
        
    prepartition = compute_prepartition(data)
    wl_infty = normalize_labels(partition_refinement_WL(data, prepartition)) 
    wl_1 = normalize_labels(partition_refinement_WL(data, prepartition,iters=1))
    wl_2 = normalize_labels(partition_refinement_WL(data, prepartition,iters=2))
    print(f"Family: {benchmark.family}, Name: {benchmark.name}, Directed: {data.is_directed()}, Public: {public_split}, Size: {len(prepartition)}, WL-0: {len(np.unique(prepartition))}, WL-1: {len(np.unique(wl_1))}, WL-2: {len(np.unique(wl_2))}, WL-infty: {len(np.unique(wl_infty))}, BE: {len(np.unique(be))}")

def transpose_data(data):
    # Transpose the edge index
    transposed_edge_index = data.edge_index[[1, 0], :]

    # Create a new Data object with the transposed edge index
    transposed_data = Data(
        x=data.x,
        edge_index=transposed_edge_index,
        edge_attr=data.edge_attr,
        y=data.y,
        val_mask = data.val_mask,
        train_mask = data.train_mask,
        test_mask= data.test_mask
    )

    # Copy any additional attributes present in the original data object
    #for attr in set(data.keys) - {'edge_index', 'x', 'edge_attr', 'y'}:
    #    transposed_data[attr] = data[attr]

    return transposed_data

if __name__ == '__main__':
    test_compare_wl_benchmarks()

            

        

