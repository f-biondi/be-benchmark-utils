from torch_geometric.data import Data
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt

def to_networkx(data):
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes

    # Adding nodes and their features
    for i in range(num_nodes):
        G.add_node(i)

    # Adding edges
    for i in range(edge_index.shape[1]):
        src, dest = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dest)
    return G

def visualize_graph(data, node_size=100, node_color='skyblue', with_labels=True):
    G = to_networkx(data)
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)  # Positions for all nodes using Spring layout

    # Create labels dictionary: node number and label from data.y
    labels = {i: f"{i}: {data.y[i].item()}" for i in range(data.num_nodes)}

    # Draw the graph
    nx.draw(G, pos, with_labels=with_labels, labels=labels, node_color=node_color, 
            node_size=node_size, edge_color='k', linewidths=1, font_size=15)
    plt.show()


def generate_graph(N, M, E, train=0.6, val=0.2, undirected=1.0):
    """Generate a random graph with N disconnected nodes
    Each such node is connected to a random number of new nodes
    between 0 and E
    Each nodes has a random class between 0 and M
    Undirected gives the probability with which the graph is undirected
    """
    assert train + val < 1.0
    assert undirected <= 1.0
    print("Initializing random seed")
    random.seed(0)
    k = 0
    edges = []
    node_features = []  
    hubs = [] # gets the ids of the hibs
    # Track the index for newly added nodes
    node_id = 0
    
    # Add random nodes to each ring node
    for _ in range(N):
        num_new_nodes = random.randint(0, E)
        node_features.append([random.randint(0,M)])  # Update hub node feature with the number of new connections
        current_hub = node_id
        hubs.append(current_hub)
        node_id += 1
        for _ in range(num_new_nodes):
            edges.append([current_hub, node_id])  # Connect hub node to new node
            if random.random() <= undirected:
                edges.append([node_id, current_hub])  # Undirected 
            node_features.append([random.randint(0,M)]) 
            k+=1
            node_id += 1


    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    # Creates mask by partitioning the hub nodes
    train_id = hubs[int(len(hubs)*train)]       # get the last train hub id
    val_id = hubs[int(len(hubs)*(train+val))]   # get the last val hub id
    print(f"Train id: {train_id}, Val_id: {val_id}")
    
    mask = torch.zeros(len(node_features), dtype=torch.bool)
    mask[0:train_id] = True
    train_mask = mask
    
    mask = torch.zeros(len(node_features), dtype=torch.bool)
    mask[train_id:val_id] = True
    val_mask = mask
    
    mask = torch.zeros(len(node_features), dtype=torch.bool)
    mask[val_id:] = True
    test_mask = mask
    data = Data(x=x, edge_index=edge_index, num_nodes=len(node_features), train_mask=train_mask,
                val_mask=val_mask, test_mask=test_mask)
    # add labels
    data.y = data.x[:,0].long()    
    return data


if __name__ == '__main__':
    data = generate_graph(N=10, M=1, E=2)
    visualize_graph(data)