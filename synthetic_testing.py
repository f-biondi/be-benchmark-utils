import torch
import numpy as np
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import synthetic_part_ref
import synthetic_graph
import synthetic_training
from synthetic_part_ref import create_BE_reduced_network
from synthetic_training import CustomGCNModel

def rows_allclose(tensor, tol=1e-5):
    """Check if all rows are equal up to a tolerance"""
    first_row = tensor[0]
    for row in tensor[1:]:
        if not torch.allclose(row, first_row, atol=tol):
            return False
    return True

def test_equal_outputs(output, partition):
    """Tests that the model output is coherent with the given partition"""
    label_blocks = np.unique(partition)
    for label in label_blocks:
        idxs = (partition == label)
        if torch.sum(idxs) == 1:
            continue
        selected_rows = output[idxs, :]
        if not rows_allclose(selected_rows):
            print('Label block:', label)
            print(selected_rows)
            raise Exception("Test on equal outputs not passed")
    
def simulate_gcn_layer(weight, data):
    """Creates a GCN layer with target to source direction explicitly"""
    num_nodes = data.x.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[data.edge_index[0], data.edge_index[1]] = data.edge_weight if data.edge_weight is not None else 1
    # Add self-loops
    adj_with_self_loops = adj + torch.eye(num_nodes) 
    # Compute the (out-)degree matrix for target-to-source
    degree = torch.sum(adj_with_self_loops, dim=1)
    degree_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
    # Compute the normalized adjacency matrix
    adj_norm = degree_inv_sqrt @ adj_with_self_loops @ degree_inv_sqrt
    # Perform the GCN layer computation
    h = adj_norm @ data.x @ weight
    return h

def simulate_Ax(data):
    """Executes A @ data.x """
    num_nodes = data.x.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[data.edge_index[0], data.edge_index[1]] = data.edge_weight if data.edge_weight is not None else 1
    return adj @ data.x

def compare_full_reduced(full_output, red_output, partition, data, compare_label=False):
    """Compare that the full output agrees with the reduced  output
    according to the given partition

    If label is True then it compares the predicted label
    Otherwise it compares the softmax output
    """
    label_blocks = np.unique(partition)
    for (macronode,label) in enumerate(label_blocks):
        idxs = (partition == label)
        # compare the representative
        if compare_label:
            full_label = full_output[idxs][0,:].argmax().item()
            red_label = red_output[macronode,:].argmax().item()
            #print(f'Full label: {full_label}, red label: {red_label}, true label:{data.y[idxs][0].item()}')
            assert full_label == red_label, "Label comparison failed at block " + str(macronode)
        else:
            if not (torch.allclose(full_output[idxs][0,:],red_output[macronode,:])):
                print('full shape', full_output.shape)
                print('red shape', red_output.shape)
                print('partition')
                print(partition)
                print('full output')
                print(full_output[idxs])
                print('reduced output')
                print(red_output[macronode,:])
                raise Exception(f"Softmax comparison failed at block {macronode}")


def run_simple_training(model, data, epochs, partition=None):
    """Runs the model and returns inference performed after each epoch
    The number of samples in number of epochs + 1. The first inference 
    is related to the output with the initialized weights

    Computes the unnormalized loss if there is no partition
    Otherwise it computes the weighted loss by counting how many original
    nodes the given macronode represents
    """
    inference = []
    losses = []
    # perform training 
    optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0005)
    # set model in training mode
    for epoch in range(epochs):
        model.train()
        # reset gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(data)
        inference.append(outputs)

        loss = synthetic_training.weighted_nll_loss(outputs=outputs, data=data, 
                                                    mask=data.train_mask, partition=partition)
        # compute loss
        # if partition is None:
        #     loss = F.nll_loss(outputs[data.train_mask], data.y[data.train_mask])
        # else:
        #     loss = F.nll_loss(outputs[data.train_mask], data.y[data.train_mask], reduction='none')
        #     # Adjust the loss to account for macronode counts
        #     unique_labels = np.unique(partition)
        #     macronode_counts = torch.tensor([torch.sum((partition == label)).item() for label in unique_labels])
        #     loss = loss * macronode_counts[data.train_mask]  
        #     loss = loss.sum() / macronode_counts[data.train_mask].sum()
        
        losses.append(loss.item())
        
        # backward pass
        loss.backward()
        
        #for name,param in model.named_parameters():
        #    if param.requires_grad:
        #        print(f"{name} gradient: {param.grad}")

        # update weights
        optimizer.step()
            
    model.eval()
    inference.append(model(data))

    return (inference, losses)

def perform_test(data, hidden_dim, epochs):
    """Performs all coherence tests on a simple data set"""
    in_channels = data.x.shape[1]
    
    coarsest_part = synthetic_part_ref.partition_refinement_BE(data, "synth_data.ode", "synth", prepartition="labels")
    #print(f"coarsest_part: {coarsest_part}")
    
    data = synthetic_part_ref.replace_synthetic_features(data, coarsest_part)
    # tests that synthetic outputs are generated correctly
    test_equal_outputs(data.x, coarsest_part)
    
    # tests that A @ data.x is BE consistent
    print("Testing AX")
    Ax = simulate_Ax(data)
    test_equal_outputs(Ax, coarsest_part)
    
    model = CustomGCNModel(in_channels=in_channels, hidden_dim=hidden_dim, 
                        out_channels=len(torch.unique(data.y)))
    
    state_file_path = 'gcn_state.pth'
    torch.save(model.state_dict(), state_file_path)
    
    print("Test GCN Layer simulation output")
    gcn_layer_sim_output = simulate_gcn_layer(model.conv1.lin.weight.t(), data)
    test_equal_outputs(gcn_layer_sim_output, coarsest_part)
   
    print("Test GCN Layer output")
    gcn_layer_output = model.conv1.forward(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight)
    #print('SIM BLOCK')
    #print(gcn_layer_sim_output[coarsest_part == 58,:])
    #
    #print('TRUE BLOCK')
    #print(gcn_layer_output[coarsest_part == 58,:])
    test_equal_outputs(gcn_layer_output, coarsest_part)

    print("Test GCN Model output")
    model.eval()
    gcn_model_output = model(data)
    test_equal_outputs(gcn_model_output, coarsest_part)
   
    full_inference, full_losses = run_simple_training(model, data, epochs=epochs, partition=None)
    
    # check that outputs are coherent with the coarsest partition 
    for (i,out) in enumerate(full_inference):
        print(f"Epoch {i}")
        test_equal_outputs(out, coarsest_part)

    red_data = create_BE_reduced_network(data, coarsest_part, aggregation_func=np.mean)
    red_model = CustomGCNModel(in_channels=in_channels, hidden_dim=hidden_dim, 
                        out_channels=len(torch.unique(red_data.y)))
    # load the weights into the reduced model
    red_model.load_state_dict(torch.load(state_file_path))
    
    red_inference, red_losses = run_simple_training(red_model, red_data, epochs=epochs, partition=coarsest_part)
    print('Full loss', full_losses)
    print(' Red loss', red_losses)
    
    assert torch.allclose(torch.tensor(full_losses), torch.tensor(red_losses), atol=1e-4), "Losses not close"        
    
    for (epoch, (full, red)) in enumerate(zip(full_inference,red_inference)):
        print('Epoch', epoch+1)
        compare_full_reduced(full, red, coarsest_part, data, compare_label=False)
    
    print("Test passed.")

def make_symmetric(data):
    edge_index = data.edge_index
    
    # Add reverse edges
    row, col = edge_index
    rev_edge_index = torch.stack([col, row], dim=0)
    
    # Combine original and reverse edges
    all_edges = torch.cat([edge_index, rev_edge_index], dim=1)
    
    # Remove duplicate edges
    all_edges = torch.unique(all_edges, dim=1)
    
    # Create new Data object with symmetric edges
    data.edge_index = all_edges
    return data

def test_random_graphs(num_graphs, hidden_dim, undirected):
    """Tests randomly generated graphs"""
    import random 
    for i in range(num_graphs):
        print(f"*******\nTest {i}\n*******")
        #data = test_graph(N=4, E=3, M=2) # this DOES GIVE a bug 
        data = synthetic_graph.generate_graph(N = random.randint(5,50), E = random.randint(2,5), M = 5, 
                                              undirected=undirected)
        perform_test(data, hidden_dim=hidden_dim, epochs=5)    

def test_citation(make_undirected = True):
    import datasets
    root = "./data/citation_full/"
    ds = datasets.get_wikipedia_network(root=root)
    print(ds[0])
    data = ds[0][-1] 
    split_no = 2 # gets a public split
    
    data.train_mask = data.train_mask[:, split_no]
    data.val_mask = data.val_mask[:,split_no]
    data.test_mask = data.test_mask[:, split_no]
    
    if make_undirected and data.is_directed():
        print('Making the graph symmetric')
        data = make_symmetric(data)
    perform_test(data, hidden_dim=32, epochs=10)

def test_wikipedia(split_no = 0):
    import datasets
    root = "./data/wikipedia_network/"
    ds = datasets.get_wikipedia_network(root=root)
    print(ds[0])
    data = ds[0][-1] 
    print(data)
    data.train_mask = data.train_mask[:, split_no]
    data.val_mask = data.val_mask[:, split_no]
    data.test_mask = data.test_mask[:, split_no]
    if data.is_directed():
        print('Making the graph symmetric')
        data = make_symmetric(data)
    return data, root

if __name__ == '__main__':
    pass 
    #print("Undirected random graphs")
    test_random_graphs(10, hidden_dim=250, undirected = 1.0)
    
    #print("Directed random graphs")
    test_random_graphs(20, hidden_dim=250, undirected = 0.25)
    
    #data = test_wikipedia()
    #perform_test(data, hidden_dim=32, epochs=10)
    
    #data = synthetic_graph.test_graph(N=4, E=3, M=2) # this used to give a bug
    #perform_test(data, hidden_dim=3, epochs=5)
