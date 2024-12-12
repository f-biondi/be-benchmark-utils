import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import time
import os
from sklearn.model_selection import ParameterGrid
import synthetic_part_ref
import datasets
import scipy.stats as stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

class CustomGCNModel(nn.Module):
    """It is important to use this module, in particular to handle target-to-source message passing"""
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(CustomGCNModel, self).__init__()
        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        # IMPORTANT: OUR GCN MODEL HAS THIS FLOW AT THE MOMENT
        kwargs = {"flow" : "target_to_source"}
        add_self_loops = True
        normalize = True
        bias = False
        self.conv1 = GCNConv(in_channels=self.in_channels, out_channels=self.hidden_dim, bias=bias,
                             normalize=normalize, add_self_loops=add_self_loops, **kwargs)
        self.conv2 = GCNConv(in_channels=self.hidden_dim, out_channels=self.out_channels,bias=bias,
                             normalize=normalize, add_self_loops=add_self_loops, **kwargs)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if not data.edge_weight is None:
            edge_weight = data.edge_weight.squeeze()
        else:
            edge_weight = None
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

def weighted_nll_loss(outputs, data, mask, partition):
    """This is a loss function that weights the loss on macronodes by 
    the number of micronodes that they represent if according to the partition 
    that is passed. Otherwise, it simply computes the weighted nll_loss"""
    if partition is None:
        loss = F.nll_loss(outputs[mask], data.y[mask])
    else:
        loss = F.nll_loss(outputs[mask], data.y[mask], reduction='none')
        # Adjust the loss to account for macronode counts
        unique_labels = np.unique(partition)
        macronode_counts = torch.tensor([torch.sum((partition == label)).item() for label in unique_labels])
        loss = loss * macronode_counts[mask]  
        loss = loss.sum() / macronode_counts[mask].sum()
    return loss

def training(model, data, optimizer, partition):
  """Evaluation on the training set"""
  # set model in training mode
  model.train()
  # reset gradients
  optimizer.zero_grad()
  # forward pass
  outputs = model(data)
  # compute loss using the train mask
  loss = weighted_nll_loss(outputs, data, data.train_mask, partition)
  # backward pass
  loss.backward()
  # update weights
  optimizer.step()
  return loss.cpu().detach().numpy()

def validation(model, data, partition):
  """Evaluation on the validation set"""
  # set model in evaluation mode
  model.eval()
  with torch.no_grad():
      # forward pass
      outputs = model(data)
      # compute loss
      loss = weighted_nll_loss(outputs, data, data.val_mask, partition)
      # get predictions
      predictions = outputs.argmax(dim=1)
      # compute accuracy
      accuracy = ( predictions[data.val_mask] == data.y[data.val_mask]).float().mean().item()
  return loss, accuracy

def test_model(model, data):
  # set model in evaluation mode
  model.eval()
  # Evaluation on the test set
  with torch.no_grad():
      # forward pass
      outputs = model(data)
      # get predictions
      predictions = outputs.argmax(dim=1)
      # compute accuracy
      accuracy = (predictions[data.test_mask] == data.y[data.test_mask] ).float().mean().item()
  return accuracy

def work_dataset(model, data, params, partition):
    """Trains the network according to given data object
    and parameters (which also contain the model)"""
    """If partition is none, it is doing the original model
    Otherwise it is doing the reduced one with the given partition mask"""

    # Parameters
    num_epochs = params['num_epochs']     # number of training epochs for each model
    lr = params['learning_rate']          # learning rate for the optimization algorithm
    weight_decay = params['weight_decay'] # weight decay optimization parameter

    data = data.to(device)

    # Start execution timer
    start = time.time()
    # Define Loss and Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_values = []
    valid_loss_values = []
    valid_accuracy_values = []

    # Variables for early stopping
    best_val_acc = 0.0  # Initialize the best validation accuracy
    best_val_loss = float('inf') # Initialize the best validation loss
    patience = 200  # Default patience

    # Training loop
    for epoch in range(num_epochs):
        # Train the model
        train_loss = training(model=model, data=data, optimizer=optimizer, partition=partition)
        train_loss_values.append(train_loss)

        # Validate the model
        val_loss, val_acc = validation(model=model, data=data, partition=partition)
        valid_loss_values.append(val_loss)
        valid_accuracy_values.append(val_acc)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss

        if epoch >= 0:
          if patience > 0 and epoch > patience:
            tmp = torch.tensor( valid_loss_values[-(patience + 1):-1] )
            if val_loss > tmp.mean().item():
              break

    # Stop runtime
    runtime = time.time() - start

    # Perform Test
    test_accuracy = test_model(model=model, data=data)
    
    # return results
    results = {
        #'train_loss_values': train_loss_values,
        #'valid_loss_values': valid_loss_values,
        #'valid_accuracies': valid_accuracy_values,
        'train_epoch': epoch,
        'accuracy': test_accuracy,
        'runtime': runtime
    }

    print(f"TEST ACCURACY: {test_accuracy}")
    return results

def launch_model(data, red_data, partition, params):
   # perform analysis on full network #############################################
  print('------------------ Learning full network ------------------------------------------------------------------------------------------------------------')
  
  # Initialize the model
  Model = params['model']
  num_features = data.x.shape[1]          # x = [num_nodes, num_node_features]
  num_classes = len(torch.unique(data.y)) # y = [num_nodes, 1]
  #print(f'Working with {num_features} features and {num_classes} labels.')
  #print('   Shape of data.x: ', data.x.shape)
  #print('   Shape of data.y: ', data.y.shape)
  print(f"Working with model: {Model.__name__}")

  model = Model(in_channels=num_features, 
                hidden_dim=params['hidden_dim'], 
                out_channels=num_classes)
  
  # save model random parameters for later reuse in reduced model
  state_file_path = 'gcn_state.pth' # TODO save it somewhere else
  torch.save(model.state_dict(), state_file_path)

  # Put data and model on the device
  model = model.to(device)

  # Perform analysis with different graph neural network models on the given dataset,
  full_results = work_dataset(model=model, data=data, params=params, partition=None)

  # perform analysis on reduced network #############################################
  print('------------------ Learning reduced network ------------------------------------------------------------------------------------------------------------')
  #
  # Load initial random weights
  model.load_state_dict(torch.load(state_file_path))
  # Perform analysis with different graph neural network models on the given dataset,
  red_results = work_dataset(model=model,data=red_data, params=params, partition=partition)
  
  return full_results, red_results

def compare_analysis(dataset:datasets.Dataset, params:dict, split=0):
  
  """
    Compare the analysis between the original graph and the BE reducted graph.
    It saves the reduction (for later reuse) into a file that is uniquely determined
    by the dataset full name and by the split number of the dataset 
  """
  family = dataset.family
  graph_name = dataset.name
  data = dataset.get_data(split=split)

  print(f"-"*200)
  print(f"Dataset name: {family} / {graph_name}")
  
  # Dictionary to store results
  results = {}
  results['dataset_name'] = family
  results['graph_name'] = graph_name
  results['org_size'] = data.x.shape[0]   
  results['features'] = data.x.shape[1]
  results['is_directed'] = data.is_directed()
  results['full_train_mask_count'] = torch.sum(data.train_mask)
  results['val_mask_count'] = torch.sum(data.val_mask)
  results['test_mask_count'] = torch.sum(data.test_mask)
  
  # obtain reduced network ######################################################
  print('------------------ Computing reduced network ------------------------------------------------------------------------------------------------------------')
  print(f'{graph_name} is directed: {data.is_directed()}')
  # Start partition refinement timer
  par_ref_start = time.time()
  # Perform partition refinement and save the coarsest partition

  full_path = os.path.join(dataset.root, family + "_" + 
                           graph_name + "_split" + str(split) + ".ode")
  # Make new different label for each test and valid node
  coarsest_part = synthetic_part_ref.partition_refinement_BE(data, full_path, 
        graph_name, prepartition=params['do_prepartition'])

  # Save partition refinement time
  par_ref_end = time.time() - par_ref_start
  print(f'...done in {par_ref_end:.3f} s')
  results['part_ref'] = par_ref_end
  # Save reduced network size
  reduced_size = len(np.unique(coarsest_part))
  results['red_size'] = reduced_size

  # Create synthetic features if needed
  if params['synthetic_features'] == True:
    data = synthetic_part_ref.replace_synthetic_features(data=data, partition=coarsest_part)

  
  # Perform BE reduction and save the reduced network
  reduced_name = family + "_" + graph_name + "_split" + str(split) + ".reduction"
  red_data_path = os.path.join(dataset.root, reduced_name)
  
  if os.path.exists(red_data_path):
    print('Loading reduced network from:', red_data_path)
    red_data = torch.load(red_data_path)
  else:
    print('Creating BE reduced network...')
    # Start BE reduction timer
    be_red_start = time.time()
    red_data = synthetic_part_ref.create_BE_reduced_network(data, coarsest_part, 
                                                          params['aggregation_func'])
    # Save BE reduction time
    be_red_end = time.time() - be_red_start
    print(f'...done in {be_red_end:.3f} s')
    results['reduced_model_build'] = be_red_end
    torch.save(red_data, red_data_path)
    print('Saved reduced network to:', red_data_path)
     
  #print(f"Created red_data: {red_data}")

  results['red_train_mask_count'] = torch.sum(red_data.train_mask)
  
  full_accuracies = []
  red_accuracies = []
  for i in range(params['runs']):
    full_results, red_results = launch_model(data=data, 
                                           red_data=red_data, 
                                           partition=coarsest_part, 
                                           params=params)
    full_accuracies.append(full_results['accuracy'])
    red_accuracies.append(red_results['accuracy'])

  results['full_accuracies'] = full_accuracies
  results['red_accuracies'] = red_accuracies
  # Print results #################################################################
  print('*'*100)
  print(f'Summary of: {family} / {graph_name}')
  print('*'*100)
  print(f"    Size original: {results['org_size']}")
  print(f"     Size reduced: {results['red_size']}")
  print(f"  Train reduction: {results['red_train_mask_count']/float(results['full_train_mask_count'])}")
  print(f" Time partition refinement: {results['part_ref']}s")
  #print(f" Time BE reduction: {results['reduced_model_build']}s")
  print('*'*100)
  print(f"\n")

  return results


def compute_95_ci(data):
    # Convert the list to a NumPy array for easier calculations
    data = np.array(data)
    
    # Calculate the mean
    mean = np.mean(data)
    
    # Calculate the standard error of the mean (SEM)
    sem = stats.sem(data)
    
    # Determine the critical value (z-score) for a 95% confidence interval
    confidence_level = 0.95
    degrees_freedom = len(data) - 1
    critical_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    
    # Calculate the margin of error
    margin_of_error = critical_value * sem
    
    # Compute the confidence interval
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return (ci_lower, ci_upper)

if __name__ == '__main__':
    params = {
        'do_prepartition': 'labels',
        'synthetic_features': True,     # creates synthetic features consistent with the BE partition
        'aggregation_func': np.mean,    # aggregation function for features in the same block
        'learning_rate': 0.05,
        'runs': 10,                     # number of independent runs with random initial weights
        'num_epochs': 1000,
        'weight_decay': 0.0005,
        'hidden_dim': 64,
        'model': CustomGCNModel
    }
    import random
    import synthetic_graph
    import datasets
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
   
    
    syn = datasets.Synthetic()

    # TODO consider forcing dataset to be symmetric 
    
    #sq = datasets.Squirrel(root='./data/wikipedia_network/')
    # run analysis
    
    results = compare_analysis(syn, split=0, params=params)
    print('Full mean', np.mean(results['full_accuracies']))
    print('Red mean', np.mean(results['red_accuracies']))

    print('Full ci', compute_95_ci(results['full_accuracies']))
    print('Red ci', compute_95_ci(results['red_accuracies']))
     
    # save results
    #new_string = f"Graph_Data_Cornell_Randomized_Syn/cornell_graph_{i}"
    #save_dictionary(results, new_string)
    print(results)    
        


