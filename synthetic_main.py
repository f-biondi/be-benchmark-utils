import torch
from synthetic_graph import test_graph
import synthetic_part_ref
import synthetic_training
import numpy as np
from torch_geometric.data import Data
from scipy import stats

def confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data)
    n_obs = len(data)
    margin_of_error = (std / np.sqrt(n_obs)) * stats.t.ppf((1 + 0.99) / 2, n_obs - 1)
    return (mean - margin_of_error, mean + margin_of_error)

def print_mask_stats(graph_data):
    print(f"Size Train: {graph_data.train_mask.sum().item()}")
    print(f"  Size Val: {graph_data.val_mask.sum().item()}")
    print(f" Size Test: {graph_data.test_mask.sum().item()}")

# 100,10,5 - 50% of train dataset reduction
# 200,10,10 - 50% of train dataset reduction
data = test_graph(N=200, E=12, M=3) 
assert hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask')
print_mask_stats(data)

# Get partition using new_data (with new labels)
#coarsest_part = synthetic_part_ref.partition_refinement_BE(data, "synth_data.ode", "synth", prepartition=True)
#print(f"coarsest_part: {coarsest_part}")

# Perform BE reduction and save the reduced network
#red_data = synthetic_part_ref.create_BE_reduced_network(data, coarsest_part, np.mean)
#print_mask_stats(red_data)

#quit()

params = {
        'do_prepartition': True,
        'synthetic_features': False,     # creates synthetic features consistent with the BE partition
        'aggregation_func': np.mean,    # aggregation function for features in the same block
        'params_grid': {                # grid of parameters for hyperparameter tuning
                'learning_rate': [ 0.05],
                'num_epochs': [1000],
                'weight_decay': [0.0005],
                'hidden_dim': [64],
                'dropout': [0.5],
        },
        'models': {                     # dictionary of models to be tested
                'GCN': synthetic_training.CustomGCNModel,
                #'GraphConv': GraphConvModel,
                #'SGConv': SGConvModel,
        }
}

# run analysis
acc_full = []
acc_red = []
for i in range(30):
    comp_results = synthetic_training.compare_analysis(dataset_name="synth", graph_name="synth", 
                                              data=data, params=params, red_data_path='synth_red.data')
    if i == 0:
        print_mask_stats(data)
        input()

    for model, results in comp_results['results_full_network'].items():
        for hyperparam_key, result in results.items(): 
            acc_full.append(result['test_accuracy'])
    
    for model, results in comp_results['results_reduced_network'].items():
        for hyperparam_key, result in results.items(): 
            acc_red.append(result['test_accuracy'])
    
print(confidence_interval(acc_full))
print(confidence_interval(acc_red))
# save results
#new_string = f"Graph_Data_Cornell_Randomized_Syn/cornell_graph_{i}"
#save_dictionary(results, new_string)
#print(results)