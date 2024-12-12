from datasets import (Squirrel,Chameleon,Questions,AttributedPubMed, 
                      AttributedCora,AttributedCiteSeer,get_actor,get_LINKXDataset, NELL)
from synthetic_part_ref import partition_refinement_BE
import torch
# Creates the table showing the size of the reduced network when attempting
# the exact reduction of the network up to unique _initial features_

def reduction_by_initial_features(dataset):
  data = dataset.get_data(split=-1) # gets a random split because train/test/val are ignored  
  coarsest_part = partition_refinement_BE(data, dataset.name + '.ode', 
                                          dataset.name, 'features')
  print(dataset.name, data.x.shape[0], len(torch.unique(coarsest_part)))

if __name__ == '__main__':
    ds = [
      #Chameleon(root='./data/wikipedia_network/'),
      #Squirrel(root='./data/wikipedia_network/'), 
      #Questions(root='./data/heterophilius'),
      #AttributedPubMed(root='./data/attributed'),
      #AttributedCiteSeer(root='./data/attributed'),
      #AttributedCora(root='./data/attributed')
    ]
    #for dataset in ds:
    #    reduction_by_initial_features(dataset)
    
    #ds = get_actor('./data')
    #data = ds[0][2]
    #coarsest_part = partition_refinement_BE(data, 'actor' + '.ode', 
    #                                      'actor', 'features')
  
    #ds = get_LINKXDataset('./data/lynkx')
    
    #data = ds[0][2]
    #coarsest_part = partition_refinement_BE(data, 'genius' + '.ode', 
    #                                      'genius', 'none')
    
    ds = NELL('./data')
    coarsest_part = partition_refinement_BE(ds.get_data(), 'NELL' + '.ode', 
                                          'NELL', 'none')
    



  