from datasets import Genius
from synthetic_part_ref import partition_refinement_BE,create_BE_reduced_network
import sys
import os

# Add the directory containing the package to sys.path
package_dir = '../lynkx'
sys.path.append(package_dir)
import homophily

data = Genius(root='./data/').get_data(-1)
measure = homophily.our_measure(data.edge_index, data.y)
print(measure)

coarsest_part = partition_refinement_BE(data, 'genius' + '.ode', 
                                          'genius', 'labels')
red_data = create_BE_reduced_network(data, coarsest_part)
measure = homophily.our_measure(red_data.edge_index, red_data.y)
print(measure)
