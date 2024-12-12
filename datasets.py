# Large-scale analysis of the different datasets for **node classification task**
# Starting from here as a reference: https://paperswithcode.com/task/node-classification
import torch
import torch_geometric.transforms as T
  
class Dataset:
  
  def __init__(self, family, name, root) -> None:
    """Main root where all the datasets can be found"""
    self.root = root
    self.family = family
    self.name = name

  def train_test_val_split(self, data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of the ratios must be 1."

    # Set the seed for reproducibility
    torch.manual_seed(seed)

    num_nodes = data.num_nodes
    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)
    num_test = num_nodes - num_train - num_val

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    perm = torch.randperm(num_nodes)
    train_idx = perm[:num_train]
    val_idx = perm[num_train:num_train + num_val]
    test_idx = perm[num_train + num_val:]

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
     
  def get_data(self, split=0):
    """Gets a public split, or a random one if split is -1"""
    data = self._get_all_data()
    if split == -1:
      print('Generating random split')
      return self.train_test_val_split(data)

    print('Dataset returning split', split)
    assert data.train_mask.shape == data.val_mask.shape == data.test_mask.shape
    if data.train_mask.dim() == 1:
      return data
    no_splits = data.train_mask.shape[1] 
    assert split < no_splits and split >= 0
    data.train_mask = data.train_mask[:, split]
    data.val_mask = data.val_mask[:, split]
    data.test_mask = data.test_mask[:, split]
    return data

  def _get_all_data(self):
    pass

class Synthetic(Dataset):
  """Wrapper for synthetic datasets"""
  def __init__(self) -> None:
    super().__init__(family="synthetic", name="synthetic", root="./")

  def get_data(self, split=0):
    import synthetic_graph
    import random
    # TODO Pull up at the constructor level
    data = synthetic_graph.generate_graph(N = 100, 
                                          E = random.randint(2,5), M = 5, 
                                          undirected=0.25)
    return data
    

class Actor(Dataset):
  def __init__(self, root) -> None:
    super().__init__(family = "Actor", name="actor", root=root)
    
  def _get_all_data(self):
    from torch_geometric.datasets import Actor
    dataset = Actor(root = self.root)
    return dataset[0]
  

class Genius(Dataset):
  def __init__(self, root):
    super().__init__(root=root, family='LINKX', name='genius')

  def _get_all_data(self):
    from torch_geometric.datasets import LINKXDataset
    dataset = LINKXDataset(root=self.root, name=self.name)
    return dataset[0]
  
# We get good reductions here, but the labels are not categorical 
# The task is regression and we should change the GNN parameters for that
# At present we consistently use CrossEntropyLoss for all cases.
#
# Notice however the following:
# Pei et al. (2020) [Geom-GCN] converted the task to node classification by grouping nodes into
# five categories based on the original regression target, and this preprocessing 
# became standard in the literature. 
# This is performed through the parameter geom_gcn_preprocess set to True
# ['We classify the nodes into five categories in term of the number of the average monthly traffic of the web page.']
# From: A CRITICAL LOOK AT THE EVALUATION OF GNNS UNDER HETEROPHILY: ARE WE REALLY MAKING PROGRESS? ICLR 2023
class Wikipedia(Dataset):
   
  def __init__(self, name, root) -> None:
    super().__init__(family = "WikipediaNetwork", name=name, root=root)
   
  def _get_all_data(self):
    from torch_geometric.datasets import WikipediaNetwork
    dataset = WikipediaNetwork(root = self.root, name = self.name, geom_gcn_preprocess=True)
    return dataset[0] 
 
# ************************************************
# WikipediaNetwork / chameleon
# ************************************************
#     Size original: 2277
#      Size reduced: 797
class Chameleon(Wikipedia):
  
  # crocodile is not available but the preprocessing makes it a classification problem
  def __init__(self, root) -> None:
    super().__init__(name="chameleon", root=root)
  
# ************************************************
# WikipediaNetwork / squirrel
# ************************************************
#     Size original: 5201
#      Size reduced: 2193
# ************************************************
class Squirrel(Wikipedia):
  
  def __init__(self, root) -> None:
    super().__init__(name="squirrel", root=root)

# ************************************************
# WikipediaNetwork / crocodile
# ************************************************
#     Size original: 11631
#      Size reduced:  4828
# ************************************************
class Crocodile(Wikipedia):
  
  def __init__(self, root) -> None:
    super().__init__(name="crocodile", root=root)
  
  def _get_all_data(self):
    from torch_geometric.datasets import WikipediaNetwork
    # crocodile does not want geom_gcn_preprocess
    dataset = WikipediaNetwork(root = self.root, name = self.name, geom_gcn_preprocess=False)
    return dataset[0] 


class CiteSeer(Dataset):

  def __init__(self, root):
    super().__init__(root = root, family = 'CitationFull', name ='CiteSeer')

  def _get_all_data(self):
    from torch_geometric.datasets import CitationFull
    dataset = CitationFull(root = self.root, name = self.name)
    return dataset[0]
    
class HeterophilousGraph(Dataset):
   
  def __init__(self, name, root) -> None:
    super().__init__(family = "HeterophilousGraph", name=name, root=root)
   
  def _get_all_data(self):
    from torch_geometric.datasets import HeterophilousGraphDataset
    dataset = HeterophilousGraphDataset(root = self.root, name = self.name)
    return dataset[0] 

# ************************************************
# HeterophilousGraphDataset / Questions
# ************************************************
#     Size original: 48921
#      Size reduced: 28214
# ************************************************
class Questions(HeterophilousGraph):

  def __init__(self, root) -> None:
    super().__init__(name="Questions", root=root)


# Requires torch sparse, densify all matrices
# Some datasets (Facebook, PPI, ... )
# have labels that are not scalar
# THERE ARE OTHERS, CHECK THEM OUT 
# ************************************************
# AttributedGraph, Wiki, 2405, 2309, 96.01%
# AttributedGraph, BlogCatalog, 5196, 5196, 100.00%
# AttributedGraph, Flickr, 7575, 7573, 99.97%
class AttributedGraph(Dataset):
   
  def __init__(self, name, root) -> None:
    super().__init__(family = "AttributedGraph", name=name, root=root)
   
  def _get_all_data(self):
    from torch_geometric.datasets import AttributedGraphDataset
    dataset = AttributedGraphDataset(root = self.root, name = self.name)
    return dataset[0] 

  def get_data(self, split=0):
    data =  super().get_data(split)
    # need to densify the x matrix
    data.x = data.x.to_dense()
    return data
  
 
# ************************************************
# AttributedGraph / Cora
# ************************************************
#     Size original: 2708
#      Size reduced:  926
class AttributedCora(AttributedGraph):
  def __init__(self, root) -> None:
    super().__init__(name="Cora", root=root)
  
# ************************************************
# AttributedGraph / CiteSeer
# ************************************************
#     Size original: 3312
#      Size reduced:  812
class AttributedCiteSeer(AttributedGraph):
  def __init__(self, root) -> None:
    super().__init__(name="CiteSeer", root=root)

# ************************************************
# AttributedGraph / PubMed
# ************************************************
#     Size original: 19717
#      Size reduced:  2973
class AttributedPubMed(AttributedGraph):
  def __init__(self, root) -> None:
    super().__init__(name="PubMed", root=root)


# Requires torch-sparse, so x is converted to dense format before hand
# Could more elegantly use todense transform, but it does not work 
# Also, data.y is not in the format [0, ..., number of classes] so we 
# need to do a remapping of that
# NELL, NELL, 65755, 11150, 16.96%
class NELL(Dataset):
   
  def __init__(self, root) -> None:
    super().__init__(family = "NELL", name="NELL", root=root)
   
  def _get_all_data(self):
    from torch_geometric.datasets import NELL
    datasets = []
    dataset = NELL(root = self.root)
    return dataset[0] 

  def get_data(self, split=0):
    data =  self._get_all_data()
    # need to densify the x matrix
    data.x = data.x.to_dense()
    print('Shape of data.x', data.x.shape)
    print('Shape of data.y', data.x.shape)
    
    class_labels = torch.unique(data.y)
    new_labels = torch.zeros(data.y.shape,dtype=torch.long) 
    for idx,label in enumerate(class_labels):
      new_labels[data.y == label] = idx
    data.y = new_labels    
    return self.train_test_val_split(data)
  
#Planetoid, Cora, 2708, 2417, 89.25%
#Planetoid, CiteSeer, 3327, 2374, 71.36%
#Planetoid, PubMed, 19717, 13900, 70.50%
def get_planetoid(root):
  from torch_geometric.datasets import Planetoid
  datasets = []
  for name in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root = root, name = name)
    datasets.append(('Planetoid', name, dataset[0]))
  return datasets

#CitationFull, Cora, 19793, 18537, 93.65%
#
# ************************************************
# CitationFull / CiteSeer
# ************************************************
#     Size original: 4230
#      Size reduced: 2260
# Accuracy original: 0.851
#  Accuracy reduced: 0.883
#  Runtime original: 2.205s
#   Reduced reduced: 1.029s
# ************************************************
#CitationFull, DBLP, 17716, 15449, 87.20%
def get_citation_full(root):
  from torch_geometric.datasets import CitationFull
  datasets = []
  for name in ['CiteSeer']: #['Cora', 'CiteSeer', 'DBLP']:
    dataset = CitationFull(root = root, name = name)
    datasets.append(('CitationFull', name, dataset[0]))
  return datasets



# Genius is pretty good!
#LINKXDataset, penn94, 41554, 41523, 99.93%
#LINKXDataset, reed98, 962, 959, 99.69%
#LINKXDataset, amherst41, 2235, 2235, 100.00%
#LINKXDataset, cornell5, 18660, 18608, 99.72%
#LINKXDataset, johnshopkins55, 5180, 5162, 99.65%
#LINKXDataset, genius, 421961, 127594, 30.24%
def get_LINKXDataset(root):
  from torch_geometric.datasets import LINKXDataset
  datasets = []
  for name in ["genius"]: # ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
    dataset = LINKXDataset(root = root, name=name)
    datasets.append(('LINKXDataset', name, dataset[0]))
  return datasets

# no features ... using constant degree transform
# PolBlogs, polblogs, 1490, 1173, 78.72%
def get_polblogs(root):
  from torch_geometric.datasets import PolBlogs
  datasets = []
  transform = T.Constant()
  dataset = PolBlogs(root=root,transform=transform)
  datasets.append(('PolBlogs', 'polblogs', dataset[0]))
  return datasets

#WebKB, Cornell, 183, 139, 75.96%
#WebKB, Texas, 183, 135, 73.77%
#WebKB, Wisconsin, 251, 208, 82.87%
def get_WebKB(root):
  from torch_geometric.datasets import WebKB
  datasets = []
  for name in ["Cornell", "Texas", "Wisconsin"]:
    dataset = WebKB(root = root, name=name)
    datasets.append(('WebKB', name, dataset[0]))
  return datasets

# GitHub, github, 37700, 35616, 94.47%
def get_github(root):
  from torch_geometric.datasets import GitHub
  datasets = []
  dataset = GitHub(root=root)
  datasets.append(('GitHub', 'github', dataset[0]))
  return datasets

#Actor, actor, 7600, 7234, 95.18%
def get_actor(root):
  from torch_geometric.datasets import Actor
  datasets = []
  dataset = Actor(root = root)
  datasets.append(('Actor', 'actor', dataset[0]))
  return datasets

#Coauthor, CS, 18333, 17911, 97.70%
#Coauthor, Physics, 34493, 33673, 97.62%
def get_coauthors(root):
  from torch_geometric.datasets import Coauthor
  datasets = []
  for name in ['CS', 'Physics']:
    dataset = Coauthor(root = root, name = name)
    datasets.append(('Coauthor', name, dataset[0]))
  return datasets
    
#Amazon, Computers, 13752, 13390, 97.37%
#Amazon, Photo, 7650, 7480, 97.78%
def get_amazon(root):
  from torch_geometric.datasets import Amazon
  datasets = []
  for name in ['Computers', 'Photo']:
    dataset = Amazon(root = root, name = name)
    datasets.append(('Amazon', name, dataset[0]))
  return datasets


# does not have features, using constant transform
# EmailEUCore, email_EU_Core, 1005, 990, 98.51%
def get_EmailEUCore(root):
  from torch_geometric.datasets import EmailEUCore
  datasets = []
  transform = T.Constant()
  dataset = EmailEUCore(root = root, transform=transform)
  datasets.append(('EmailEUCore', 'email_EU_Core', dataset[0]))
  return datasets

# FacebookPagePage, fb, 22470, 21291, undirected, 94.75%
def get_FacebookPagePage(root):
  from torch_geometric.datasets import FacebookPagePage
  datasets = []
  dataset = FacebookPagePage(root = root)
  datasets.append(('FacebookPagePage', 'fb', dataset[0]))
  return datasets


# Twitch, DE, 9498, 9404, 99.01%
# Twitch, EN, 7126, 6744, 94.64%
# Twitch, ES, 4648, 4586, 98.67%
# Twitch, FR, 6551, 6516, 99.47%
# Twitch, PT, 1912, 1895, 99.11%
# Twitch, RU, 4385, 4272, 97.42%
def get_twitch(root):
  from torch_geometric.datasets import Twitch
  datasets = []
  for name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']:
    dataset = Twitch(root = root, name=name)
    datasets.append(('Twitch', name, dataset[0]))
  return datasets

# Cannot be easily handled, 2GB ode file
def get_reddit(root):
  from torch_geometric.datasets import Reddit
  datasets = []
  dataset = Reddit(root=root)
  datasets.append(('Reddit', 'reddit', dataset[0]))
  return datasets

# No features, does not work with transform
# To be excluded because it has temporal data
def get_JODIEDataset(root):
  from torch_geometric.datasets import JODIEDataset
  datasets = []
  for name in ["Wikipedia", "MOOC"]: # not including reddit and lastfm because they have only one class
    dataset = JODIEDataset(root = root, name=name)
    datasets.append(('JODIE', name, dataset[0]))
  return datasets

# neither features nor labels.
# Does not seem to be OK for node classification, it has things like data.edge_type
# and data.train_y, etc.
# See, e.g., here: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn.py
def get_Entities(root):
  from torch_geometric.datasets import Entities
  transform = T.Constant()
  datasets = []
  for name in ["AIFB", "MUTAG"]: # not including BGS and AM because too large
    dataset = Entities(root = root, name=name, transform=transform)
    datasets.append(('Entities', name, dataset[0]))
  return datasets

# Double check the paper to see what they do
def get_PascalPF(root):
  from torch_geometric.datasets import PascalPF
  datasets = []
  for name in ["Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", 
               "Chair", "Diningtable", "Dog", "Horse", "Motorbike", "Person", 
               "Pottedplant", "Sheep", "Sofa", "Train", "TVMonitor"]:
    dataset = PascalPF(root = root, category=name)
    datasets.append(('PascalPF', name, dataset[0]))
  return datasets

# Quite large
def get_flickr(root):
  from torch_geometric.datasets import Flickr
  datasets = []
  dataset = Flickr(root = root)
  datasets.append(('Flickr', 'flickr', dataset[0]))
  return datasets

if __name__ == '__main__':
  sq = Squirrel(root='./data/wikipedia_network/')
  print(sq.get_data())

  ch = Chameleon(root='./data/wikipedia_network/')
  print(ch.get_data(), ch.root, ch.family, ch.name)

  #cr = Crocodile(root='./data/wikipedia_network/')
  #print(cr.get_data(), cr.root, cr.family, cr.name)
