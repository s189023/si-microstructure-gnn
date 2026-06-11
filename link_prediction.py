import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, negative_sampling
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, to_hetero
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import random
import math
from scipy import stats

# load data
data = np.load('/Users/rustgi/Desktop/data_master.npy', allow_pickle = True)

data[0]

data.ndim

# 1d numpy array bc single list of elements, where each element is 2D array or none

data[0][1] # None

# Nones are to input data for different lasers (not same format as this data - just scalars (energy densities))

for value in data[0]: 
    print(f"Value: {value}, Type: {type(value)}")

# within each 2D array, last col = last row (grain sizes) 

# have 268 tuples, where middle element is None -- filtering out the None placeholders

filtered_data = [
    tuple(x for x in item if x is not None) if isinstance(item, (list, tuple)) else item
    for item in data
]

# for item in data iterates through each element in original list (data) 
# isinstance(item, (list, tuple)) checks if current item is a list or tuple 
# if list or tuple, creates new tuple, iterating through x in original tuple
# and keeps them only if x is not None 
# else, if item is not list or tuple, keeps item as is (ie None in between tuples) 

filtered_data[0]

sum(item is None for item in filtered_data)

# have to split the data on None into groups 

# iterate through filtered_data and organize items into sublists in clean_data list
# use None as delimiter to separate sublists 

clean_data = [] # stores final result 
sim = [] # temporarily hold items for current simulation before adding to clean_data list 

for item in filtered_data: # loop through each item in filtered_data
    if item is None: # check if current item = delimiter
        if len(sim) > 0 : # check if current sim list is not empty 
            # prevents multiple empty lists being added potentially
            clean_data.append(sim) # append non-empty chunk to clean_data list 
            sim = [] # reset to empty list to start collecting items for next group 
    else:
        sim.append(item) # add item to current sim list if item is not None

if len(sim) > 0: # need this bc have data for a simulation but still have remaining data for a 
    # simulation that need to add
    # since transfer to clean_data only happens when hit None
    clean_data.append(sim)

print(f"no. of simulations = {len(clean_data)}")

# converting clean_data to 4D tensor here 

# tensor = fundamental data structure in pytorch for building deep learning models 
# conceptually, a multi-dimensional array, similar to numpy ndarray, but with specialized features, 
# making  it ideal for ML tasks 

# tensor = torch.tensor(clean_data)
# tensor.ndim
# expected sequence of length 30 at dim 1 (got 8) -- shapes differ --> 
# not every array in every chunk has same shape --> can't do torch.tensor(clean_data) if
# shapes differ 

clean_data # clean data is a list of a list containing a tuple of arrays 

# convert arrays in each group to tensor in clean_data

tensor = [
    [torch.tensor(arr) for arr in group]
    for group in clean_data]

# nested list comprehension 

tensor # structure is a list of lists of tensors 

# each inner list contains 2 tensor objects (before and after shot simulation pair) 

tensor[0] # grabbing first simulation data

# len(tensor) = 12 

tensor[0][0].ndim

tensor[0][0].shape # grabbing first simulation pair -- 3D tensor (without None tuples present)

tensor[0][0][0].shape # indexing into 2nd dimension -- shows before state

# tensor[0][0][1] after state 

tensor[0][0][0][0].shape # indexing into 1st dimension -- first row of before state

dataset = [] 

for i in range(len(tensor)):  
    sample_before = tensor[i][0][0]
    
    adj = sample_before[:-1, :-1].float()
    
    x = sample_before[:-1, -1].unsqueeze(1).float() 
    # grain sizes / features?  
    # why does x need to be unsqueezed for link prediction? 
    
    pos_edge_index = (adj != 0).nonzero(as_tuple=False).t().long() 
    # grabbing indices where adjacency matrix is 1 (connected edges) 
    
    mask = pos_edge_index[0] < pos_edge_index[1]
    # mask is where 1st index is less than second 
    
    pos_edge_index = pos_edge_index[:, mask]
    # indexing all rows, only columns where index 1 < index 2 

    # should we do negative edge sampling here? 
    
    data = Data(
        x=x,
        edge_index=pos_edge_index, 
        num_nodes=x.size(0)
    )
    dataset.append(data)

train_frac = 0.7
val_frac = 0.15
test_frac = 0.15 

random.shuffle(dataset)  # shuffle all chunks of data 

# split into training, validation, testing data: 

num_total = len(dataset)
num_train = int(train_frac * num_total)
num_val   = int(val_frac * num_total)
num_test  = num_total - num_train - num_val  # remainder goes to test

train_data = dataset[:num_train] # until sim 8 
val_data   = dataset[num_train:num_train + num_val] #sim 8 until sim 9 
test_data  = dataset[num_train + num_val:] # sim 9 through end simulation (12) 

# each *_data list now contains all data per certain number of simulations 

# load and prepare data for model training 

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=1) 
test_loader  = DataLoader(test_data, batch_size=1)

# batch size = number of samples processed at once during a training step 
# ex: take 8 graphs at time, compute forward pass, loss, backpropagate 

# shuffle = True randomly reorders dataset at start of each epoch 
# ensures model doesn't see data in same order every epoch 

# define GCN + classifier 

# pass node features, edge connectivity to GCN, GCN produces node_emb with shape 
# of [num_nodes, hidden_channels] 
# node embedding = vector representation of a node capturing its features and neighborhood 
# structure in the graph 

# pass node embeddings + edge_label_index to classifier --> get predicted edge scores 
# separate node embedding generation (GCN) from edge scoring (Classifier) - what you need for 
# link prediction setup 

# separating encoder and decoder steps: 

class GCN(torch.nn.Module): #define GCN 
    # torchnn.Module is the base class for all neural networks 
    # init method -- initializes attributes of GCN object 
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(train_data[0].x.shape[1], hidden_channels)
        # initializes first GCN layer 
        # defines input features per node (here, 4) 
        # output is hidden_channels (how many features each node will have in hidden layers; 
        # node embedding size) 
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        # two more GCN layers 
        # stacking layers allows nodes to aggregate info from neighbors multiple hops away 

    # forward method -- refers to forward propagation in AI, which is the process of passing 
    # input data through a neural network's layers to generate an output 
    def forward(self, x: tensor, edge_index: tensor, edge_weight = None):
        # takes in node features, edge_index (adjacency information of shape [2, num_edges])  
        #  weights for edges of shape [num_edges] 
        
        x = F.relu(self.conv1(x, edge_index))
        # self.conv1 does a graph convolution -- each node updates its features by aggregating 
        # its neighbors' features weighted by adjacency 
        # relu activation function is applied element-wise to output of first layer, introducing 
        # non-linearity into model: negative values become 0, positive stay the same 
        # without non-linearity, each layer would be lin comb of neighbors' features and stacking
        # layers wouldn't inc representational power 

        x = F.dropout(x, p=0.1, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        
        x = F.relu(self.conv3(x, edge_index))
        # each layer aggregates neighbor features and updates the node's own embedding ***
        
        return x
        # final output x is node embeddings [num_nodes, hidden_channels], which are used 
        # by classifier to predict edges 
        # (renamed node_emb later for clarity) 


class EdgeClassifier(torch.nn.Module):
    def __init__(self, hidden_dim):
        # hidden_dim = size of internal hidden representation for features per edge used in 
        # edge classifier 
        
        super().__init__()
        # calls the init method of the parent class of edgeclassifier (in this case, 
        # torch.nn.Module) 
        # need so edgeclassifier inherits functionality of a pytorch module 

        self.fc1 = torch.nn.Linear(3 * hidden_dim, hidden_dim)

        # fully connected layer taking a combined edge feature vector of size 4*hidden_dim and 
        # outputs hidden_dim
        # using 4 because edge feature vector below has 4 terms 
        
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        # outputs 1 scalar per edge, which is the predicted score for whether the edge exists
        
    def forward(self, node_emb, edge_index): 
        # node_emb = node embeddings from GCN, shape = [num_nodes, hidden_dim] 
        # edge_index specifies which edges to compute predictions for, shape: [2, num_edges] 
        
        node_1 = node_emb[edge_index[0]] 
        # edge_index[0] lists source nodes of all edges, 
        # node_emb[edge_index[0]] picks the embeddings of all source nodes 
        
        node_2 = node_emb[edge_index[1]]
        # edge_index[1] lists target nodes of all edges
        # node_emb[edge_index[1]] picks the embeddings of all target nodes 
        
        edge_feat = torch.cat([torch.abs(node_1 - node_2), node_1 + node_2, node_1 * node_2], dim=1)
        
        x = F.relu(self.fc1(edge_feat))
        
        return self.fc2(x).squeeze()
        #return (node_1 * node_2).sum(dim=1)  
        
        # fc2 maps hidden_dim --> 1, producing a single scalar per edge 
        # .squeeze() makes output shape [num_edges]

# hard negative sampling - focuses on selecting negatives that look the most similar 
# to positive edges (ie two grains very close together but not adjacent) and trains 
# on these hardest cases 
# cons - can be biased if training on only hard negatives as opposed to random negatives 
# combining easy and hard samples allows for dynamic weighting strategy
def hard_negatives(node_emb, pos_edge_index):
    # defining funcino taking node embeddings and existing pos edge indices 

    num_nodes = node_emb.size(0) # retrieves total number of nodes in graph 
    # from embedding matrix 
    
    device = node_emb.device # makes operations on same device as tensor 
    
    scores = torch.matmul(node_emb, node_emb.t()) # does node matrix multiplication 
    # results in score matrix of shape [num nodes, num nodes] 
    
    mask = torch.zeros((num_nodes, num_nodes), device=node_emb.device)
    # creates matrix of zeroes   
    
    mask[pos_edge_index[0], pos_edge_index[1]] = 1.0 # sets positions in mask to 
    # 1.0 where a positive edge alr exists, marking them as true edges 
    
    mask.fill_diagonal_(1.0) # set diagonal to 1.0 ->  marks self-loops as 
    # positive, ensuring they aren't chosen as negative samples
    
    scores_masked = scores.clone() # create copy of scores matrix 
    
    scores_masked[mask == 1] = -1e9 # replaces similarity scores of true edges 
    # and self-loops with small number, ensuring they aren't selected
    
    hard_neg_indices = torch.argmax(scores_masked, dim=1) # for each node, finds 
    # index of column with hardest negative (score) 
    
    src_nodes = torch.arange(num_nodes, device=device) # create list of node indices 
    # to serve as source node for each hard negative edge 
    
    return torch.stack([src_nodes, hard_neg_indices], dim=0).long()
    # stack source nodes and 
    # fuond hard negative target nodes into a [2, num_nodes] tensor, so has 
    # appropriate pytorch g edge index  

edge_classifier = EdgeClassifier(hidden_dim=200) #creates an edge classifier object, and tells 
# classifier how big the node embeddings are coming from the GCN 

model = GCN(hidden_channels=200) # creates GCN object and sets dimension of node embeddings 
# produced by GCN 
# in model, each node will have 200-dimensional feature vector

# common to have training, validation, and test loops 
# what matters is not mixing training and evaluation in the same iteration

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)#weight_decay = 10e-5)
# initializes adam optimizer (updates the model's weights based on calculated gradients to 
# minimize loss)

params = list(model.parameters()) + list(edge_classifier.parameters()) 

optimizer = torch.optim.AdamW(params, lr = 0.0007, weight_decay = 1e-4) 
# Every time the model updates its weights to get a better score, 
# the optimizer looks at the size of those weights. 
# If they are getting too big or too complex, the optimizer decays them (shrinks them) 
# slightly back toward zero.

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
  optimizer, 
  mode = 'min', 
  factor = 0.5, 
  patience = 5
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cuda for faster parallel computation, otherwise default to cpu 

model = model.to(device) # move model (incl all params) to selected device 
edge_classifier = edge_classifier.to(device) # move edge classifier to selected device 

test_losses = [] 

for epoch in range(1, 60): # epoch = one full pass through training data 
    avg_train_loss = 0 
    for batch in train_loader: 
        model.train() # set model to training mode 
        edge_classifier.train() 

        optimizer.zero_grad() # clear gradients of previous batch (before each new optimization step) 
        # crucial step as by default, gradients are accumulated across iterations 

        # moving components to selected device 
        batch = batch.to(device) 
        batch.edge_index = batch.edge_index.long().to(device)  

        node_emb = model(batch.x, batch.edge_index) # performs forward pass: 
        # model taking node features and connectivity info as input and produces prediction 

# create hard negatives dynamically (as opposed to static in primary data object)
        hard_neg_edge_index = hard_negatives(node_emb, batch.edge_index)
        
        # combine real edges and our new hard negatives
        edge_label_index = torch.cat([batch.edge_index, hard_neg_edge_index], dim=-1).long()
        
        # create labels: 1 for positives, 0 for negatives
        pos_labels = torch.ones(batch.edge_index.size(1), device=device)
        neg_labels = torch.zeros(hard_neg_edge_index.size(1), device=device)
        edge_label = torch.cat([pos_labels, neg_labels], dim=0)

        # calculate predictions, compute loss, compute gradients, use gradients to update params
        
        pred = edge_classifier(node_emb, edge_label_index.long())
        # node_emb is output of GCN (renamed for clarity) 
        # takes embeddings of nodes for each edge in edge_label_index
        # and produces scalar prediction per edge (logits, which are raw scores for edges) 
        
        train_loss = F.binary_cross_entropy_with_logits(pred, edge_label)
        # computes BCE loss directly from logits (raw, unbounded output of a model before 
        # applying sigmoid), compares predicted edge existence vs. true edge label 
        
        train_loss.backward() # computes gradient of train_loss wRT all model params (weights)
        
        optimizer.step() # update parameters using optimizer based on gradients 
        
        avg_train_loss += train_loss.item() / len(train_loader)
        # first term calculates avg loss for current batch 
        # divide by number of batches in training set 
        # get avg loss contribution of that single batch to overall epoch avg 
        # accumulate contribution so avg_train_loss represents avg training loss for entire epoch 
        
        # measures how well model is learning patterns present in training data


    preds_list = []
    labels_list = []
    val_loss = 0.0 
    
    model.eval() 
    edge_classifier.eval() 
    
    for batch in val_loader:
        batch = batch.to(device)
        
        with torch.no_grad(): 
            # generate embeddings
            node_emb = model(batch.x, batch.edge_index)
        
            # mine hard negatives for validation dynamically
            hard_neg_edges = hard_negatives(node_emb, batch.edge_index)
            edge_index = torch.cat([batch.edge_index, hard_neg_edges], dim=-1).long()
        
            # create labels locally (1 for real edges, 0 for mined fakes)
            pos_labels = torch.ones(batch.edge_index.size(1), device=device)
            neg_labels = torch.zeros(hard_neg_edges.size(1), device=device)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

            # predict and Calculate Loss
            pred = edge_classifier(node_emb, edge_index)
            loss = F.binary_cross_entropy_with_logits(pred, labels)
            val_loss += loss.item()

            # collect results for AUC calculation
            preds_list.append(torch.sigmoid(pred).cpu())
            labels_list.append(labels.cpu())

    avg_val_loss = val_loss / len(val_loader)
    test_losses.append(avg_val_loss)

    # combine all batch results into one array
    y_true = torch.cat(labels_list).numpy()
    y_pred = torch.cat(preds_list).numpy()
    
    # calculate AUC
    auc = roc_auc_score(y_true, y_pred) 

    print(f"Epoch: {epoch}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Val AUC: {auc:.4f}, Node emb std: {node_emb.std(dim=0).mean():.4f}") 
    
    # update scheduler based on validation loss
    scheduler.step(avg_val_loss)

print(f"Min Avg Test Loss: {min(test_losses):.4f}")
# train loss higher than test loss -- what want to see with hard neg sampling; means training task 
# (distinguishing real edges from absolute hardest fakes) is uch ore difficult than validation task 
# (distinguishing real edges from rando fakes) 

# increased node_emb (by 100x) -- features more expressive and spread out -- direct sign that 
# weight decay and MLP architecture prevent embeddings from collapsing 