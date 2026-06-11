import torch
import torch.nn as nn 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv 
from torch_geometric.loader import DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
import random
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

# within each 2D array, last col = last row 

# have 268 tuples, where middle element is None -- filtering out the None placeholders

filtered_data = [
    tuple(x for x in item if x is not None) if isinstance(item, (list, tuple)) else item
    for item in data
]

# for item in data iterates through each element in original list (data) 

# isinstance(item, (list, tuple)) checks if current item is a list or tuple 
# if list or tuple, creates new tuple, iterating through x in original tuple and keeps them only if x is not None 
# else, if item is not list or tuple, keeps item as is (ie None in between tuples) 


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
    # transfer to clean_data only happens when hit None
    clean_data.append(sim)

print(f"no. of simulations = {len(clean_data)}")

clean_data # clean data is a list of a list containing a tuple of arrays 


# convert arrays in each group to tensor in clean_data

tensor = [
    [torch.tensor(arr) for arr in group]
    for group in clean_data]

# nested list comprehension 

dataset = [] # build list of samples 
for i in range(len(tensor)): #for i in range but don't use i; creating same datapoint over and over again 

    sample_before = tensor[i][0][0] # after selecting sample i, selects first element of dimension 1 
    sample_after  = tensor[i][1][0] # selects second element of dimension 1 because filtered out nones in clean data 

    adj = sample_before[:-1, :-1].float()
    # selecting subset of sample_before matrix to act as graph adjacency matrix 
    # rows/columns of adj corr to nodes; nonzero entries indicate edges/connectivity 
    # size is [num_nodes, num_nodes] because saying for every pair of nodes (i,j) are they connected 
    
    x = sample_before[:-1, -1].unsqueeze(1).float() # defines node features, taking all rows except 
    # last; unsqueeze converts [100] --> [100,1] 
#    print(x.shape)
    # slice is picking a subset of the matrix (all rows except last, last column) 

    y = sample_after[:-1, -1].unsqueeze(1).float() 
 #   print(y.shape); unsqueeze converts [100] --> [100,1] 
    # defines target node features post-shot 

    edge_index = (adj != 0).nonzero().t() # .nonzero() finds the coordinates of non-zero elements
    # converts adjacency into list of edges [2, num_edges] for PyTorch Geometric 

    edge_weight = adj[adj != 0] 
    # creating boolean mask and only returning where mask is true 
    # if extracting only where edges are non-zero, extracting weights only where nodes 
    # influence each other 

    data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_weight,
    y=y
    )
    
    dataset.append(data) 

train_frac = 0.7
val_frac = 0.15
test_frac = 0.15 

random.shuffle(dataset)  # shuffle all chunks of data 

# split into training, validation, testing data: 

num_total = len(dataset)
num_train = int(train_frac * num_total)
num_val = int(val_frac * num_total)
num_test  = num_total - num_train - num_val   

train_data = dataset[:num_train] # until sim 8 
val_data = dataset[num_train:num_train + num_val] #sim 8 until sim 9 
test_data  = dataset[num_train + num_val:] # sim 9 through end simulation (12) 

# each *_data list now contains all data per certain number of simulations 

# load and prepare data for model training 
# dec batch_sizes - experiment with 

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=4) 
test_loader  = DataLoader(test_data, batch_size=4)

# creating model 

# 2-layer GCN for node prediction 
       # layers = how many graph convolution layers you stack; represents depth of message passing 
        # 2 layers means each node can aggregate info from 2 hops away in graph 
# hidden channels - represent dimensionality (width) of node embeddings (feature vectors) within layers 

# troubleshooting minimizing loss -- 2x better loss w sageconv as opposed to gcn 

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels): 
        # init method -- constructor method that initializes the model's layers and parameters 
        super().__init__() # calls constructure of torch.nn.Module to properly initialize model
        
        # x.shape, y.shape are tuples -- 100 nodes, 1 feature/target (grain size before/after)
        # need integers for input/ouput dimensions, not tensors 
        self.conv1 = SAGEConv(dataset[0].x.shape[1], hidden_channels) 
        # initializes first sageconv layer (sageconv - conv layer aggregating features from node's 
        # local neighborhood) 
        # defines input features per node (1 feature) 
        # output is hidden_channels (which defines the output features per node) for hidden layer 

        self.bn1 = nn.LayerNorm(hidden_channels) 
        # pytorch module used to apply layer normalization to node features produced by sageconv layer, 
        # independently for each node 
        # primarly designed to stabilize and accelerate learning  
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels) # initializing second layer
        
        self.bn2 = nn.LayerNorm(hidden_channels) 
        # pytorch module used to apply layer normalization to node features produced by sageconv layer, 
        # independently for each node 
        # primarly designed to stabilize and accelerate learning  
        
        self.conv3 = SAGEConv(hidden_channels, dataset[0].y.shape[1]) 
        # initializes third layer
        # receives hidden representation from conv2 
        # output projects hidden state to final output dimension (final number of features produced
        # by last layer) 
    
    # forward method -- refers to forward propagation in AI, which is the process of passing input data 
    # through a neural network's layers to generate an output 
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) # input node features are passed through first graph convolutional layer 

        x = self.bn1(x)
        # 2D batch normalization layer -- normalizes the output of first convolutional layer 
        # helps stabilize and speed up training process 
        # ensures values entering activation function are in well-behaved range; also helps keep 
        # gradients at manageable scale during backpropagation 
        
        x = F.leaky_relu(x) 
        # leaky relu is a type of activation function for neural networks that allows a small, 
        # non-zero gradient for negative inputs (as opposed to 0, as seen with relu) 
        # instead of mapping negative inputs to zero, keeps values slightly active, allowing neurons to 
        # remain active and learn during training 

        # "dying neuron" issue -- when using reLU, if a neuron's weights update st it always receives 
        # negative inputs, its output becomes 0 and gradient becomes 0 --> neuron dies and stops learning
        
        x = F.dropout(x, p=0.2, training=self.training) # dropout is applied for regularization to prevent 
        # overfitting. randomly sets half of input features to zero during training, as controlled by  
        # self.training 
        
        # overfitting = low train loss but fails to generalize on unseen test/validation data 
        # dropout rate (eg p = 0.5) determines the probability of skipping a node 
        # lower dropout rate (eg 0.1-0.2) provides milder regularization 
        # higher dropout rate (eg 0.5-0.8) aggressively reduces overfitting 
        
        x = self.conv2(x, edge_index)
        # modified features are passed through the second graph convolutional layer 
        
        x = self.bn2(x) # batch normalization 
        
        x = F.leaky_relu(x) # activation function that allows a small, non-zero gradient for neg inputs 

        x = F.dropout(x, p=0.5, training=self.training) # to reduce overfitting

        x = self.conv3(x, edge_index)
        # modified features are passed through the third graph convolutional layer 

        x = F.leaky_relu(x) # activation function that allows a small, non-zero gradient for neg inputs

        return x
        # output features from third layer are returned as final output of network 

model = GCN(hidden_channels = 100)   
# the more hidden channels --> each node can carry richer info when aggregating neighbors 

# ex: here, means that after the input features are processed, each node in the graph is represented by a 
# vector of 200 numerical values 
# a higher number allows the model to learn more complex features but increases the risk of overfitting, while 
# a lower number is used for smaller/simpler models 

print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cuda for faster parallel computation, otherwise default to cpu 

model = model.to(device)
# move model (incl all params) to selected device 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 10e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
   optimizer, # optimizer whose LR to adjust
   mode = 'min', # scheduler waits for monitored metric to stop decreasing
   factor = 0.5, # the factor by which the LR will be reduced. when a plateau 
    # (typically in validation loss) is hit, 
    # the new LR = old LR * factor (here, LR cut in half) 
   patience = 3  # number of epochs to wait for improvement before reducing the LR 
)
# learning rate scheduler, which reduces the learning rate when a specific metric 
# (typically validation loss) has stopped improving 
# adaptive scheduler -- meaning it responds to the training progress rather than reducing the
# learning rate at fixed intervals 

criterion = torch.nn.L1Loss() # defines loss function 
# huber loss function is a popular loss function used in regression tasks, designed to be 
# robust to outliers combining the best properties of 2 common loss functions: MSE and MAE 

# useful when dataset contains noisy data or outliers, helping models learn more reliably and 
# avoid being skewed by extreme values 

# default delta is 1.0; delta param is threshold determining the transition between 
# quadratic and linear loss behaviors; by controlling delta (ie making it lower), makes model 
# more sensitive to outliers 

# experiment with delta = 0.1 or 0.5 --> makes the loss more robust to "outlier nodes" that 
# might be causing ~0.07 high-loss runs.


test_losses = [] 

model.train() # sets model to training mode 

# training
for epoch in range(1, 100): # epoch = one full pass through training data 
    
    avg_train_loss = 0
    for graph in train_loader: 
        
        optimizer.zero_grad() # clear gradients of previous batch (before each new optimization step) 
        # crucial step as by default, gradients are accumulated across iterations 
        
        graph = graph.to(device) # moves entire graph object and underlying tensors from current memory 
        # location (usually CPU RAM) to specified device (eg cuda: 0: for first GPU) 
        
        graph.edge_index = graph.edge_index.long().to(device)  # casts the edge_index tensor to a 64-bit integer type 
        # also moves this tensor to the target device, ensuring it resides in the same memory as the rest of 
        # the model and data 
        
        graph.y = graph.y.float().to(device) # casts labels to a 32-bit floating-point data type 
        # sometimes just a general standard in GNNs for the output layer's loss calculation 
        # moves labels to the specified device 
        #*** both the model's output and target labels must be on the same device for loss calculation***

        pred = model(graph.x, graph.edge_index) # performs a forward pass: model taking node features and 
        # connectivity info as input and produces prediction  
        
        ground_truth = graph.y # these are target node values (ground truth labels) 
        
        train_loss = criterion(pred, ground_truth) # computing the 
        # training loss (error) between predictions and true labels (ground_truth) for training nodes 
        
        train_loss.backward() # computes gradient of train_loss wRT all model params (weights)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # GNN gradients can become very large, leading to unstable weight updates (instead of weights 
        # descending smoothly), this step solves this by scaling down gradients if total norm exceeds 
        # max_norm (calculates L2-norm), keeping training stable 
        
        optimizer.step() # update parameters (weights) using optimizer based on gradients 
        
        avg_train_loss += train_loss.item() / len(train_loader) 
        # first term calculates avg loss for current batch 
        # divide by number of batches in training set 
        # get avg loss contribution of that single batch to overall epoch avg 
        # accumulate contribution so avg_train_loss represents avg training loss for entire epoch 
        
        # measures how well model is learning patterns present in training data

# perform evaluation/testing of GNN
    test_loss = 0 # initializes a variable to accumulate loss over all graphs in dataset 

    model.eval() # sets the model to evaluation mode 
    
    with torch.no_grad():
        for graph in val_loader: 
         # disables gradient calculation for all operations within block 
        # during evaluation, don't need to compute gradients for propagation, so this speeds up computation

            graph = graph.to(device)
            
            out = model(graph.x, graph.edge_index) # performs forward pass of model, taking node 
            # features and graph connectivity info as input and produces out (predicted node labels) 
            
            test_loss += criterion(out, graph.y).item()
            # accumulates loss for entire current batch in test set 
            
    avg_test_loss = test_loss / len(test_loader) # calculates avg loss per batch over all test batches   
    # divide by len(test_loader) yields mean test loss per batch 
    # measures model's ability to generalize to unseen data 

    test_losses.append(avg_test_loss)
    
    
    print(f"Epoch: {epoch:}, Avg Train Loss: {avg_train_loss:.4f}, Avg Test Loss: {avg_test_loss:.4f}")


    scheduler.step(avg_test_loss)
    
print(f"Min Avg Test Loss: {min(test_losses):.4f}")

    # train and test losses are decreasing across epochs -- model is learning 
    # no sign of overfitting = low train loss but high test loss