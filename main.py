#import libs
import loader
from tree import SoftDecisionTree

import torch
import torch.optim as optim
import torch.nn.functional as F


#create empty container
class A(object):
    pass

args = A()

#data configurations
args.input_dim = 28*28
args.output_dim = 10

#tree configurations
args.max_depth = 4

#NN configurations
args.batch_size = 100
args.test_batch_size = 1280
args.epochs = 2
args.lr = 0.1
args.momentum = 0.5
args.no_cuda = False
args.weight_decay = 0.1

device = torch.device('cuda:0' if (torch.cuda.is_available() and args.no_cuda==False) else 'cpu')

#misc. configurations:
args.log_interval = 10
args.seed = 197


#load data
path = './data'
train_loader = loader.mnist_loader(path = path, nClasses=args.output_dim, batch_size=args.batch_size, train=True)
test_loader = loader.mnist_loader(path = path, nClasses=args.output_dim, batch_size=args.test_batch_size, train=False)
    #we need the capability of loading batches for the NN and loading it all at once for pi (check this part)

####run model

network = SoftDecisionTree(args).to(device)
optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

#train & test
for epoch in range(1, args.epochs + 1):
    network.train() #also transfer data
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        y_pred = network(x,y)
        loss = F.nll_loss(y_pred, y)
        if batch_idx % network.args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx }/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.mean().item()}')
        loss.backward()
        optimizer.zero_grad()
    test()
#test


#analyze wavelets
print("hi!")