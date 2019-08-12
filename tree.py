import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SoftDecisionTree(torch.nn.Module):

    def __init__(self, args):
        super(SoftDecisionTree, self).__init__()
        
        #figure out args
        #in the future: calculate a way to do convolution sizes depending on features

        self.nodes = []
        self.leaves = []

        self.args = args
        
        self.nLeaves = 2**args.max_depth
        
        self.nNodes = self.node_calc()
              
        D_out = 1
        
        hidden_features = 10
        
        
        self.mu = (1/self.nLeaves)*torch.ones([self.args.batch_size,self.nLeaves], requires_grad=True).to(device)
        self.pi = (1/self.args.output_dim)*torch.ones([self.nLeaves,self.args.output_dim], requires_grad=True).to(device)
        self.cal_P()

        #pre-tree NN
        self.conv1 = nn.Conv2d(1, 20, 5, 1).to(device)
        self.bn1 = nn.BatchNorm2d(20).to(device)
        
        self.conv2 = nn.Conv2d(20, 50, 5, 1).to(device)
        self.bn2 = nn.BatchNorm2d(50).to(device)
        self.fc1 = nn.Linear(4*4*50, hidden_features).to(device)
        self.bn3 = nn.BatchNorm1d(hidden_features).to(device)
        
        #tree NN
        self.theta = torch.nn.ModuleList([torch.nn.Linear(hidden_features, D_out) for i in range(self.nNodes)]).to(device)
        self.bnt = torch.nn.ModuleList([torch.nn.BatchNorm1d(D_out) for i in range(self.nNodes)]).to(device)

        self.root = Node(1, self.args,self)

    def forward(self, x,y):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # if phase is 'train':
        #     self.dTree.mu = self.dTree.mu_train
        #     self.dTree.pi = self.dTree.iter_pi(self.dTree.P,self.dTree.pi,self.dTree.mu)
        # elif phase is 'val':
        #     self.dTree.mu = self.dTree.mu_val
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.bn3(self.fc1(x)))
        
        # return x
        self.mu = (1/self.nLeaves)*torch.ones([x.size(0),self.nLeaves], requires_grad=True).to(device)
        self.path_prob_init = torch.ones(x.size(0), 1, requires_grad=True).to(device)
        self.root.cal_prob(x, self.path_prob_init) ## GG figure out how to use nodes during forward phase

        # calculate new probability
        self.cal_pi(y)
        self.cal_P()

        P_log = torch.log(self.P)
        return P_log

    def node_calc(self):
        nNodes = 0
        for i in range(self.args.max_depth): nNodes += 2**i
        return nNodes
    
    def cal_pi(self, y):
       
        #we have a normalization problem with values that are not updated
        pi_holder = torch.zeros(len(self.leaves),self.args.output_dim)
        
        for l in range(len(self.leaves)):
            for i in range(len(y)):
                    y_i = int(y[i])
                    pi_holder[l,y_i] = pi_holder[l,y_i]+(self.pi[l,y_i]*self.mu[i,l]/self.P[i,y_i])
            z = torch.sum(pi_holder[l,:])
            self.pi[l,:] = pi_holder[l,:]/z

    def cal_P(self):
            self.P = torch.matmul(self.mu,self.pi)
            self.P = torch.autograd.Variable(self.P, requires_grad = True)

class Node():

    def __init__(self, depth, args, tree):
        tree.nodes.append(self)
        self.node_n = len(tree.nodes)-1
        # print(f"node number {len(tree.nodes)}")
        self.tree = tree
        self.args = args
        self.fc = tree.theta[self.node_n]

        self.leaf = False
        self.prob = None

        self.build_child(depth)

    def build_child(self, depth):
        if depth < self.args.max_depth:
            self.left = Node(depth+1, self.args,self.tree)
            self.right = Node(depth+1, self.args,self.tree)
        else :
            self.left = Leaf(self.args,self.tree)
            self.right = Leaf(self.args,self.tree)

    def cal_prob(self, x, path_prob):
        self.prob = torch.sigmoid(self.fc(x)) #probability of selecting right node
        # self.path_prob = path_prob #path_prob is the probability route
        self.left.cal_prob(x, path_prob * (1-self.prob))
        self.right.cal_prob(x, path_prob * self.prob)

class Leaf():
    def __init__(self, args,tree):
        tree.leaves.append(self)
        self.leaf_n = len(tree.leaves)-1
        self.args = args
        self.leaf = True
        self.tree = tree

    def cal_prob(self, x, path_prob):
        
        self.tree.mu[:,self.leaf_n] = path_prob.squeeze()
        