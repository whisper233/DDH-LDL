import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import utils


class LinearBranchNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearBranchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),

            nn.Linear(1024, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class GCNBranchNet(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5, act=nn.Tanh()):
        super(GCNBranchNet, self).__init__()
        self.dropout = dropout
        self.act = act
        self.weight1 = Parameter(torch.FloatTensor(in_dim, out_dim))
        torch.nn.init.xavier_normal_(self.weight1)
    
    def forward(self, x, S):
        x = F.dropout(x, self.dropout, self.training)
        output = S @ x @ self.weight1
        output = self.act(output)
        
        return output


class HashNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HashNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class HashCodeNet(nn.Module):
    def __init__(self, feat_dim, out_dim, hidden_dim=512):
        super(HashCodeNet, self).__init__()
        self.linear_branch = LinearBranchNet(feat_dim, hidden_dim)
        self.gcn_branch = GCNBranchNet(feat_dim, hidden_dim, dropout=0.5)
        self.att_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.att_linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.hash_net = HashNet(hidden_dim, out_dim)

    def forward(self, x, S):
        x1 = self.linear_branch(x)
        x2 = self.gcn_branch(x, S)
        xf = self.att_linear_1(x1) + self.att_linear_2(x2)
        h = self.hash_net(xf)
        return h
    

class HashFuncNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HashFuncNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),

            nn.Linear(1024, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

