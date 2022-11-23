import torch
import toch.nn as nn

class Actor(nn.Module):
    def __init__(self, lin1_dim = 400, lin2_dim = 300, out_dim)
        super(Actor,self).__init__()
        self.lin1_dim = lin1_dim
        self.lin2_dim = lin2_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(lin1_dim, lin2_dim)
        self.linear2 = nn.linear(lin2_dim, lin2_dim)
        self.linear3 = nn.linear(lin2_dim, out_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLu

    def forward(self, states):
        lin1 = self.linear1(states)
        lin1_out = self.relu(lin1)
        lin2 = self.linear2(lin1_out)
        lin2_out = self.relu(lin2)
        action = self.linear3(lin2_out)
        action = self.tanh(action)
        return action
################################
#####     TODO #################
#  Define backward (here, or in main?)
################################