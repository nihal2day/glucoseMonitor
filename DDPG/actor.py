import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, lin1_dim = 400, lin2_dim = 300, out_dim=1):
        super(Actor,self).__init__()
        self.lin1_dim = lin1_dim
        self.lin2_dim = lin2_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(lin1_dim, lin2_dim)
        self.layer_norm_1 = nn.LayerNorm(lin2_dim)
        self.linear2 = nn.Linear(lin2_dim, lin2_dim)
        self.layer_norm_2 = nn.LayerNorm(lin2_dim)
        self.linear3 = nn.Linear(lin2_dim, out_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, states):
        lin1 = self.linear1(states)
        lin1 = self.layer_norm_1(lin1)
        lin1_out = self.relu(lin1)
        lin2 = self.linear2(lin1_out)
        lin2 = self.layer_norm_1(lin2)
        lin2_out = self.relu(lin2)
        action = self.linear3(lin2_out)
        action = self.tanh(action)
        return action
