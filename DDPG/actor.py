import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, device, lin1_dim = 400, lin2_dim = 300, out_dim=1, my_init_hyp = 0):
        super(Actor,self).__init__()
        
        self.device = device
        self.my_init_hyp = my_init_hyp
        self.lin1_dim = lin1_dim
        self.lin2_dim = lin2_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(lin1_dim, lin2_dim).to(self.device)
        self.layer_norm_1 = nn.LayerNorm(lin2_dim).to(self.device)
        self.linear2 = nn.Linear(lin2_dim, lin2_dim).to(self.device)
        self.layer_norm_2 = nn.LayerNorm(lin2_dim).to(self.device)
        self.linear3 = nn.Linear(lin2_dim, out_dim).to(self.device)
        if self.my_init_hyp == 1:
            nn.init.kaiming_uniform_(self.linear3.weight, mode='fan_out')
        if self.my_init_hyp == 2:
            self.linear3.weight = nn.Parameter(self.linear3.weight/100)
        if self.my_init_hyp == 3:
            nn.init.kaiming_normal_(self.linear3.weight)
        if self.my_init_hyp == 4:
            nn.init.xavier_uniform_(self.linear3.weight)
        if self.my_init_hyp == 5:
            nn.init.xavier_uniform_(self.linear3.weight, gain=nn.init.calculate_gain('relu'))
        if self.my_init_hyp == 6:
            nn.init.xavier_uniform_(self.linear3.weight, gain=0.01)
        if self.my_init_hyp == 7:
            nn.init.zeros_(self.linear3.weight)
        self.tanh = nn.Tanh().to(self.device)
        self.relu = nn.ReLU().to(self.device)

    def forward(self, states):
        lin1 = self.linear1(states).to(self.device)
        lin1 = self.layer_norm_1(lin1).to(self.device)
        lin1_out = self.relu(lin1).to(self.device)
        lin2 = self.linear2(lin1_out).to(self.device)
        lin2 = self.layer_norm_1(lin2).to(self.device)
        lin2_out = self.relu(lin2).to(self.device)
        action = self.linear3(lin2_out).to(self.device)
        action = self.tanh(action).to(self.device)
        return action
