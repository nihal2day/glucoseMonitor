import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_size, action_space, output_size, hidden_size):
        super(Critic, self).__init__()
        self.linear_layer_1 = nn.Linear(state_size+action_space, hidden_size)
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.linear_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.linear_layer_3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, states, actions):
        out = self.linear_layer_1(torch.cat((states, actions), dim=1))
        out = self.layer_norm_1(out)
        out = self.relu(out)
        out = self.linear_layer_2(out)
        out = self.layer_norm_2(out)
        out = self.relu(out)
        out = self.linear_layer_3(out)
        return out
