import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Critic, self).__init__()
        self.linear_layer_1 = nn.Linear(input_size, hidden_size)
        self.linear_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_layer_3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, states, actions):
        inputs = torch.cat((states, actions), dim=1)
        out = self.linear_layer_1(inputs)
        out = self.relu(out)
        out = self.linear_layer_2(out)
        out = self.relu(out)
        out = self.linear_layer_3(out)
        out = self.relu(out)

        return out