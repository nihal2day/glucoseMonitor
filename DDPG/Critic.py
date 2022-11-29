import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, device, state_size, action_space, output_size, hidden_size):
        super(Critic, self).__init__()
        
        #if running on colab
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #if running on m1 mac
        self.device = torch.device("mps" if torch.has_mps else "cpu")
        
        self.linear_layer_1 = nn.Linear(state_size+action_space, hidden_size).to(self.device)
        self.layer_norm_1 = nn.LayerNorm(hidden_size).to(self.device)
        self.linear_layer_2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.layer_norm_2 = nn.LayerNorm(hidden_size).to(self.device)
        self.linear_layer_3 = nn.Linear(hidden_size, output_size).to(self.device)
        self.relu = nn.ReLU().to(self.device)

    def forward(self, states, actions):
        inputs = torch.cat((states, actions), dim=1).to(self.device)
        out = self.linear_layer_1(inputs).to(self.device)
        out = self.layer_norm_1(out).to(self.device)
        out = self.relu(out).to(self.device)
        out = self.linear_layer_2(out).to(self.device)
        out = self.layer_norm_2(out).to(self.device)
        out = self.relu(out).to(self.device)
        out = self.linear_layer_3(out).to(self.device)
        return out
