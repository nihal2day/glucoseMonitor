import numpy as np
import torch
from collections import deque, namedtuple

# Reference:
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "new_state", "done"],
)


class ReplayBuffer(object):
    # Reference:
    # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
    def __init__(self, device, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, new_state, done):
        experience = Experience(state, action, reward, new_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[idx] for idx in indices))
        return (
            torch.stack(list(states), dim=0).squeeze(dim=1).to(self.device),
            torch.stack(list(actions), dim=0).squeeze(dim=1).to(self.device),
            torch.stack(list(rewards), dim=0).squeeze(dim=1).to(self.device),
            torch.stack(list(next_states), dim=0).squeeze(dim=1).to(self.device),
            torch.stack(list(dones), dim=0).squeeze(dim=1).to(self.device)
        )
