import numpy as np
import torch
from collections import deque, namedtuple

# Reference:
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer(object):
    # Reference:
    # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, done, new_state):
        experience = Experience(state, action, reward, done, new_state)
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))
        return (
            torch.stack(list(states), dim=0).squeeze(dim=1),
            torch.stack(list(actions), dim=0).squeeze(dim=1),
            torch.stack(list(rewards), dim=0).squeeze(dim=1),
            torch.stack(list(dones), dim=0).squeeze(dim=1),
            torch.stack(list(next_states), dim=0).squeeze(dim=1)
        )
