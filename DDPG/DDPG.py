from ReplayBuffer import ReplayBuffer
from Actor import Actor
from Critic import Critic
import torch.optim as optim


class DDPG:
    def __init__(self, state_size, action_space, hidden_size, replay_buffer_size, alpha, gamma, tau, device):

        self.state_size = state_size
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.replay_buffer_size = replay_buffer_size

        self.alpha = alpha  # Learning Rate
        self.gamma = gamma  # Future Discounted Reward Rate
        self.tau = tau      # Target network update rate

        self.device = device

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.actor = Actor()
        self.actor_target = Actor()

        self.critic = Critic()
        self.critic_target = Critic()

        self.update_target(self.actor_target, self.actor, 1.0)  # Hard Update target to match actor
        self.update_target(self.critic_target, self.critic, 1.0)  # Hard Update target to match critic

        self.actor_optimizer = optim.Adam(lr=self.alpha)
        self.critic_optimizer = optim.Adam(lr=self.alpha)

    @staticmethod
    def update_target(target, source, tau):
        for target_parameter, source_parameter in zip((target.parameters(), source.parameters())):
            target_parameter.data.copy_(target_parameter.data * (1.0 - tau) + source_parameter * tau)

    def act(self, state, exploration_noise=None):
        x = state.to(self.device)
        self.actor.eval()
        action = self.actor(x)
        self.actor.train()
        action = action.data

        epsilon exploration_noise

        clip to actionlow/high

        pass

    def step(self, state, action, reward, next_state, done):
        pass

    def learn(self):
        pass