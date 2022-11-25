from DDPG.ReplayBuffer import ReplayBuffer
from DDPG.Actor import Actor
from DDPG.Critic import Critic
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DDPG:
    def __init__(self, state_size, action_space, hidden_size, replay_buffer_size, batch_size,
                 lr_actor, lr_critic, gamma, tau, epsilon):

        self.state_size = state_size.shape[0]
        self.action_space = action_space.shape[0]
        self.action_high = action_space.high
        self.action_low = action_space.low
        self.hidden_size = hidden_size
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size

        self.lr_actor = lr_actor        # Learning Rate Actor
        self.lr_critic = lr_critic      # Learning Rate Critic
        self.gamma = gamma              # Future Discounted Reward Rate
        self.tau = tau                  # Target network update rate
        self.epsilon = epsilon          # Exploration amount

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.actor = Actor(self.state_size, self.action_space, self.hidden_size)
        self.actor_target = Actor(self.state_size, self.action_space, self.hidden_size)

        self.critic = Critic(self.state_size + self.action_space, 1, hidden_size)
        self.critic_target = Critic(self.state_size + self.action_space, 1, hidden_size)

        self.update_target(self.actor_target, self.actor, 1.0)  # Hard Update target to match actor
        self.update_target(self.critic_target, self.critic, 1.0)  # Hard Update target to match critic

        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Send to device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    @staticmethod
    def update_target(target, source, tau):
        for target_parameter, source_parameter in zip(*(target.parameters(), source.parameters())):
            target_parameter.data.copy_(target_parameter.data * (1.0 - tau) + source_parameter * tau)

    def act(self, state, with_noise=False):
        state = state.to(self.device)
        self.actor.eval()
        action = self.actor(state)
        if with_noise:
            action = np.random.normal(0, self.action_high * self.epsilon)  # add gaussian noise
        else:
            action = action.detach().numpy()
        action = action.clip(self.action_low, self.action_high)
        self.actor.train()
        return action

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)
        if len(self.replay_buffer) > self.batch_size:
            self.learn()

    def learn(self):
        # Reference: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Critic Loss
        q = self.critic.forward(states, actions).squeeze()
        next_actions = self.actor_target.forward(next_states)
        q_target = self.critic_target.forward(next_states, next_actions).squeeze()
        q_prime = rewards + self.gamma * q_target
        critic_loss = self.critic_criterion(q, q_prime)

        # Actor Loss
        actor_actions = self.actor.forward(states)
        policy_loss = -self.critic.forward(states, actor_actions).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_target(self.actor_target, self.actor, self.tau)    # Soft Update of target actor
        self.update_target(self.critic_target, self.critic, self.tau)  # Soft Update of target critic
