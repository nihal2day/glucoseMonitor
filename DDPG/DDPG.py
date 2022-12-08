import gc
import os
from DDPG.ReplayBuffer import ReplayBuffer
from DDPG.actor import Actor
from DDPG.Critic import Critic
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from DDPG.noise import OUNoise


class DDPG:
    def __init__(self, state_size, action_space, actor_hidden_size, critic_hidden_size, replay_buffer_size, batch_size,
                 lr_actor, lr_critic, gamma, tau, sigma, theta, dt, gpu='colab'):

        self.state_size = state_size.shape[0]
        self.action_space = action_space.shape[0]
        self.action_high = action_space.high
        self.action_low = action_space.low
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size

        self.lr_actor = lr_actor        # Learning Rate Actor
        self.lr_critic = lr_critic      # Learning Rate Critic
        self.gamma = gamma              # Future Discounted Reward amount
        self.tau = tau                  # Target network update rate
        self.sigma = sigma              # OUNoise sigma
        self.theta = theta              # OUNoise theta
        self.dt = dt                    # OUNoise dt
        self.gpu = gpu
        
        if self.gpu=="colab":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.gpu=='mps':
            self.device = torch.device("mps" if torch.has_mps else "cpu")
        
        
        print("My device is: " + str(self.device))

        self.replay_buffer = ReplayBuffer(self.device,self.replay_buffer_size)

        self.actor = Actor(self.device,self.state_size, self.actor_hidden_size, self.action_space)
        self.actor_target = Actor(self.device,self.state_size, self.actor_hidden_size, self.action_space)
        action_dim = action_space.shape[0]
        self.actor_noise = OUNoise(mu=np.zeros(action_dim), sigma=self.sigma, theta=self.theta, dt=self.dt)

        self.critic = Critic(self.device,self.state_size, self.action_space, 1, critic_hidden_size)
        self.critic_target = Critic(self.device,self.state_size, self.action_space, 1, critic_hidden_size)

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
        action = self.actor(state).cpu().detach().numpy()
        if with_noise:
            action = action + self.actor_noise()
        else:
            action = action
        self.actor.train()
        return action

    def clip_action(self, unnormalized_action):
        return unnormalized_action.clip(self.action_low, self.action_high)

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)
        if len(self.replay_buffer) > self.batch_size:
            self.learn()

    def reset(self):
        self.actor_noise.reset()

    def learn(self):
        # Reference: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        self.actor.train()
        self.critic.train()

        next_actions = self.actor_target.forward(next_states)
        q_target = self.critic_target.forward(next_states, next_actions).squeeze()
        q_prime = rewards + self.gamma * (1.0-dones) * q_target

        # Update Critic
        self.critic_optimizer.zero_grad()
        q = self.critic.forward(states, actions).squeeze()
        critic_loss = self.critic_criterion(q, q_prime)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic.forward(states, self.actor.forward(states))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.update_target(self.actor_target, self.actor, self.tau)    # Soft Update of target actor
        self.update_target(self.critic_target, self.critic, self.tau)  # Soft Update of target critic

    def save_checkpoint(self, last_timestep, path):
        # Reference: https://github.com/schneimo/ddpg-pytorch/blob/master/ddpg.py
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        # Reference: https://github.com/schneimo/ddpg-pytorch/blob/master/ddpg.py
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.device)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.replay_buffer = checkpoint['replay_buffer']

            return True
        return False
