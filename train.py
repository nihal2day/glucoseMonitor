import os

import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import gym
from gym.envs.registration import register
import numpy as np
from DDPG.DDPG import DDPG
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# def custom_reward(BG_last_hour):
#     if BG_last_hour[-1] > 250:
#         return -30
#     if BG_last_hour[-1] > 200:
#         return -10
#     if BG_last_hour[-1] > 180:
#         return -3
#     if BG_last_hour[-1] < 75:
#         return -30
#     if BG_last_hour[-1] < 90:
#         return -10
#     else:
#         return 10
#
#
# register(
#     id='simglucose-adolescent2-v0',
#     entry_point='simglucose.envs:T1DSimEnv',
#     kwargs={'patient_name': 'adolescent#002',
#             'reward_fun': custom_reward})
#
# env = gym.make('simglucose-adolescent2-v0')

writer = SummaryWriter('runs/run_1')

env = gym.make('Pendulum-v1')

state_size = env.observation_space
action_space = env.action_space
hidden_size = 64
replay_buffer_size = 1000000
batch_size = 256
lr_actor = 1e-4
lr_critic = 5e-3
gamma = 0.99
tau = 0.2
number_of_episodes = 5000
epsilon = 1.0
epsilon_minimum = 0.01
epsilon_decay = ((epsilon-epsilon_minimum) / number_of_episodes)

agent = DDPG(state_size, action_space, hidden_size, replay_buffer_size, batch_size,
             lr_actor, lr_critic, gamma, tau, epsilon)

rewards = []
average_rewards = []

for episode in range(number_of_episodes):
    state = env.reset()
    episode_reward = 0
    agent.epsilon = agent.epsilon - epsilon_decay
    if agent.epsilon < epsilon_minimum:
        agent.epsilon = epsilon_minimum
    done = False
    while not done:
        action = agent.act(torch.Tensor(state), with_noise=True)
        next_state, reward, done, _ = env.step(action)
        # print(f"NextState: {next_state}, reward: {reward}, done: {done}")
        agent.step(
            torch.FloatTensor(np.array([state])),
            torch.FloatTensor(np.array([action])),
            torch.FloatTensor(np.array([reward])),
            torch.FloatTensor(np.array([next_state])),
            torch.LongTensor(np.array([done])))
        state = next_state
        episode_reward += reward

        if done:
            sys.stdout.write(f"Episode: {episode}  Reward: {episode_reward}  Epsilon: {agent.epsilon} \r\n")

    writer.add_scalar('episode/reward', episode_reward, episode)
    rewards.append(episode_reward)
    average_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(average_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# Test
test_rewards = []
test_average_rewards = []
for episode in range(10):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        env.render()
        action = agent.act(torch.Tensor(state))
        next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward

        if done:
            sys.stdout.write(f"Episode: {episode}  Reward: {episode_reward}  Epsilon: {agent.epsilon} \r\n")

    test_rewards.append(episode_reward)
    test_average_rewards.append(np.mean(rewards[-10:]))

plt.plot(test_rewards)
plt.plot(test_average_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


