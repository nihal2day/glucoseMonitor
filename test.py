import os
import sys
import warnings
import numpy as np
import gym
from gym.envs.registration import register
import torch
from scipy.stats import variation
from DDPG.DDPG import DDPG
from Normalized_Actions import NormalizedActions


os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def custom_reward(bg_last_hour, slope=None):
    bg = bg_last_hour[-1]
    if bg >= 180:
        x = [180, 350]
        y = [0, -2]
        return np.interp(bg, x, y)
    if bg <= 70.729:
        return -0.025 * (bg - 95) ** 2 + 15
    else:
        return -0.005 * (bg - 125) ** 2 + 15

register(
     id='simglucose-adolescent2-v0',
     entry_point='simglucose.envs:T1DSimEnv',
     kwargs={'patient_name': 'adolescent#002',
             'reward_fun': custom_reward})

env = gym.make('simglucose-adolescent2-v0')
env = NormalizedActions(env)

state_size = env.observation_space
action_space = env.action_space
hidden_size = 32
actor_hidden_size = hidden_size
critic_hidden_size = hidden_size
replay_buffer_size = 100000
batch_size = 256
lr_actor = 1e-4
lr_critic = 1e-4
gamma = 0.99                           # DDPG - Future Discounted Rewards amount
tau = 0.001                             # DDPG - Target network update rate
sigma = 1.5                             # OUNoise sigma - used for exploration
theta = 0.5                             # OUNoise theta - used for exploration
dt = 1e-2                               # OUNoise dt - used for exploration
alpha = 1.0                             # Replay Buffer alpha
beta = 1.0                              # Replay Buffer beta


agent = DDPG(state_size, action_space, actor_hidden_size, critic_hidden_size, replay_buffer_size, batch_size,
             lr_actor, lr_critic, gamma, tau, sigma, theta, dt)

# Load Checkpoint if set
load_checkpoint = True
if load_checkpoint:
    checkpoint = f"./Checkpoints/CheckpointFinal-12-12-2022_0853.gm"
    print(f"Loading Checkpoint: {checkpoint}")
    agent.load_checkpoint(checkpoint)

# Test
test_rewards = []
test_cv = []
test_time_in_range = []
episode_lengths = []
test_episodes = 10
for episode in range(test_episodes):
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    while not done:
        env.render('human')
        action = agent.act(torch.Tensor(state))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        episode_reward += reward
        episode_length += 1
        if done:
            sys.stdout.write(f"Episode: {episode} Reward: {episode_reward} \r\n")
    bgh = env.show_history()
    bgh['in_range'] = 1
    bgh.loc[bgh['BG'] < 80, 'in_range'] = 0
    bgh.loc[bgh['BG'] > 180, 'in_range'] = 0
    cv = variation(bgh['BG'], ddof=1)
    time_in_range = bgh['in_range'].mean()
    test_cv.append(cv)
    test_time_in_range.append(time_in_range)
    test_rewards.append(episode_reward)
    episode_lengths.append(episode_length)

mean_cv = sum(test_cv)/len(test_cv)
mean_time_in_range = sum(test_time_in_range)/len(test_time_in_range)
mean_rewards = sum(test_rewards)/len(test_rewards)
mean_episode_length = sum(episode_lengths)/len(episode_lengths)

sys.stdout.write(f"Mean CV: {mean_cv} \n Mean Time in Range: {mean_time_in_range} \n Mean Rewards: {mean_rewards} \n Mean Episode Length: {mean_episode_length} \r\n")