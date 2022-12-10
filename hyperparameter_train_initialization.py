#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:58:09 2022

@author: brentdooley, nihalsatyadev, jakefemmenino, kevinchang
"""

import os
import sys
import csv
import warnings
from datetime import datetime
import numpy as np
import gym
from gym.envs.registration import register
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import variation
import pandas as pd
from DDPG.DDPG import DDPG
from Normalized_Actions import NormalizedActions
import matplotlib.pyplot as plt
import time


os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore", category=DeprecationWarning)


#choose between 1 and 2 for your parallel runs
runType = 1

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
#env = gym.make('Pendulum-v1')
env = NormalizedActions(env)

writer = SummaryWriter()

state_size = env.observation_space
action_space = env.action_space


hidden_size = 32
learning_rate = 1e-4

replay_buffer_size = 10000
batch_size = 256


## BRENT REPLACE THESE VALUES (GAMMA, TAU, THETA) WITH THE FINAL VALUES
gamma = 0.999999                           # DDPG - Future Discounted Rewards amount
tau = 0.0000001                             # DDPG - Target network update rate
sigma = 3                                     # OUNoise sigma - used for exploration
theta = 0.000001                             # OUNoise theta - used for exploration

run_adder = 2000
 
dt = 1e-2                               # OUNoise dt - used for exploration
number_of_episodes = 10              # Total number of episodes to train for
validation_rate = 25                    # Run validation every n episodes

my_init_hyp = 0

csv_filename = 'initialization_search.csv'
csv_fields = ['initalization_type', 'date_time',
              'mean_time_in_range', 'mean_coefficient_of_variance', 'mean_reward','mean_episode_length']

if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_fields = ['initialization_type', 'date_time',
                      'mean_time_in_range', 'mean_coefficient_of_variance', 'mean_reward','mean_episode_length']
        csv_writer.writerow(csv_fields)



myRange = None

if runType == 1:
    myRange = range(0,5)
if runType == 2:
    myRange = range(5,8)

for i in myRange:

    my_init_hyp = i
    
    initialization_types = ['kaiming_uniform', 'kaiming_uniform_with_fan_out', 'kaiming_uniform_div_100', 'kaiming_normal', 'xavier_uniform', 'xavier_uniform_with_gain_relu', 'xavier_uniform_with_gain_0.01', 'zeros']
    
    print("My initalization is:")
    print(initialization_types[i])

    actor_hidden_size = hidden_size
    critic_hidden_size = hidden_size
    
    lr_actor = learning_rate
    lr_critic = learning_rate

    agent = DDPG(state_size, action_space, actor_hidden_size, critic_hidden_size, replay_buffer_size, batch_size,
                 lr_actor, lr_critic, gamma, tau, sigma, theta, dt, my_init_hyp)

    actor_losses_per_episode = np.zeros(number_of_episodes)
    critic_losses_per_episode = np.zeros(number_of_episodes)
    start_t = time.time()
    for episode in range(number_of_episodes):
        state = env.reset()
        agent.reset()
        episode_reward = 0
        min_action = np.inf
        max_action = -np.inf
        episode_length = 0
        done = False
        while not done:
            episode_length += 1
            normalized_action = agent.act(torch.Tensor(state), with_noise=True)
            unnormalized_action = env.action(normalized_action)
            clipped_action = agent.clip_action(unnormalized_action)
            if clipped_action > max_action:
                max_action = clipped_action
            if clipped_action < min_action:
                min_action = clipped_action
            action = env.reverse_action(clipped_action.copy())
            next_state, reward, done, _ = env.step(action)
            agent.step(
                torch.FloatTensor(np.array([state])),
                torch.FloatTensor(np.array([action])),
                torch.FloatTensor(np.array([reward])),
                torch.FloatTensor(np.array([next_state])),
                torch.LongTensor(np.array([done])))
            state = next_state
            episode_reward += reward
    
            if done:
                sys.stdout.write(f"Episode: {episode} Length: {episode_length} Reward: {episode_reward} MinAction: {min_action} MaxAction: {max_action} \r\n")
                critic_losses, actor_losses = agent.get_losses()
                critic_losses_per_episode[episode] = np.mean(critic_losses)
                actor_losses_per_episode[episode] = np.mean(actor_losses)

        writer.add_scalar('Train episode/reward', episode_reward, episode)
        writer.add_scalar('Train episode/length', episode_length, episode)
    
        if episode % validation_rate == 0 and episode != 0:
            for val_episode in range(3):
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                min_action = np.inf
                max_action = -np.inf
                done = False
                while not done:
                    action = agent.act(torch.Tensor(state))
                    unnormalized_action = env.action(action)
                    if unnormalized_action > max_action:
                        max_action = unnormalized_action
                    if unnormalized_action < min_action:
                        min_action = unnormalized_action
                    action = env.reverse_action(unnormalized_action.copy())
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    if done:
                        sys.stdout.write(f"Validation Episode: {val_episode} Reward: {episode_reward} MinAction: {min_action} MaxAction: {max_action} \r\n")
                        writer.add_scalar('Validation episode/reward', episode_reward, episode)
                        writer.add_scalar('Validation episode/length', episode_length, episode)

    timestamp = datetime.timestamp(datetime.now())
    timestamp_str = datetime.now().strftime('%m-%d-%Y_%H%M')
    fig, (ax1, ax2) = plt.subplots(2)
    fig.tight_layout(pad=3)
    ax1.plot(range(number_of_episodes), critic_losses_per_episode)
    ax1.set_title('Critic Loss vs Episodes')
    ax2.plot(range(number_of_episodes), actor_losses_per_episode)
    ax2.set_title('Actor Loss vs Episodes')
    ax1.set(xlabel='Episode', ylabel='Loss')
    ax2.set(xlabel='Episode', ylabel='Loss')
    plt.savefig("./runs/" + timestamp_str + "_losses.png")
    df = pd.DataFrame({"aloss": actor_losses_per_episode, "closs": critic_losses_per_episode})
    df.to_csv("./runs/" + timestamp_str + "_losses.csv", index=False)
    end_t = time.time()
    print('exec time sec: ', end_t - start_t, ' per episode: ', (end_t - start_t) / number_of_episodes)
    print(f"Saving Trial {initialization_types[i]} Checkpoint")
    agent.save_checkpoint(timestamp, f"./Checkpoints/Checkpoint-{initialization_types[i]}.gm")

    
    # Test
    test_rewards = []
    test_cv = []
    test_time_in_range = []
    test_episode_lengths = []
    test_episodes = 5
    for episode in range(10):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        while not done:
            # env.render('human')
            episode_length += 1
            action = agent.act(torch.Tensor(state))
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
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
        test_episode_lengths.append(episode_length)
    mean_cv = sum(test_cv)/len(test_cv)
    mean_time_in_range = sum(test_time_in_range)/len(test_time_in_range)
    mean_rewards = sum(test_rewards)/len(test_rewards)
    average_episode_length = sum(test_episode_lengths)/len(test_episode_lengths)

    sys.stdout.write(f"Mean CV: {mean_cv} \nMean Time in Range: {mean_time_in_range} \nMean Rewards: {mean_rewards} \nAverage Episode Length {average_episode_length}\r\n")
    
    
    with open(csv_filename, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_row = [initialization_types[i], datetime.now(),
                   mean_time_in_range, mean_cv, mean_rewards, average_episode_length]
        csv_writer.writerow(csv_row)
    