#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 20:17:00 2022

@author: nihalsatyadev
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

hyperparameter_set = 8

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
hidden_size = [32,64,128,256]
learning_rate = [1e-4,1e-3,1e-2]

replay_buffer_size = [100,1000,10000,100000]
batch_size = [32,64,128,256]

gamma = [0.9,0.99,0.999]                           # DDPG - Future Discounted Rewards amount
tau = [0.001,0.01]                             # DDPG - Target network update rate
sigma = [1,2,3]                             # OUNoise sigma - used for exploration
theta = [0.01,0.1,1]                             # OUNoise theta - used for exploration
dt = 1e-2                               # OUNoise dt - used for exploration
number_of_episodes = 2000              # Total number of episodes to train for
validation_rate = 25                    # Run validation every n episodes


csv_filename = 'grid_search.csv'
csv_fields = ['hyperparameter_set', 'date_time',
              'hidden_size', 'replay_buffer_size', 'batch_size', 'learning_rate', 'gamma', 'tau', 'sigma', 'theta',
              'time_in_range', 'coefficient_of_variance', 'total_reward']

if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_fields = ['hyperparameter_set', 'date_time',
                      'hidden_size', 'learning_rate', 'replay_buffer_size', 'batch_size', 'gamma', 'tau', 'sigma',
                      'theta',
                      'time_in_range', 'coefficient_of_variance', 'total_reward']
        csv_writer.writerow(csv_fields)

tuning_hyperparameter_names = ['hidden_size','learning_rate','replay_buffer_size','batch_size','gamma','tau','sigma','theta']
hyperparameter_values = [hidden_size,learning_rate,replay_buffer_size,batch_size,gamma,tau,sigma,theta]

hyperparameter_dict = {}

for i in range(0,len(tuning_hyperparameter_names)):
    hyperparameter_dict[tuning_hyperparameter_names[i]] = hyperparameter_values[i].copy()



freeze_value = None
freeze_hyperparameter = None
moving_hyperparameter = None
if hyperparameter_set % 2 == 1:
    freeze_value = 'min'
else:
    freeze_value = 'max'

if hyperparameter_set == 1 or hyperparameter_set == 2:
    freeze_hyperparameter = 'hidden_size'
    moving_hyperparameter = 'learning_rate'
elif hyperparameter_set == 3 or hyperparameter_set == 4:
    freeze_hyperparameter = 'replay_buffer_size'
    moving_hyperparameter = 'batch_size'
elif hyperparameter_set == 5 or hyperparameter_set == 6:
    freeze_hyperparameter = 'gamma'
    moving_hyperparameter = 'tau'
elif hyperparameter_set == 7 or hyperparameter_set == 8:
    freeze_hyperparameter = 'sigma'
    moving_hyperparameter = 'theta'
    
for key, value in hyperparameter_dict.items():
    if key == freeze_hyperparameter:
        if freeze_value == 'min':
            hyperparameter_dict[key] = min(value)
        elif freeze_value == 'max':
            hyperparameter_dict[key] = max(value)
    elif not key == moving_hyperparameter:
        value.sort()
        mid = len(value) // 2
        hyperparameter_dict[key] = value[mid]  

temp_moving_hyperparameter = hyperparameter_dict[moving_hyperparameter].copy()
for k in range(0,2):
    if k == 0:
        hyperparameter_dict[moving_hyperparameter] = min(temp_moving_hyperparameter)
    if k == 1:
        hyperparameter_dict[moving_hyperparameter] = max(temp_moving_hyperparameter)

    hidden_size = hyperparameter_dict['hidden_size']
    learning_rate = hyperparameter_dict['learning_rate']
    replay_buffer_size = hyperparameter_dict['replay_buffer_size']
    batch_size = hyperparameter_dict['batch_size']
    gamma = hyperparameter_dict['gamma']                           # DDPG - Future Discounted Rewards amount
    tau = hyperparameter_dict['tau']                             # DDPG - Target network update rate
    sigma = hyperparameter_dict['sigma']                             # OUNoise sigma - used for exploration
    theta = hyperparameter_dict['theta']
    
    print("My hyperparameters are:")
    print(hyperparameter_dict)

    actor_hidden_size = hidden_size
    critic_hidden_size = hidden_size
    
    lr_actor = learning_rate
    lr_critic = learning_rate

    agent = DDPG(state_size, action_space, actor_hidden_size, critic_hidden_size, replay_buffer_size, batch_size,
                 lr_actor, lr_critic, gamma, tau, sigma, theta, dt)

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

    
    # Test
    test_rewards = []
    test_cv = []
    test_time_in_range = []
    test_episodes = 5
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # env.render('human')
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
    mean_cv = sum(test_cv)/len(test_cv)
    mean_time_in_range = sum(test_time_in_range)/len(test_time_in_range)
    mean_rewards = sum(test_rewards)/len(test_rewards)

    sys.stdout.write(f"Mean CV: {mean_cv} \nMean Time in Range: {mean_time_in_range} \nMean Rewards: {mean_rewards} \r\n")

    with open(csv_filename, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_row = [hyperparameter_set, datetime.now(),
                   hidden_size, replay_buffer_size, batch_size, learning_rate, gamma, tau, sigma, theta,
                   mean_time_in_range, mean_cv, mean_rewards]
        csv_writer.writerow(csv_row)