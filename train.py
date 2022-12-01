import os
import sys
import warnings
from datetime import datetime
import numpy as np
import gym
from gym.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
import torch
from torch.utils.tensorboard import SummaryWriter

from DDPG.DDPG import DDPG
from Normalized_Actions import NormalizedActions


#import platform
# print(platform.mac_ver())

os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore", category=DeprecationWarning)


def custom_reward(bg_last_hour):
    if bg_last_hour[-1] > 300:
        return -4.0
    if bg_last_hour[-1] > 250:
        return -2.0
    if bg_last_hour[-1] > 180:
        return -1.0
    if bg_last_hour[-1] < 60:
        return -6.0
    if bg_last_hour[-1] < 70:
        return -1.5
    else:
        return 0.5

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
actor_hidden_size = 512
critic_hidden_size = 512
replay_buffer_size = 100000
batch_size = 512
lr_actor = 1e-3
lr_critic = 1e-4
gamma = 0.999                           # DDPG - Future Discounted Rewards amount
tau = 0.001                             # DDPG - Target network update rate
sigma = 0.2                             # OUNoise sigma - used for exploration
theta = .15                             # OUNoise theta - used for exploration
dt = 1e-2                               # OUNoise dt - used for exploration
number_of_episodes = 10000              # Total number of episodes to train for
episode_length_limit = 250              # Length of a single episode
save_checkpoint_rate = 1000             # Save checkpoint every n episodes


agent = DDPG(state_size, action_space, actor_hidden_size, critic_hidden_size, replay_buffer_size, batch_size,
             lr_actor, lr_critic, gamma, tau, sigma, theta, dt)

# Load Checkpoint if set
load_checkpoint = False
if load_checkpoint:
    agent.load_checkpoint(f"./Checkpoints/CheckpointFinal-XXXX.gm")

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
        next_state, reward, _, _ = env.step(action)
        done = False
        if episode_length == episode_length_limit:
            done = True
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

    # Save Checkpoint every save_checkpoint_rate episodes
    if episode % save_checkpoint_rate == 0 and episode != 0:
        print("Saving checkpoint")
        timestamp = datetime.timestamp(datetime.now())
        agent.save_checkpoint(timestamp, f"./Checkpoints/Checkpoint{episode}-{datetime.now().strftime('%m-%d-%Y_%H%M')}.gm")
    # TODO: Need to add periodic validation.
    #  Validate agent in environment without noise and post scalar results to tensorboard
    writer.add_scalar('Train episode/reward', episode_reward, episode)

print("Saving Final Trained Checkpoint")
agent.save_checkpoint(timestamp, f"./Checkpoints/CheckpointFinal-{datetime.now().strftime('%m-%d-%Y_%H%M')}.gm")

# Test
test_rewards = []
for episode in range(5):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        env.render('human')
        action = agent.act(torch.Tensor(state))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        episode_reward += reward
        if done:
            sys.stdout.write(f"Episode: {episode} Reward: {episode_reward} \r\n")

    test_rewards.append(episode_reward)
