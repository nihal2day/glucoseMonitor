import os
import sys
import warnings
from datetime import datetime
import numpy as np
import gym
from gym.envs.registration import register
import torch
from torch.utils.tensorboard import SummaryWriter

from DDPG.DDPG import DDPG
from Normalized_Actions import NormalizedActions


os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore", category=DeprecationWarning)


def custom_reward(bg_last_hour, slope=None):
    bg = bg_last_hour[-1]
    if bg >= 202.46:
        x = [202.46, 350]
        y = [-15, -20]
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
hidden_size = 128
learning_rate = 1e-4
actor_hidden_size = hidden_size
critic_hidden_size = hidden_size
replay_buffer_size = 100000
batch_size = 256
lr_actor = learning_rate
lr_critic = learning_rate
gamma = 0.99                           # DDPG - Future Discounted Rewards amount
tau = 0.001                             # DDPG - Target network update rate
sigma = 2.5                             # OUNoise sigma - used for exploration
theta = 0.5                             # OUNoise theta - used for exploration
dt = 1e-2                               # OUNoise dt - used for exploration
number_of_episodes = 20              # Total number of episodes to train for
save_checkpoint_rate = 250             # Save checkpoint every n episodes
validation_rate = 25                    # Run validation every n episodes

agent = DDPG(state_size, action_space, actor_hidden_size, critic_hidden_size, replay_buffer_size, batch_size,
             lr_actor, lr_critic, gamma, tau, sigma, theta, dt)

# Load Checkpoint if set
load_checkpoint = False
if load_checkpoint:
    agent.load_checkpoint(f"./Checkpoints/CheckpointFinal-12-04-2022_0523.gm")

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

    # Save Checkpoint every save_checkpoint_rate episodes
# =============================================================================
#     if episode % save_checkpoint_rate == 0 and episode != 0:
#         print("Saving checkpoint")
#         timestamp = datetime.timestamp(datetime.now())
#         agent.save_checkpoint(timestamp, f"./Checkpoints/Checkpoint{episode}-{datetime.now().strftime('%m-%d-%Y_%H%M')}.gm")
# =============================================================================

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


# =============================================================================
# print("Saving Final Trained Checkpoint")
# agent.save_checkpoint(timestamp, f"./Checkpoints/CheckpointFinal-{datetime.now().strftime('%m-%d-%Y_%H%M')}.gm")
# =============================================================================

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
