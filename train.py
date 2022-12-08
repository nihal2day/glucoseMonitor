import os
import sys
import warnings
from datetime import datetime
import numpy as np
import gym
from gym.envs.registration import register
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from DDPG.DDPG import DDPG
from Normalized_Actions import NormalizedActions
import matplotlib.pyplot as plt
import time


os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore", category=DeprecationWarning)


def custom_reward(bg_last_hour, slope=0, insulin=0):
    bg_now = bg_last_hour[-1]
    punishment = 0
    if bg_now < 120 and slope < 0 and insulin > 1:
        punishment = -5
    if 70 <= bg_now <= 180:
        return 0.5 + punishment
    elif 180 < bg_now <= 300:
        return -0.8
    elif 300 < bg_now <= 350:
        return -1
    elif 30 <= bg_now < 70:
        return -1.5 + punishment
    else:
        return -2.0 + punishment


def custom_reward_2(bg_last_hour, slope=0, insulin=0):
    bg = bg_last_hour[-1]
    punishment = 0
    if bg < 120 and slope < 0 and insulin > 1:
        punishment = -5
    if bg >= 202.46:
        x = [202.46, 350]
        y = [-15, -20]
        return np.interp(bg, x, y)
    if bg <= 70.729:
        return -0.025 * (bg - 95) ** 2 + 15 + punishment
    else:
        return -0.005 * (bg - 125) ** 2 + 15 + punishment

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
actor_hidden_size = 128
critic_hidden_size = 128
replay_buffer_size = 1000
batch_size = 32
lr_actor = 1e-3
lr_critic = 1e-3
gamma = 0.95                           # DDPG - Future Discounted Rewards amount
tau = 0.001                             # DDPG - Target network update rate
sigma = 0.5                             # OUNoise sigma - used for exploration, could add sigma decay
theta = 0.05                             # OUNoise theta - used for exploration
dt = 1e-1                               # OUNoise dt - used for exploration
number_of_episodes = 10               # Total number of episodes to train for
episode_length_limit = 250              # Length of a single episode
save_checkpoint_rate = 250             # Save checkpoint every n episodes
validation_rate = 25                    # Run validation every n episodes
timestamp_str = datetime.now().strftime('%m-%d-%Y_%H%M')
outfile = "./runs/" + timestamp_str + "out.txt"
verbose = True

agent = DDPG(state_size, action_space, actor_hidden_size, critic_hidden_size, replay_buffer_size, batch_size,
             lr_actor, lr_critic, gamma, tau, sigma, theta, dt, gpu='mps', verbose=verbose, outfile=outfile)

# Load Checkpoint if set
load_checkpoint = False
if load_checkpoint:
    agent.load_checkpoint(f"./Checkpoints/CheckpointFinal-12-04-2022_0523.gm")

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
        done = False
        if episode_length >= episode_length_limit:
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
            if verbose:
                with open(outfile, "a+") as f:
                    f.write(f"Episode: {episode} Length: {episode_length} Reward: {episode_reward} MinAction: {min_action} MaxAction: {max_action} \r\n\r\n")
            print(f"Episode: {episode} Length: {episode_length} Reward: {episode_reward} MinAction: {min_action} MaxAction: {max_action} \r\n")
            critic_losses, actor_losses = agent.get_losses()
            critic_losses_per_episode[episode] = np.mean(critic_losses)
            actor_losses_per_episode[episode] = np.mean(actor_losses)

    # Save Checkpoint every save_checkpoint_rate episodes
    if episode % save_checkpoint_rate == 0 and episode != 0:
        print("Saving checkpoint")
        timestamp = datetime.timestamp(datetime.now())
        agent.save_checkpoint(timestamp, f"./Checkpoints/Checkpoint{episode}-{datetime.now().strftime('%m-%d-%Y_%H%M')}.gm")

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


print("Saving Final Trained Checkpoint")
timestamp = datetime.timestamp(datetime.now())
agent.save_checkpoint(timestamp, f"./Checkpoints/CheckpointFinal-{datetime.now().strftime('%m-%d-%Y_%H%M')}.gm")
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
sim_results = env.render()
# sim_results.to_csv('./runs/' + timestamp_str + '.csv')
end_t = time.time()
print('exec time sec: ', end_t - start_t, ' per episode: ', (end_t - start_t) / number_of_episodes)
# Test
# test_rewards = []
# for episode in range(5):
#     state = env.reset()
#     episode_reward = 0
#     done = False
#     while not done:
#         env.render('human')
#         action = agent.act(torch.Tensor(state))
#         next_state, reward, done, _ = env.step(action)
#         state = next_state
#         episode_reward += reward
#         if done:
#             sys.stdout.write(f"Episode: {episode} Reward: {episode_reward} \r\n")
#
#     test_rewards.append(episode_reward)
