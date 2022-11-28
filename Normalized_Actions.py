# Reference: https://github.com/schneimo/ddpg-pytorch/blob/master/wrappers/normalized_actions.py
import gym

class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        action = (action + 1) / 2
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action
