import random

import torch


class RandomAgent:
    def __init__(self, action_num):
        self._action_list = range(0, action_num)

    def select_action(self, state, action_sample):
        return torch.tensor(random.choice(self._action_list))

    def learn(self, state, action, next_state, reward):
        # Random Agent cant learn :(
        pass
