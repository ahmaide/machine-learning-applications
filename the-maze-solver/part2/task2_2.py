import numpy as np
import gymnasium as gym
from typing import Optional
from task2_1 import abstraction
import random

class Env(gym.Env):
    def __init__(self, a_map, agent_location, target_location):
        self.map = a_map
        self.nrows, self.ncols = self.map.shape
        self.initial_location = np.array(agent_location)
        self.agent_location = self.initial_location.copy()
        self.target_location = np.array(target_location)
        self.action_to_direction = np.array([
            [-1, 0],  # up
            [0, 1],  # right
            [1, 0],  # down
            [0, -1],  # left
        ])
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_location = self.initial_location.copy()
        return self.agent_location, {}

    def step(self, action):

        if random.random() < 0.1:
            action = self.action_space.sample()

        direction = self.action_to_direction[action]
        next_loc = np.clip(self.agent_location + direction, [0, 0], [self.nrows - 1, self.ncols - 1])

        y, x = next_loc

        if self.map[y, x] == 1:
            self.agent_location = next_loc
        else:
            #print("blocked at: ",tuple(next_loc))
            pass

        terminated = np.array_equal(self.agent_location, self.target_location)
        truncated = False
        reward = 1.0 if terminated else -0.01
        observation = self.agent_location.copy()

        #print("terminated: ", terminated)

        return observation, reward, terminated, truncated, {}
