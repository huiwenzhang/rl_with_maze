#!/usr/bin/env python
from gym.envs.registration import register

register(
    id = 'maze-random-5x5-v0',
    entry_point = 'env.maze_envs:MazeEnvRandom5x5',
    max_episode_steps = 2000
)