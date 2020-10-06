#!/usr/bin/gym_maze python
from gym.envs.registration import register

register(
    id = 'MazeRandom5x5-v0',
    entry_point = 'gym_maze.maze_envs:MazeEnvRandom5x5',
    max_episode_steps = 500
)

register(
    id = 'MazeSample5x5-v0',
    entry_point = 'gym_maze.maze_envs:MazeEnvSample5x5',
    max_episode_steps = 500
)

register(
    id = 'MazeRandom10x10-v0',
    entry_point = 'gym_maze.maze_envs:MazeEnvRandom10x10',
    max_episode_steps = 500
)

register(
    id = 'MazeSample10x10-v0',
    entry_point = 'gym_maze.maze_envs:MazeEnvSample10x10',
    max_episode_steps = 500
)