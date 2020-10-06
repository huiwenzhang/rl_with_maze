from setuptools import setup
setup(
    name="rl_gym_maze",
    version="0.0",
    author="Alvin",
    packages=["gym_maze", "agents", "gym_maze.maze_envs"],
    install_requires=["gym", "pygame", "numpy"]
)