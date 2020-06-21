import gym
import env.maze_envs

task = 'maze-random-5x5-v0'
MAX_STEPS = 5000

if __name__ == "__main__":
    env = gym.make(task)
    env.reset()

    for _ in range(MAX_STEPS):
        action = env.action_space.sample()
        obs_, r, done, info = env.step(action)
        env.render()
