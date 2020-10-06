import gym

env = gym.make('FrozenLake-v0')
obs = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    env.step(action)
    env.render()

