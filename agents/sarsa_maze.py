"""
Solve maze problem with SARSA
"""

import gym, numpy as np
import gym_maze
import random
import matplotlib.pyplot as plt

# 超参数
num_epis_train = 5000  # 训练的回合
learning_rate = 0.01  # 值函数更新的步长
discount = 0.95  # 折扣系数
eps_start = 0.4  # 探索频率

# 初始化迷宫环境
env = gym.make("MazeSample5x5-v0")
# 初始化一个表格，用于存放迷宫中每个cell的Q值
Q = np.zeros([*env.maze_size, env.action_space.n])


# 线性的衰减探索的概率
def epsilon_decay(starting_epsilon, iterations, i):
    decay = starting_epsilon / 10.
    current_step = int(i / (iterations / 10))
    return starting_epsilon - decay * current_step


# epsilon-贪婪策略
def epsilon_greedy_policy(q, observation, epsilon=0.05, greedy=False):
    x, y = observation
    x, y = int(x), int(y)
    if greedy:
        return np.argmax(q[x, y, :])

    most_greedy_action = np.argmax(q[x, y, :])
    actions_count = q[x, y, :].shape[0]
    weights = [epsilon] * actions_count
    weights[most_greedy_action] += 1 - actions_count * epsilon

    return random.choices(list(range(actions_count)), weights=weights, k=1)[0]


# 训练
def sarsa_train():
    for epis in range(num_epis_train):
        # 一个回合为一个循环，更新值函数
        print("Training episode: {}".format(epis))
        state = env.reset()
        # 采用的是epsilon贪婪策略，因此会以epsilon的概率随机选择动作，这里对探索概率做了线性衰减
        epsilon = epsilon_decay(eps_start, num_epis_train, epis)
        action = epsilon_greedy_policy(Q, state, epsilon)
        step = 0
        while True:
            # 执行当前的动作，获得下一步的状态、回报和相关信息
            state_new, reward, done, _ = env.step(env.ACTION[action])
            # 这里因为要用状态作为索引，因此变化为int类型
            x, y, x_, y_ = int(state[0]), int(state[1]), int(state_new[0]), int(state_new[1])
            # 获得下一个时刻的动作
            action_new = epsilon_greedy_policy(Q, state_new, epsilon)

            # 更新值函数
            Q[x, y, action] = Q[x, y, action] + learning_rate * (
                    reward + discount * Q[x_, y_, action_new] - Q[x, y, action])
            # 更新当前状态和动作为下一个时刻的状态和动作
            state = state_new
            action = action_new
            step += 1
            if done:
                # 因为迷宫环境的最大步长是500，这里做了步数判断，以分辨是否到达目标
                if step < 500:
                    print("Reach goal use {} steps".format(step))
                break


# 测试
def test(num_episode=50):
    success = 0
    for epi in range(num_episode):
        print("Test episode: {}".format(epi))
        state = env.reset()
        while True:
            action = epsilon_greedy_policy(Q, state)
            state_new, reward_episode, done, _ = env.step(env.ACTION[action])
            env.render()
            state = state_new
            if done:
                if reward_episode == 5:
                    success += 1
                break

    print('---Success rate=%.3f' % (success * 1.0 / num_episode))
    print('-------------------------------')


if __name__ == "__main__":
    sarsa_train()
    test()
