
import numpy as np
import gym
import matplotlib.pyplot as plt
from enum import Enum
import random

class Actions(Enum):
    LEFT = 0.
    DOWN = 1.
    RIGHT = 2.
    UP = 3.

# 利用policy产生一条的仿真轨迹，并计算其累积回报
def run_episode(env, policy, gamma=1.0, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        # 计算折扣累积回报
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

# 利用policy产生多个episode，计算平均得分
def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

# 主要步骤２：策略提升
def extract_policy(env, v, gamma=1.0, draw=True):
    # 初始化策略
    policy = np.zeros(env.env.nS-1)
    for s in range(env.env.nS-1):
        q_sa = np.zeros(env.env.nA)
        # 评估当前状态下，每个动作的收益
        for a in range(env.env.nA):
            # 计算动作值函数q
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
        # 更新后的策略是相对于q的贪婪策略，所以使用max操作
        policy[s] = np.random.choice(np.argwhere(q_sa == np.max(q_sa)).flatten())
    if draw:
        draw_value_func(env, v, 1000)
        draw_policy(env, policy, v, 1000)
    # 返回提升的策略，根据策略提升理论，这个策略肯定比上一次迭代的策略要好
    return policy

# 主要步骤１：计算最优值函数
def value_iteration(env, gamma = 1.0):
    # 初始化值函数
    v = np.zeros(env.env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            q_sa = [sum([p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v

def draw_value_func(env, v, iteration):
    row, col = env.nrow, env.ncol
    v = v.reshape((row, col))
    fig, ax = plt.subplots()
    im = ax.imshow(v)

    # 显示地图坐标点的通过性
    for i in range(len(env.desc)):
        for j in range(len(env.desc[0])):
            text = ax.text(j, i, env.desc[i, j].decode('UTF-8') + " " + str(np.around(v[i, j], decimals=3)),
                           ha='center', va='center', color='w')
    ax.set_title("Value map iterated at step {}".format(iteration + 1))
    fig.tight_layout()
    # plt.draw()
    # plt.pause(0.001)
    # input("Press [enter] to continue.")
    plt.show()


def draw_policy(env, policy, v, iteration):
    row, col = env.nrow, env.ncol
    fig, ax = plt.subplots()
    v = v.reshape(row, col)
    ax.imshow(v)
    for i in range(row):
        for j in range(col):
            if i == row -1 and j == col -1:
                continue
            ax.text(j, i, env.desc[i, j].decode('UTF-8'), ha='center', va='center', color='w')
            action = policy[i * col + j]
            if action == 0.0:  # left
                endxy = (i, max(j - 1, 0))
            elif action == 1.0:  # down
                endxy = (min(i + 1, row - 1), j)
            elif action == 2.0:  # right
                endxy = (i, min(j + 1, col - 1))
            elif action == 3.0:  # up
                endxy = (max(i - 1, 0), j)
            else:
                print("Invalid action")
            if i == endxy[0] and j == endxy[1]:
                continue
            plt.arrow(j, i, (endxy[1]-j) * 0.4, (endxy[0]-i) * 0.4, width=0.03)

    ax.set_title("Policy map iterated at step {}".format(iteration + 1))
    fig.tight_layout()
    # plt.draw()
    # input("Press [enter] to continue.")
    plt.show()


if __name__ == '__main__':

    env_name  = 'FrozenLake-v0' # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    gamma = 1.0
    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(env, optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, gamma, n=1000)
    print('Policy average score = ', policy_score)
