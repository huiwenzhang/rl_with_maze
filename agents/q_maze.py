"""
Solve maze problem with q-learning
"""

import gym, numpy as np, random
import gym_maze

# 超参数
num_epis_train = 5000
learning_rate = 0.01
discount = 0.95
eps_start = 0.25

# 加载一个固定的迷宫地图，如果是MazeRandom5x5环境，则随机生成一个迷宫地图
env = gym.make("MazeSample5x5-v0")
# 动作值函数
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
def q_train():
    for epis in range(num_epis_train):
        print("Training episode: {}".format(epis))
        state = env.reset()
        step = 0
        epsilon = epsilon_decay(eps_start, num_epis_train, epis)
        while True:
            action = epsilon_greedy_policy(Q, state, epsilon)
            state_new, reward, done, _ = env.step(env.ACTION[action])
            # env.render()
            x, y, x_, y_ = int(state[0]), int(state[1]), int(state_new[0]), int(state_new[1])
            Q[x, y, action] = Q[x, y, action] + learning_rate * (reward + discount * np.max(
                Q[x_, y_, :]) - Q[x, y, action])
            state = state_new
            step += 1
            if done:
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
    q_train()
    test()
