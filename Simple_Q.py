import world
import policy

import numpy as np
import pickle

nA = 4 # Number of actions
nC = 5 # Number of columns
nR = 5 # Number of rows
discount = 0.95

def run(num_episodes):
    env = world.RabbitWorld()
    epsilon = 0.5
    alpha = 0.5
    episode_rewards = []
    Q_table = np.zeros([(nC*nR)**3, nA])

    for i in range(num_episodes):
        #print("Simple Q episode: " + str(i))

        ep_reward = 0
        done = False

        state = env.reset()

        got_food = 0
        actions = 0

        action = policy.e_greedy(Q_table, state, nA, epsilon)

        while not done:

            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            next_action = policy.e_greedy(Q_table, next_state, nA, epsilon)

            Q_table[state][action] = Q_table[state][action] + alpha*(reward + discount*Q_table[next_state][next_action] - Q_table[state][action])
            state = next_state
            action = next_action

        episode_rewards.append(ep_reward)

        if alpha > 0.01:
            alpha *= 0.99

    return np.array(episode_rewards)
