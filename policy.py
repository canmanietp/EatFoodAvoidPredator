# Given a Q table and state, choose an action

# act on e-greedy policy
import numpy as np
import random


def e_greedy(q_table, state, nA, e):

    if random.uniform(0, 1) < e:
        action = np.random.randint(0, nA)
    else:
        action = np.argmax(q_table[state])  # Exploit learned values

    return action


def greedy(q_table, state):
    return np.argmax(q_table[state])
