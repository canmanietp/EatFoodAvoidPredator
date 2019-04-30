import matplotlib.pyplot as plt
import numpy as np

import SARSA

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

S_rewards = SARSA.run(50000)
S_plot = moving_average(S_rewards, 5000) # smooth plot with a moving average
plt.plot(S_plot)
plt.show()
