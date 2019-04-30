import matplotlib.pyplot as plt
import numpy as np

import Simple_Q

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

q_rewards = Simple_Q.run(50000)
q_plot = moving_average(q_rewards, 5000) # smooth plot with a moving average
plt.plot(q_plot)
plt.show()
