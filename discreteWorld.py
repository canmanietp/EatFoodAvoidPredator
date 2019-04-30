import numpy as np
import seeding

max_steps = 100 # Usually not used
nR = 5 # Number of rows
nC = 5 # Number of columns

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv():
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA
        self.step_count = 0

        # self.action_space = spaces.Discrete(self.nA)
        # self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.seed()
        self.lastaction = None
        self.step_count = 0
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]

	# Food appears in random location if eaten
        if r == 1: # dangerously hardcoded here, change value if reward changes
            rabrow, rabcol, food_idx, wolf_idx = self.decode(s)
            s = self.encode(rabrow, rabcol, np.random.randint(nR * nC), wolf_idx)

        # Every other step, wolf does not move
	# Comment this out for a faster wolf
        if self.step_count % 2 == 0:
            rabrow, rabcol, food_idx, ex_wolf_idx = self.decode(self.s)
            rabrow, rabcol, food_idx, wolf_idx = self.decode(s)
            s = self.encode(rabrow, rabcol, food_idx, ex_wolf_idx)

        self.s = s
        self.lastaction = a
        self.step_count += 1

	# Comment this out if you want episode to end after maximum number of steps
        return s, r, d, {"prob": p} 

        if self.step_count > max_steps:
            return s, r, True, {"prob": p}
        else:
            return s, r, d, {"prob": p}

