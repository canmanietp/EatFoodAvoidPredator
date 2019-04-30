import numpy as np
import discreteWorld

nR = 5 # number of rows
nC = 5 # number of columns


class RabbitWorld(discreteWorld.DiscreteEnv):
    """
    Actions:
    There are 4 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    """

    def __init__(self):
        nS = (nR*nC)**3  # rabbit locations x food locations x wolf locations
        maxR = nR - 1
        maxC = nC - 1
        nA = 4

        isd = np.zeros(nS)
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for row in range(nR):
            for col in range(nC):
                    for food_idx in range(nR*nC):
                        for wolf_idx in range(nR*nC):
                            state = self.encode(row, col, food_idx, wolf_idx)

                            for act in range(nA):
                                newrow, newcol, newfood_idx, newwolf_idx = row, col, food_idx, wolf_idx
                                done = False
                                reward = 0.5 # reward ever time step

                                # if wolf caught rabbit, episode ends
                                if (row * nR + col) == wolf_idx:
                                    reward = 0 
                                    done = True
                                else:
                                    isd[state] += 1
                                    # for all the actions
                                    if act == 0:  # south
                                        newrow = min(row+1, maxR)
                                    elif act == 1:  # north
                                        newrow = max(row-1, 0)
                                    elif act == 2:  # east
                                        newcol = min(col+1, maxC)
                                    elif act == 3:  # west
                                        newcol = max(col-1, 0)

                                    if (newrow * nR + newcol) == food_idx:
                                        reward = 1 # reward for eating food

                                    wolfcol = wolf_idx % nC
                                    wolfrow = int((wolf_idx - wolfcol) / nR)
                                    newwolfcol = wolfcol
                                    newwolfrow = wolfrow

                                    # wolf moves toward rabbit every time step
                                    if wolfrow == newrow:  # if wolf and rabbit are in same row
                                        if wolfcol > newcol:
                                            newwolfcol = max(wolfcol - 1, 0)
                                        else:
                                            newwolfcol = min(wolfcol + 1, maxC)
                                    else:  # move up or down to get closer to rabbit
                                        if wolfrow > newrow:
                                            newwolfrow = max(wolfrow - 1, 0)
                                        else:
                                            newwolfrow = min(wolfrow + 1, maxR)

                                    newwolf_idx = newwolfrow * nR + newwolfcol

                                newstate = self.encode(newrow, newcol, newfood_idx, newwolf_idx)
                                P[state][act].append((1.0, newstate, reward, done))

        isd /= isd.sum()
        discreteWorld.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, rabrow, rabcol, food_idx, wolf_idx):
        # (5) 5, 25, 25
        # (nR) nC, 25, 25
        i = rabrow
        i *= nC  # number of rab col ids
        i += rabcol
        i *= nC*nR  # number of food ids
        i += food_idx
        i *= nC*nR  # number of wolf ids
        i += wolf_idx
        #if i < 0 : print(rabrow, rabcol, food_idx, wolf_idx)
        return i

    def decode(self, i):
        out = []
        out.append(i % (nC*nR))
        i = i // (nC*nR)
        out.append(i % (nC*nR))
        i = i // (nC*nR)
        out.append(i % nR)
        i = i // nR
        out.append(i)
        assert 0 <= i < nR
        return reversed(out)
