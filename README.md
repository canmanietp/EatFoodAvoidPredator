This code has been made available to students of the course CS4210-B - Intelligent Decision Making.

# Eat-Food-Avoid-Predator domain<sup>1</sup>

The agent is a rabbit in a grid world. A food item and wolf also exist in the world. The rabbit gets reward of 0.5 for every time step it avoids the wolf. The rabbit gets reward of 1 every time it reaches a food. When food is `got', it reappears randomly somewhere in the world. The wolf moves toward the rabbit every other time step. Each episode ends when the rabbit shares a grid with the wolf.

<sub><sup>1</sup>Satinder Singh and David Cohn. How to Dynamically Merge Markov Decision Processes. Advances in Neural Information Processing Systems 10, pages 1057-1063, 1998.</sub>

## Necessary python packages
numpy <br />
matplotlib

## To test
run ```python test.py``` in the console

### Things to play around with
- the reward function <br />
- epsilon (when agent is `smart' enough, you will find that if epsilon decays to a value that is too low the rabbit will never get caught by the wolf) <br />
- implementing maximum number of episodes (comment out the appropriate line in ```discreteWorld.py```) <br />
- size of state space <br />
- number of actions available to agent <br />


