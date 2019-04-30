# Simulator of the Eat-Food-Avoid-Predator domain

Agent is a rabbit in a grid world. A food item and wolf also exist in the world. The rabbit gets reward of 0.5 for every time step it avoids the wolf. The rabbit gets reward of 1 for each food it gets to. When food is `got', it reappears randomly somewhere in the world. The wolf moves toward the rabbit every other time step. Each episode ends when the rabbit shares a grid with the wolf.

## Necessary python packages
numpy <br />
matplotlib

## To test
run ```python test.py``` in the console

