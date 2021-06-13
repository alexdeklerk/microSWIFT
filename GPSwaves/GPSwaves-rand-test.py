## GPSwaves.py Test Script 
# Written By: EJ Rainville, Spring 2021

# Import Packages
import numpy as np
print('Packages Loaded')

# Define random u and v
num_points = 2400
waveHeight_range = 3
u = waveHeight_range * np.random.random_sample(num_points) 
v = waveHeight_range * np.random.random_sample(num_points) 
z = waveHeight_range * np.random.random_sample(num_points) 
fs = 4

# Run the GPSwaves.py function
from GPSwaves import GPSwaves
Hs, Tp, Dp, E, f, a1, b1, a2, b2 = GPSwaves(u, v, z, fs)
print('Hs = ', Hs)
print('Tp = ', Tp)
print('Dp = ', Dp)
print('E = ', E)