import numpy as np
import noise, random

for i in range(10, 15000, 15):
    flatLength = 500
    startingPoint = random.randint(0, 100000)
    steepness_Level = np.interp(i, [0, 14000], [130, 140])
    noisedY = abs(noise.pnoise1(startingPoint + (i - flatLength) / (700 - steepness_Level), octaves=4))
    print(f"noised {noisedY}, i {i}")