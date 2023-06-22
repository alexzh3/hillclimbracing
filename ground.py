import Box2D
import numpy
import numpy as np
import world
from world import World
import random
import noise


class Ground(World):
    def __init__(self):
        super().__init__()
        self.world = World
        self.ground_vectors = []
        self.distance = 15 * self.width
        self.x = 0
        self.y = 0
        self.smoothness = 15
        self.grass_thickness = 5
        self.steepness = 250
        self.grass_positions = []
        self.steepness_Level = 50 + self.difficulty
        self.estimated_difficulty = 0

    def randomizeGround(self):
        startingPoint = random.randint(0, 100000)
        totalDifference = 0
        for i in range(0, self.distance, self.smoothness):
            self.steepness_Level = np.interp(i, [0, self.distance], [130, 250])
            flatLength = 500
            noisedY = noise.pnoise1(startingPoint + (i - flatLength) / (700 - self.steepness_Level))
            maxHeight = 300 + np.interp(self.steepness_Level, [0, 200], [0, 350])
            minHeight = 30
            heightAddition = 0
            if i < flatLength:
                noisedY = noise.pnoise1(startingPoint)
                heightAddition = (flatLength - i) / 7

            self.ground_vectors.append(Box2D.b2Vec2(i, self.height - np.interp(noisedY, [0, 1], [minHeight, maxHeight]) + heightAddition))
            if i > 0:
                totalDifference += abs(self.ground_vectors[-2].y - self.ground_vectors[-1].y)
