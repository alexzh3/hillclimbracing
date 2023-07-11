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

        # Iterate over a range from 0 to self.distance with a step size of self.smoothness
        for i in range(0, self.distance, self.smoothness):
            # Set the length of the flat section of the ground vector
            flatLength = 500
            # Initialize a variable to store the additional height to be added for the flat section of the ground vector
            heightAddition = 0
            # Calculate the steepness level using linear interpolation between 130 and 250
            self.steepness_Level = np.interp(i, [0, self.distance], [130, 250])
            # Calculate the noisedY value using Perlin noise with the starting point and adjusted i value
            noisedY = noise.pnoise1(startingPoint + (i - flatLength) / (700 - self.steepness_Level), octaves=4)
            # Determine the maximum and minimum heights for the ground vector based on the steepness level
            maxHeight = 300 + np.interp(self.steepness_Level, [0, 200], [0, 350])
            minHeight = 30
            # If the current iteration value is less than the flat section length, recalculate noisedY and heightAddition
            if i < flatLength:
                noisedY = noise.pnoise1(startingPoint, octaves=4)
                heightAddition = (flatLength - i) / 7

            # Create a new Box2D.b2Vec2 object with x-value i and adjusted y-value based on noisedY and heightAddition
            self.ground_vectors.append(
                Box2D.b2Vec2(i, self.height - np.interp(noisedY, [0, 1], [minHeight, maxHeight]) + heightAddition))
            # Calculate the absolute difference between the previous and current y-values and add it to the total difference
            if i > 0:
                totalDifference += abs(self.ground_vectors[-2].y - self.ground_vectors[-1].y)

    # Function to see if ground is too steep
    def groundTooSteep(self):
        for vector in self.ground_vectors:
            oi = self.getPositions(vector.x, 10, 1)
            totalDifference = 0
            for i in range(1, len(oi)):
                totalDifference += max(0, oi[i - 1] - oi[i])
            if totalDifference > 5:
                return True
        return False

    # returns a list of Y positions directly after the input x.
    # the list contains numberOfPositions Y values which represent the upcoming hills
    def getPositions(self, x, numberOfPositions, skip):
        returnList = []
        for i in range(len(self.ground_vectors)):
            if self.ground_vectors[i].x >= x:
                for j in range(0, min(skip * numberOfPositions, len(self.ground_vectors) - i), skip):
                    returnList.append(self.ground_vectors[i + j].y)
                break
        while len(returnList) < numberOfPositions:
            returnList.append(returnList[-1])  # append last element to list again
        return returnList
