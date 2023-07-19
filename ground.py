import Box2D
from Box2D import *
import numpy as np
import pygame
import main
import random
import noise


class Ground:
    def __init__(self, world):
        self.world = world
        self.ground_vectors = []
        self.dirtBody = None
        self.grassBody = None
        self.distance = 15 * main.SCREEN_WIDTH
        self.x = 0
        self.y = 0
        self.smoothness = 15
        self.grass_thickness = 5
        self.steepness = 250
        self.grass_positions = []
        self.steepness_Level = 50 + main.DIFFICULTY
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
            # If the current iteration value is less than the flat section length, recalculate noisedY and
            # heightAddition
            if i < flatLength:
                noisedY = noise.pnoise1(startingPoint, octaves=4)
                heightAddition = (flatLength - i) / 7

            # Create a new Box2D.b2Vec2 object with x-value i and adjusted y-value based on noisedY and heightAddition
            self.ground_vectors.append(
                b2Vec2(i, main.SCREEN_HEIGHT - np.interp(noisedY, [0, 1], [minHeight, maxHeight]) + heightAddition))
            # Calculate the absolute difference between the previous and current y-values and add it to the total
            # difference
            if i > 0:
                totalDifference += abs(self.ground_vectors[-2].y - self.ground_vectors[-1].y)

        self.ground_vectors.append(b2Vec2(self.distance, main.SCREEN_HEIGHT + self.grass_thickness * 2))
        self.ground_vectors.append(b2Vec2(0, main.SCREEN_HEIGHT + self.grass_thickness * 2))
        spawningY = self.ground_vectors[10].y - 100

        for vect in self.ground_vectors:
            vect.x /= main.SCALE
            vect.y /= main.SCALE

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

    def cloneFrom(self, otherGround):
        for v in otherGround.vectors:
            self.ground_vectors.append(pygame.Vector2(v.x, v.y))

    def setBodies(self, worldToAddTo):
        self.world = worldToAddTo
        self.makeBody()
        for i in range(1, len(self.ground_vectors)):
            self.addEdge(self.ground_vectors[i - 1], self.ground_vectors[i], main.DIRT_MASK, main.DIRT_CATEGORY, False)

        for i in range(1, len(self.ground_vectors)):
            self.addEdge(
                b2Vec2(self.ground_vectors[i - 1].x, self.ground_vectors[i - 1].y - self.grass_thickness / main.SCALE),
                b2Vec2(self.ground_vectors[i].x, self.ground_vectors[i].y - self.grass_thickness / main.SCALE),
                main.GRASS_MASK, main.GRASS_CATEGORY, True)

    def makeBody(self):
        bodyDef = b2BodyDef()
        bodyDef.type = b2_staticBody
        bodyDef.position.x = 0  # canvas.width / 2 / SCALE
        bodyDef.position.y = 0  # (canvas.height - 20) / SCALE
        self.dirtBody = self.world.CreateBody(bodyDef)
        self.grassBody = self.world.CreateBody(bodyDef)
        self.dirtBody.userData = self
        self.grassBody.userData = self

    def addEdge(self, vec1, vec2, mask, category, isGrass):
        fixDef = b2FixtureDef()
        fixDef.friction = 0.99  # 0.95
        fixDef.restitution = 0.1
        fixDef.shape = b2EdgeShape()

        # Use vertices instead of SetAsEdge
        fixDef.shape.vertices = [vec1, vec2]

        # Use filter instead of filterData
        fixDef.filter.categoryBits = category
        fixDef.filter.maskBits = mask

        if isGrass:
            self.grassBody.CreateFixture(fixDef)
        else:
            self.dirtBody.CreateFixture(fixDef)

    def showGround(self, screen):
        # Light brown
        # ground_color = (102, 50, 20);
        # Brown
        ground_color = (88, 35, 0)
        grass_color = (0, 120, 0)

        # Draw ground and grass hills
        vertices = []
        for i in range(0, len(self.ground_vectors) - 2):  # Scale the ground vectors back
            vertices.append((self.ground_vectors[i].x * main.SCALE, self.ground_vectors[i].y * main.SCALE))
        pygame.draw.polygon(screen, grass_color, vertices, width=self.grass_thickness)  # Draw the grass
        # Append beginning and end vertices of screen
        vertices.append([(self.distance, main.SCREEN_HEIGHT + self.grass_thickness + main.panY),
                         (0, main.SCREEN_HEIGHT + self.grass_thickness + main.panY)])
        pygame.draw.polygon(screen, ground_color, vertices)  # Draw the ground

        # Draw the transition colours from ground to grass (down to up)
        for i in range(len(self.ground_vectors) - 3):
            pygame.draw.line(screen, (66, 60, 0),
                             (self.ground_vectors[i].x * main.SCALE, self.ground_vectors[i].y * main.SCALE + 9),
                             (self.ground_vectors[i + 1].x * main.SCALE, self.ground_vectors[i + 1].y * main.SCALE + 9),
                             3)

        for i in range(len(self.ground_vectors) - 3):
            pygame.draw.line(screen, (44, 90, 0),
                             (self.ground_vectors[i].x * main.SCALE, self.ground_vectors[i].y * main.SCALE + 6),
                             (self.ground_vectors[i + 1].x * main.SCALE, self.ground_vectors[i + 1].y * main.SCALE + 6),
                             3)

        for i in range(len(self.ground_vectors) - 3):
            pygame.draw.line(screen, (0, 140, 0),
                             (self.ground_vectors[i].x * main.SCALE, self.ground_vectors[i].y * main.SCALE - 5),
                             (self.ground_vectors[i + 1].x * main.SCALE, self.ground_vectors[i + 1].y * main.SCALE - 5),
                             3)

        for i in range(len(self.ground_vectors) - 3):
            pygame.draw.line(screen, (0, 130, 0),
                             (self.ground_vectors[i].x * main.SCALE, self.ground_vectors[i].y * main.SCALE - 3),
                             (self.ground_vectors[i + 1].x * main.SCALE, self.ground_vectors[i + 1].y * main.SCALE - 3),
                             3)
