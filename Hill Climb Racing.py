import math
import sys
from abc import ABC
from typing import Optional, TYPE_CHECKING
import warnings

import numpy as np
import gym
import Box2D
import pygame
from pygame.locals import *
import random

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Define constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PPM = 20  # Pixels per meter
WORLD_WIDTH, WORLD_HEIGHT = SCREEN_WIDTH / PPM, SCREEN_HEIGHT / PPM
FPS = 60  # frames per second
TIME_STEP = 1.0 / FPS  # time step for physics simulation

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Random Hill Path')
clock = pygame.time.Clock()


class HillRace():
    def __init__(self):
        self.gravity = -10.0
        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.hill = None
        self.world = None
        self.ground_body = None

    def CreateWorld(self):
        # Create a world with gravity
        self.world = Box2D.b2World(gravity=(0.0, self.gravity), doSleep=True)

    def run(self):
        # Main loop
        while self.isopen:
            # Check for user events and handle them
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.isopen = False

        # Quit pygame and exit the program
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    test = HillRace()
    test.CreateWorld()
    test.run()
