from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")
import numpy as np
import pygame
import main
import random
import noise


class Ground:
    def __init__(self, x, y, world=None, player=None):
        self.world = world
        self.player = player
        self.wheels = []
        self.starting_position = pygame.Vector2(x, y)
        self.chassis_body = None
        self.chassis_width = 125
        self.chassis_height = 40
        self.wheel_size = 17
        self.dead = False
        self.change_counter = 0
        self.shapes = []
        self.car_density = 1
        self.car_restitution = 0.01
        self.max_distance = 0
        self.motor_state = 0

        # Create body, fixture and shape for car



