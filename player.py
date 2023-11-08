import random, math, pygame
import main, wheels, person
from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")


class Player:
    def __init__(self, real_world):
        self.dead = False
        self.score = 0
        self.world = real_world
        self.shirt_color_R = math.floor(random.randint(0, 255))
        self.shirt_color_G = math.floor(random.randint(0, 255))
        self.shirt_color_B = math.floor(random.randint(0, 255))
        self.last_grounded = 0
        self.car = None
        self.dead_count = 50
        self.motor_state = 2

    def add_to_world(self):
        ...
