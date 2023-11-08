import random, math, pygame
import main, wheels, person, car
from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")


class Agent:
    def __init__(self, real_world, spawning_y):
        self.dead = False
        self.shadow_dead = False # When agent has died but the dead count hasn't been upped yet
        self.score = 0
        self.world = real_world
        self.last_grounded = 0
        self.car = None
        self.dead_count = 50
        self.motor_state = 2
        self.spawning_y = spawning_y

        self.shirt_color_R = math.floor(random.randint(0, 255))
        self.shirt_color_G = math.floor(random.randint(0, 255))
        self.shirt_color_B = math.floor(random.randint(0, 255))

    def add_to_world(self):
        self.car = car.Car(350, self.spawning_y, self.world)
        self.car.set_shirt_colour()

    def draw_agent(self):
        if not self.shadow_dead:
            ...


