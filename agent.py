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
        self.shadow_dead = False  # When agent has died but the dead count hasn't been increased yet
        self.score = 0
        self.world = real_world
        self.last_grounded = 0
        self.car = None
        self.dead_count = 50
        self.motor_state = 2
        self.spawning_y = spawning_y
        self.x = 350
        self.shirt_color_R = math.floor(random.randint(0, 255))
        self.shirt_color_G = math.floor(random.randint(0, 255))
        self.shirt_color_B = math.floor(random.randint(0, 255))

    def add_to_world(self):
        self.car = car.Car(x=self.x, y=self.spawning_y, world=self.world)
        self.car.set_shirt_colour()

    def draw_agent(self):
        if not self.shadow_dead or self.dead_count > 0:  # Draw car when agent has died less than dead count amount
            self.car.draw_person_car()
            if main.SHOWING_GROUND:
                main.grounds[0].draw_ground()
                main.SHOWING_GROUND = True
