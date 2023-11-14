import random, math, pygame
import main, wheels, person, car
from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")


class Agent:
    def __init__(self, real_world):
        self.dead = False  # Whether the agent is dead
        self.shadow_dead = False  # When agent has died but the dead count hasn't been increased yet
        self.score = 0
        self.world = real_world
        self.last_grounded = 0
        self.car = None
        self.dead_count = 50  # Amount of maximum deaths allowed
        self.motor_state = 2
        self.x = 350  # Spawn location
        self.shirt_color_R = math.floor(random.randint(0, 255))
        self.shirt_color_G = math.floor(random.randint(0, 255))
        self.shirt_color_B = math.floor(random.randint(0, 255))

    def add_to_world(self):
        self.car = car.Car(x=self.x, y=main.SPAWNING_Y, world=self.world)
        self.car.set_shirt_colour()

    def draw_agent(self):
        if not self.shadow_dead or self.dead_count > 0:  # Draw car when agent has died less than dead count amount
            self.car.draw_person_car()
            if main.SHOWING_GROUND:
                main.grounds[0].draw_ground()
                main.SHOWING_GROUND = True

    def update(self):
        if self.car.dead:  # If the car is dead
            self.shadow_dead = True
        if not self.shadow_dead or self.dead_count > 0:
            self.car.update_status()  # If the car is not dead update its location
        else:
            self.dead = True

        # When the agent dies, update dead count
        if self.shadow_dead:
            self.dead_count -= 1
        self.score = max(1, math.floor((self.car.max_distance - 349)))  # Agent starts at x = 350

        # If agent is officially dead, remove the agent from world
        if self.dead:
            self.remove_agent_from_world()

    # Function that removes body from world
    def remove_agent_from_world(self):
        self.destroy_car()

    def reset_car(self):
        self.destroy_car()
        self.car = car.Car(350, 0, self.world)

    def destroy_car(self):
        self.world.DestroyBody(self.car.chassis_body)
        self.world.DestroyBody(self.car.wheels[0].body)
        self.world.DestroyBody(self.car.wheels[0].rim_body)
        self.world.DestroyBody(self.car.wheels[1].body)
        self.world.DestroyBody(self.car.wheels[1].rim_body)
        self.world.DestroyBody(self.car.person.head.body)
        self.world.DestroyBody(self.car.person.torso.body)
