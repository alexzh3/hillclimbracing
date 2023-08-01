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

        # vertices for car chassis
        vertices = []
        vertices.append(b2Vec2(-self.chassis_width / 2, 0 - self.chassis_height / 2))
        vertices.append(b2Vec2(self.chassis_width / 4 + 5, 0 - self.chassis_height / 2))
        vertices.append(b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 + 5))
        vertices.append(b2Vec2(self.chassis_width / 2, self.chassis_height / 2))
        vertices.append(b2Vec2(self.chassis_width / 2, self.chassis_height / 2))

        # Scale vertices
        for vertex in vertices:
            vertex.x /= main.SCALE
            vertex.y /= main.SCALE

        # Create body and fixture for car
        car_body = b2BodyDef()
        car_body.type = b2_dynamicBody
        car_body.position.x = x / main.SCALE
        car_body.position.y = y / main.SCALE
        car_body.angle = 0
        car_fixture = b2FixtureDef()
        car_fixture.density = self.car_density
        car_fixture.friction = 0.5
        car_fixture.restitution = self.car_restitution
        car_fixture.shape = b2PolygonShape()
        # b2PolygonShape(vertices=[(0, 0), (1, 0), (0, 1)])

        # Connect shape and body with fixture and vertex
        car_fixture.shape.set_vertex(vertices)
        self.shapes.append(vertices)