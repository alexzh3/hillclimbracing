from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")
import pygame
import main
import wheels


class Car:
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
        vectors = []
        vectors.append(b2Vec2(-self.chassis_width / 2, 0 - self.chassis_height / 2))
        vectors.append(b2Vec2(self.chassis_width / 4 + 5, 0 - self.chassis_height / 2))
        vectors.append(b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 + 5))
        vectors.append(b2Vec2(self.chassis_width / 2, self.chassis_height / 2))
        vectors.append(b2Vec2(self.chassis_width / 2, self.chassis_height / 2))
        self.shapes.append(vectors)

        # Scale vertices
        for vector in vectors:
            vector.x /= main.SCALE
            vector.y /= main.SCALE

        # Create main body and fixture for car
        car_body = b2BodyDef(
            type=b2_dynamicBody,
            position=(x / main.SCALE / y / main.SCALE),
            angle=0
        )
        car_fixture = b2FixtureDef(
            categoryBits=main.CHASSIS_CATEGORY,
            maskBits=main.CHASSIS_MASK,
            density=self.car_density,
            friction=0.5,
            restitution=self.car_restitution,
            shape=b2PolygonShape(vertices=vectors)
        )

        # Create body in world and connect fixture to it
        self.chassis_body = self.world.CreateBody(car_body)
        self.chassis_body.CreateFixture(car_fixture)

        # Create front part car
        vectors2 = []
        vectors2.append(b2Vec2(self.chassis_width / 4, 0 - self.chassis_height / 2))
        vectors2.append(b2Vec2(self.chassis_width / 4 - 15, 0 - self.chassis_height / 2 - 20))
        vectors2.append(b2Vec2(self.chassis_width / 4 - 5, 0 - self.chassis_height / 2 - 20))
        vectors2.append(b2Vec2(self.chassis_width / 4 + 10, 0 - self.chassis_height / 2))
        self.shapes.append(vectors2)

        for vector in vectors2:
            vector.x /= main.SCALE
            vector.y /= main.SCALE

        car_fixture2 = b2FixtureDef(
            categoryBits=main.CHASSIS_CATEGORY,
            maskBits=main.CHASSIS_MASK,
            density=self.car_density,
            friction=0.5,
            restitution=self.car_restitution,
            shape=b2PolygonShape(vertices=vectors2)
        )
        self.chassis_body.CreateFixture(car_fixture2)

        # Create back part
        vectors3 = []
        vectors3.append(b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 +5))
        vectors3.append(b2Vec2(self.chassis_width / 2 + 5, 0 - self.chassis_height / 2 + 8))
        vectors3.append(b2Vec2(self.chassis_width / 2 + 5, 0 - self.chassis_height / 2 - 5))
        vectors3.append(b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2))
        self.shapes.append(vectors3)

        for vector in vectors3:
            vector.x /= main.SCALE
            vector.y /= main.SCALE

        car_fixture3 = b2FixtureDef(
            categoryBits=main.CHASSIS_CATEGORY,
            maskBits=main.CHASSIS_MASK,
            density=self.car_density,
            friction=0.1,
            restitution=0.1,
            shape=b2PolygonShape(vertices=vectors3)
        )
        self.chassis_body.CreateFixture(car_fixture3)