import gym
import random
from gym import error, spaces
from gym.error import DependencyNotInstalled

try:
    from Box2D import *
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")


class World:
    def __init__(self,
                 gravity: float = 10.0,
                 width: int = 800,
                 height: int = 600,
                 difficulty: int = 50
                 ):
        self.gravity = gravity
        self.width = width
        self.height = height
        self.difficulty = difficulty
        self.world = b2World(b2Vec2(0, gravity), True)
