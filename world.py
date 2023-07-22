import gym
import main
import random
from gym import error, spaces
from gym.error import DependencyNotInstalled
try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")


class World:
    def __init__(self,
                 gravity: float = main.GRAVITY,
                 width: int = main.SCREEN_WIDTH,
                 height: int = main.SCREEN_HEIGHT,
                 difficulty: int = main.DIFFICULTY
                 ):
        self.gravity = gravity
        self.width = width
        self.height = height
        self.difficulty = difficulty
        self.world = b2World(b2Vec2(0, gravity), True)
