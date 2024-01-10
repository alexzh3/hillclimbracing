import hill_racing
from Box2D import *


class World:
    def __init__(self,
                 gravity: float = hill_racing.GRAVITY,
                 width: int = hill_racing.SCREEN_WIDTH,
                 height: int = hill_racing.SCREEN_HEIGHT,
                 difficulty: int = hill_racing.DIFFICULTY
                 ):
        self.gravity = gravity
        self.width = width
        self.height = height
        self.difficulty = difficulty
        self.world = b2World(b2Vec2(0, gravity), True)
