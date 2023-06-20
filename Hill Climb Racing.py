from typing import Optional, TYPE_CHECKING
import random
import pygame
import numpy as np
import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
try:
    import Box2D
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

# Define some colors
# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Define constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PPM = 30  # Pixels per meter / Scale
FPS = 60  # frames per second
TIME_STEP = 1.0 / FPS  # time step for physics simulation


class HillRace(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self,
        render_mode: Optional[str] = "human",
        gravity: float = -10.0,
        continuous: bool = False,
        difficulty: float = 10,
    ):
        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.car: Optional[Box2D.b2Body] = None
        self.render_mode = render_mode
        self.gravity = gravity
        self.difficulty = difficulty
        self.continuous = continuous
        self.hill = None

        if self.continuous:
            # Action is two floats [drive, brake]
            # Drive: 0 is off, 0..+1 throttle from 0% to 100% power. Only rear wheel.
            # Brake: So 0 is off, 0..-1 means drive in reverse/stop the car, only the rear wheel.
            self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 0.0]), dtype=np.float32)
        else:
            # Drive is drive, brake is reverse/stopping the car.
            self.action_space = spaces.Discrete(2)

    def _destroy(self):
        if not self.hill:
            return
        self.world.DestroyBody(self.hill)
        self.hill = None
        self.world.DestroyBody(self.car)
        self.car = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._destroy()
        self.game_over = False

        W = SCREEN_WIDTH / PPM
        H = SCREEN_HEIGHT / PPM
        # terrain
        CHUNKS = 20
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]
        self.hill = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.hill.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.hill.color1 = BLUE
        self.hill.color2 = BLUE

    def render(self):
        ...

if __name__ == "__main__":
    env = HillRace(render_mode="human")
    env.reset()  # Reset the environment
    game_over = False
    while True:
        env.render()  # Call the render method to draw the terrain

        # Your code for game logic, actions, and other updates goes here

        # Check for any termination conditions and break the loop if necessary
        if game_over:
            break

    pygame.quit()  # Clean up resources when done
