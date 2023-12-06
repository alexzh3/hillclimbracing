from gymnasium.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed")
import ground
import agent
import pygame
from typing import Type, TYPE_CHECKING, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

# collisionCategories represented in bits
WHEEL_CATEGORY = 0x0001
CHASSIS_CATEGORY = 0x0002
GRASS_CATEGORY = 0x0004
DIRT_CATEGORY = 0x0008
PERSON_CATEGORY = 0x0010

# collisionMasks, which category it collides with
WHEEL_MASK = GRASS_CATEGORY
CHASSIS_MASK = DIRT_CATEGORY
GRASS_MASK = (WHEEL_CATEGORY | PERSON_CATEGORY)
DIRT_MASK = CHASSIS_CATEGORY
PERSON_MASK = GRASS_CATEGORY

# Define constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCALE = 30  # Pixels per meter / Scale
FPS = 60  # frames per second
DIFFICULTY = 0  # Difficulty of terrain, max 30, min 230 (almost flat terrain)
panX = 0
panY = 0
GRAVITY = 10
WHEEL_SIZE = 35
HEAD_SIZE = 40
PERSON_WIDTH = 20
PERSON_HEIGHT = 40
SPAWNING_Y = 0

# Game variables
NUMBER_OF_WORLDS = 1
grounds = []
worlds = []
HUMAN_PLAYING = True
SHOWING_GROUND = False
RESET_WORLD = False
MAX_CHANGE_COUNTER = 5  # How long without progress in distance made do we continue the game

# Load in pictures/sprites
wheel_sprite = pygame.image.load("pictures/wheel.png")
head_sprite = pygame.image.load("pictures/headLarge2.png")
car_sprite = pygame.image.load("pictures/car.png")
torso_sprite = pygame.image.load("pictures/torsoLarge.png")


# Contact listener for head and ground
class ContactListener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact: b2Contact) -> None:
        world = contact.fixtureA.body.world
        # Fixture variables
        head_fixture = None
        ground_fixture = None

        # If we contact the head with the ground or vice versa we destroy the car's joints.
        if contact.fixtureA.body.userData.id == "head" and contact.fixtureB.body.userData.id == "ground":
            head_fixture = contact.fixtureA
            ground_fixture = contact.fixtureB
        elif contact.fixtureB.body.userData.id == "head" and contact.fixtureA.body.userData.id == "ground":
            head_fixture = contact.fixtureB
            ground_fixture = contact.fixtureA

        if head_fixture and ground_fixture and head_fixture.body.joints:
            torso = head_fixture.body.joints[0].other  # Get the torso body object using the joint
            car = torso.joints[3].other.userData  # Get the car body using the torso
            car.agent.dead = True

        # Check if we contact the wheel with the ground or vice versa.
        if contact.fixtureA.body.userData.id == "wheel" and contact.fixtureB.body.userData == "ground":
            contact.fixtureA.body.userData.on_ground = True
        elif contact.fixtureB.body.userData.id == "wheel" and contact.fixtureA.body.userData == "ground":
            contact.fixtureB.body.userData.on_ground = True

    def EndContact(self, contact: b2Contact) -> None:
        # End of contact, we need to set the on_ground variable on false
        if contact.fixtureA.body.userData.id == "wheel" and contact.fixtureB.body.userData.id == "ground":
            contact.fixtureA.body.userData.on_ground = False
        elif contact.fixtureB.body.userData.id == "wheel" and contact.fixtureA.body.userData.id == "ground":
            contact.fixtureA.body.userData.on_ground = False

    # def PreSolve(self, contact, oldManifold) -> None:
    #     pass
    #
    # def PostSolve(self, contact, impulse) -> None:
    #     pass


# Key events handler when human is playing
def handle_key_events(human_event: pygame.event, human_agent: 'agent.Agent',
                      human_right_down: bool, human_left_down: bool) -> None:
    if human_event.type == pygame.KEYDOWN:
        if human_event.key in (pygame.K_d, pygame.K_RIGHT):
            human_agent.car.motor_on(forward=True)
            human_right_down = True
        elif human_event.key in (pygame.K_a, pygame.K_LEFT):
            human_agent.car.motor_on(forward=False)
            human_left_down = True

    elif human_event.type == pygame.KEYUP:
        if human_event.key in (pygame.K_d, pygame.K_RIGHT):
            human_right_down = False
            if human_left_down:
                human_agent.car.motor_on(forward=False)
            else:
                human_agent.car.motor_off()
        elif human_event.key in (pygame.K_a, pygame.K_LEFT):
            human_left_down = False
            if human_right_down:
                human_agent.car.motor_on(forward=True)
            else:
                human_agent.car.motor_off()


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Hill climb")
clock = pygame.time.Clock()


def setup_world() -> tuple['ground.Ground', 'agent.Agent', b2World]:
    # Variables
    main_world = b2World(contactListener=ContactListener(), gravity=b2Vec2(0, GRAVITY), doSleep=True)
    ground_template = ground.Ground()  # Template to store the ground vectors
    ground_template.randomizeGround()  # Randomizes the ground using the difficulty and perlin noise

    # Generate until we find ground that is not too steep
    while ground_template.groundTooSteep():
        ground_template = ground.Ground()
        ground_template.randomizeGround()

    # Set up the ground
    main_ground = ground.Ground(main_world)
    main_ground.cloneFrom(ground_template)
    main_ground.setBodies(main_world)

    # Set up the world and agent
    human_agent = agent.Agent(real_world=main_world)
    human_agent.add_to_world()
    return main_ground, human_agent, main_world


def draw(render_ground, render_agent) -> None:
    # Clear the screen
    screen.fill((255, 255, 255))
    # Fill screen with sky colour
    screen.fill((135, 206, 235))
    # Draw the ground to screen
    render_ground.draw_ground()
    # Draw the agent
    render_agent.draw_agent()
    # Update the screen
    pygame.display.flip()


# For human manual play
if __name__ == "__main__":
    # Initialize world
    current_ground, current_agent, current_world = setup_world()
    while not current_agent.dead:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:  # Escape to quit game
                print("Escape was pressed, quiting the game...")
                pygame.quit()
            if HUMAN_PLAYING:
                # Initialize key variables for when human plays
                right_key_down = False
                left_key_down = False
                handle_key_events(event, current_agent, right_key_down, left_key_down)

        # Call the draw function
        draw(current_ground, current_agent)
        # Box2D simulation
        current_world.Step(timeStep=1.0 / FPS, velocityIterations=6 * 30, positionIterations=2 * 30)
        # Update Agent
        current_agent.update()
        # # Drive forward
        # current_agent.car.motor_on(forward=True)
        # Clear forces
        current_world.ClearForces()
        # Update render screen and fps
        pygame.display.flip()
        clock.tick(FPS)
    # Print final distance
    print(f"Final distance: {current_agent.car.max_distance}")
    # Quit the game
    pygame.quit()


    class HillRacing(gym.Env):
        metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": FPS
        }

        def __init__(self, render_mode: Optional[str] = None):
            self.world = b2World(gravity=GRAVITY, doSleep=True)
            self.ground: Optional[ground.Ground] = None  # List of ground that needs to be generated
            self.agent: Optional[agent.Agent] = None  # The agent class contains the car, wheels and person
            self.difficulty = DIFFICULTY  # Difficulty of the env, scales from -250 to 80 (easiest to hardest)
            self.action_space = spaces.Discrete(3)  # 2 do-able actions: gas, reverse, 3rd action is idling
            self.observation_space = spaces.Dict(
                {
                    # x coordinate from 0 to 1000 and y from 0 to 700.
                    "chassis_position": spaces.Box(low=np.array([0, 0]), high=np.array([1000, 700]), shape=(2,),
                                                   dtype=np.float32),
                    # Angle in degrees, can be -720 to 720.
                    "chassis_angle": spaces.Box(low=-720, high=720, shape=(1,), dtype=np.float32),
                    # Wheels speed, back and front wheel have same speed limits
                    "wheels_speed": spaces.Box(low=-10 * math.pi, high=10 * math.pi, shape=(2,), dtype=np.float32),
                    # "wheels_position": ...,
                    # "current_score": ...
                }
            )
            assert render_mode is None or render_mode in self.metadata["render_modes"]
            self.render_mode = render_mode
            self.screen: Optional[pygame.Surface] = None
            self.clock = None

        def _destroy_world(self):
            if not self.ground:
                return
            self.world.contactListener = None
            # Destroy ground bodies
            for gras in self.ground.grassBody:
                self.world.DestroyBody(gras)
            for dirt in self.ground.dirtBody:
                self.world.DestroyBody(dirt)
            self.ground = None
            if not self.agent:
                return
            # Function that destroys the whole agent, which means, car, person and wheels
            self.agent.destroy_agent()
            self.agent = None

        def _generate_ground(self):
            # Variables
            ground_template = ground.Ground()  # Template to store the ground vectors
            ground_template.randomizeGround()  # Randomizes the ground using the difficulty and perlin noise

            # Generate until we find ground that is not too steep
            while ground_template.groundTooSteep():
                ground_template = ground.Ground()
                ground_template.randomizeGround()

            # Add the ground to the world
            main_ground = ground.Ground(self.world)
            main_ground.cloneFrom(ground_template)  # Copy the ground_template to main_ground
            main_ground.setBodies(self.world)  # Add the bodies to the world

        def _generate_agent(self):
            self.agent = agent.Agent(real_world=self.world)
            self.agent.add_to_world()

        def _get_obs(self):
            return {
                "chassis_position": (self.agent.car.chassis_body.position.x, self.agent.car.chassis_body.position.y),
                "chassis_angle": math.degrees(-self.agent.car.chassis_body.angle),
                "wheels_speed": (self.agent.car.wheels[0].joint.speed, self.agent.car.wheels[1].joints.speed)
            }

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            # Destroy world
            self._destroy_world()
            self.game_over = False
            # Generate new world
            self.world.contactListener = ContactListener
            self._generate_ground()
            self._generate_agent()
            # Get the initial observations
            observations = self._get_obs()
            # Render mode
            if self.render_mode == "human":
                self.render()

            return observations

        def step(self, action: int):
            terminated = False
            reward = 0
            observation = {}
            info = {}

            match action:
                case 0:  # Idle
                    self.agent.car.motor_off()
                case 1:  # Gas
                    self.agent.car.motor_on(forward=True)
                case 2:  # Reverse
                    self.agent.car.motor_on(forward=False)

            # Step forward in the world
            self.world.Step(timeStep=1.0 / FPS, velocityIterations=6 * 30, positionIterations=2 * 30)

            # Dying is termination
            if self.agent.dead:
                terminated = True
            # When agent reaches the end, meaning 1000 meters
            elif self.agent.car.chassis_body.position.x >= 999:
                terminated = True

            # Reward is equal to 1 + current_distance - max_distance
            if self.agent.car.chassis_body.position.x >= self.agent.car.max_distance:
                reward = 1 + (self.agent.car.chassis_body.position.x - self.agent.car.max_distance)
            # Reward is equal to -1 + current_distance - max_distance
            elif self.agent.car.chassis_body.position.x < self.agent.car.max_distance:
                reward = -1 + (self.agent.car.chassis_body.position.x - self.agent.car.max_distance)

            # Get the current step observation
            observation = self._get_obs()

            return observation, reward, terminated, False, info

        def render(self):
            ...

        def close(self):
            ...
