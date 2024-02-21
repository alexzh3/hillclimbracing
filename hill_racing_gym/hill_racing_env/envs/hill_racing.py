from gymnasium.error import DependencyNotInstalled
try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, try 'pip install box2d box2d-kengz'")
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import numpy as np
import math
import ground
import agent

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

# Fundamental constants (not recommended to change)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCALE = 30  # Pixels per meter / Scale, Box2D counts in meters, pygame counts in pixels.
FPS = 60  # frames per second
GRAVITY = 10
WHEEL_SIZE = 35
HEAD_SIZE = 40
PERSON_WIDTH = 20
PERSON_HEIGHT = 40
panX = 0
panY = 0

# Gameplay variables
HUMAN_PLAYING = True
SPAWNING_Y = 0  # Spawn location y-coordinate (in pixels)
SPAWNING_X = 200  # Spawn location x-coordinate (in pixels)
MAX_SCORE = 1000  # Max score achievable (-/+ 10)
GROUND_DISTANCE = int(MAX_SCORE * SCALE + SPAWNING_X)  # How long the ground terrain should in pixel size
DIFFICULTY = -150  # Difficulty of terrain, max 30, min 230 (almost flat terrain)

# Load in pictures/sprites
wheel_sprite = pygame.image.load("pictures/wheel.png")
head_sprite = pygame.image.load("pictures/headLarge2.png")
car_sprite = pygame.image.load("pictures/car.png")
torso_sprite = pygame.image.load("pictures/torsoLarge.png")


# Contact listener for head and ground (Bad code)
class ContactListener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact: b2Contact) -> None:
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
        if contact.fixtureA.body.userData.id == "wheel" and contact.fixtureB.body.userData.id == "ground":
            contact.fixtureA.body.userData.on_ground = True
        if contact.fixtureB.body.userData.id == "wheel" and contact.fixtureA.body.userData.id == "ground":
            contact.fixtureB.body.userData.on_ground = True

    def EndContact(self, contact: b2Contact) -> None:
        # End of contact, we need to set the on_ground variable on false
        if contact.fixtureA.body.userData.id == "wheel" and contact.fixtureB.body.userData.id == "ground":
            contact.fixtureA.body.userData.on_ground = False
        if contact.fixtureB.body.userData.id == "wheel" and contact.fixtureA.body.userData.id == "ground":
            contact.fixtureB.body.userData.on_ground = False


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


def human_play():
    # Initialize world
    current_ground, current_agent, current_world = setup_world()
    # Initialize key variables for when human plays
    right_key_down = False
    left_key_down = False
    while not current_agent.dead:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:  # Escape to quit game
                print("Escape was pressed, quiting the game...")
                pygame.quit()
            handle_key_events(event, current_agent, right_key_down, left_key_down)

        # Call the draw function
        draw(current_ground, current_agent)
        # Box2D simulation
        current_world.Step(timeStep=1.0 / FPS, velocityIterations=6 * 30, positionIterations=2 * 30)
        # Print for debugging
        print(
            f"position: {current_agent.car.chassis_body.position.x, current_agent.car.chassis_body.position.y},"
            f"wheels_speeds: {current_agent.car.wheels[0].joint.speed, current_agent.car.wheels[1].joint.speed}")
        # Update Agent
        current_agent.update()
        # Update render screen and fps
        pygame.display.flip()
        clock.tick(FPS)
    # Print final distance
    print(f"Final score: {current_agent.score}")
    # Quit the game
    pygame.quit()


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
    # Fill screen with sky colour
    screen.fill((135, 206, 235))
    # Draw the ground to screen
    render_ground.draw_ground(screen)
    # Draw the agent
    render_agent.draw_agent(screen)
    # Update the screen
    pygame.display.flip()


class HillRacingEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": FPS
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            action_space: str = "discrete_3",
            reward_type: str = "distance",
            max_steps: int = metadata["render_fps"] * 20,
    ):
        self.world = b2World(gravity=(0, GRAVITY), doSleep=True)
        self.ground: Optional[ground.Ground] = None  # List of ground that needs to be generated
        self.agent: Optional[agent.Agent] = None  # The agent class contains the car, wheels and person
        self.difficulty = DIFFICULTY  # Difficulty of the env, scales from -250 to 80 (easiest to hardest)
        self.action_space_type = action_space  # What type of action space do we choose? (Discrete or continuous?)
        self.reward_type = reward_type  # Type of reward, distance-based vs action based vs wheel speed
        self.step_counter = None  # Counter to memorize the amount of steps done
        self.max_steps = max_steps  # Amount of maximum timesteps done without significant
        # progress, will be 20 seconds

        # Define action spaces
        match self.action_space_type:  # For experiments
            case "discrete_3":
                self.action_space = spaces.Discrete(n=3,
                                                    start=0)  # 3 do-able actions: gas, reverse, 3rd action is idling
            case "discrete_2":
                self.action_space = spaces.Discrete(n=2, start=1)  # 2 do-able actions: gas, reverse
            case "continuous":  # Continuous motor wheel speeds
                self.action_space = gym.spaces.Box(low=-13, high=13, shape=(1,), dtype=np.float32)
        # Define the observation space
        self.observation_space = spaces.Dict(
            {
                # x coordinate from 0 to 1000 and y from 0 to 700.
                "chassis_position": spaces.Box(low=np.array([0, 0]), high=np.array([1000, 700]), shape=(2,),
                                               dtype=np.float32),
                # Angle in degrees, can be -36000 to 36000.
                "chassis_angle": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
                # Wheels speed, back and front wheel have same speed limits, add 0.1 to avoid precision errors
                "wheels_speed": spaces.Box(low=-13 * math.pi + 0.1, high=13 * math.pi + 0.1, shape=(2,),
                                           dtype=np.float32),
                # When one of the wheels is makes contact with the ground, 0 means no contact and 1 means contact
                "on_ground": spaces.MultiBinary(n=2)
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
        self.world.DestroyBody(self.ground.grassBody)
        self.world.DestroyBody(self.ground.dirtBody)
        self.ground = None
        if not self.agent:
            return
        # Function that destroys the whole agent, which means, car, person and wheels
        self.agent.destroy_agent()
        self.agent = None

    def _generate_ground(self, seed: Optional[int] = None):
        # Variables
        ground_template = ground.Ground()  # Template to store the ground vectors
        ground_template.randomizeGround(seed=seed)  # Randomizes the ground using the difficulty and perlin noise

        # Generate until we find ground that is not too steep
        while ground_template.groundTooSteep():
            ground_template = ground.Ground()
            ground_template.randomizeGround()

        # Add the ground to the world
        self.ground = ground.Ground(self.world)
        self.ground.cloneFrom(ground_template)  # Copy the ground_template to self.ground
        self.ground.setBodies(self.world)  # Add the bodies to the world

    def _generate_agent(self):
        self.agent = agent.Agent(real_world=self.world)
        self.agent.add_to_world()

    # Function that executes an action based on the given action_space_type
    def _execute_action(self, action):
        match self.action_space_type:  # Check which action space type we have
            case "discrete_3":
                match action:
                    case 0:  # Idle
                        self.agent.car.motor_off()
                    case 1:  # Gas
                        self.agent.car.motor_on(forward=True)
                    case 2:  # Reverse
                        self.agent.car.motor_on(forward=False)
            case "discrete_2":
                match action:
                    case 1:  # Gas
                        self.agent.car.motor_on(forward=True)
                    case 2:  # Reverse
                        self.agent.car.motor_on(forward=False)
            case "continuous":  # Continuous motor wheel speeds
                self.agent.car.set_motor_wheel_speed(action[0])

    # Function that calculates the reward for a given timestep based on the reward_type
    def _get_reward(self, action):
        reward = 0
        match self.reward_type:
            case "distance":
                # Reward is equal to -1 + current_distance - max_distance vs less aggressive -0.2
                if self.agent.car.chassis_body.position.x < self.agent.car.prev_max_distance:
                    reward = -0.2 + (self.agent.car.chassis_body.position.x - self.agent.car.prev_max_distance)
                # Reward -1 if agent is at or around same position as last step vs less aggressive -0.1
                elif self.agent.car.chassis_body.position.x - self.agent.car.prev_max_distance < 0.001:
                    reward = -0.1
                # Reward is equal to 1 + current_position - max_distance
                elif self.agent.car.chassis_body.position.x > self.agent.car.prev_max_distance:
                    reward = 1 + (self.agent.car.chassis_body.position.x - self.agent.car.prev_max_distance)
            case "action":
                if action == 0:  # Idle
                    reward = -0.1
                elif action == 1:  # Gas
                    reward = 1
                elif action == 2:  # Reverse
                    reward = -0.2
            case "wheel_speed":
                wheel_speeds = [wheel.joint.speed for wheel in self.agent.car.wheels]
                # When wheel speeds are at a nearly idle state
                if all(-1 <= speed <= 1 for speed in wheel_speeds):
                    reward = -0.1
                # When we have wheel speeds that bring the car forward (negative wheel speed = forward)
                elif all(speed < 0 for speed in wheel_speeds):
                    reward = 1
                # When we have wheel speeds that bring the car backwards
                elif all(speed > 0 for speed in wheel_speeds):
                    reward = -0.2
                else:
                    reward = 0
        return reward

    def _get_obs(self):
        return {
            "chassis_position": np.array(
                [self.agent.car.chassis_body.position.x, self.agent.car.chassis_body.position.y], dtype=np.float32),
            "chassis_angle": np.array([(math.degrees(-self.agent.car.chassis_body.angle) % 360)], dtype=np.float32),
            "wheels_speed": np.array([self.agent.car.wheels[0].joint.speed, self.agent.car.wheels[1].joint.speed],
                                     dtype=np.float32),
            "on_ground": np.array([int(self.agent.car.wheels[0].on_ground), int(self.agent.car.wheels[1].on_ground)])
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        info = {}
        # Destroy world
        self._destroy_world()
        # Generate new world
        self.world.contactListener = ContactListener()
        self._generate_ground(seed=seed)
        self._generate_agent()
        self.step_counter = 0  # Set step counter to 0
        # Get the initial observations
        observations = self._get_obs()
        # Render mode
        if self.render_mode == "human":
            self.render()

        return observations, info

    def step(self, action: int | np.float32):
        terminated = False
        truncated = False
        stuck = False
        reward = 0  # initial reward of -1, if the agent does completely nothing
        info = {}

        # Execute action
        self._execute_action(action)
        # Step forward in the world
        self.world.Step(timeStep=1.0 / self.metadata["render_fps"], velocityIterations=6 * 30,
                        positionIterations=2 * 30)
        # Update agent status
        self.agent.update()
        # Update timestep counter
        self.step_counter += 1
        # Check if agent is stuck
        if math.floor(self.agent.car.max_distance) % 50 == 0:  # when we made more than 50 metres distance reset count
            self.step_counter = 0
        else:  # When no significant distance has been made for a long time, the agent must be stuck
            if self.step_counter > self.max_steps:
                stuck = True

        # If agent is dead or stuck
        if stuck:
            truncated = True
            reward = -100
        elif self.agent.dead:
            terminated = True
            reward = -100
        elif self.agent.score > MAX_SCORE:  # If max score is achieved
            terminated = True

        # Reward shaping if agent is still alive or not stuck
        if not truncated and not terminated:
            reward = self._get_reward(action)

        # Get the current step observation and info for debugging
        observation = self._get_obs()
        info = {
            "car_position": self.agent.car.chassis_body.position.x,
            "prev_max_distance": self.agent.car.prev_max_distance,
            "score": self.agent.score,
            "steps": self.step_counter,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )
            pygame.display.set_caption("Hill climb RL")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            self.clock.tick(self.metadata["render_fps"])

        assert self.screen is not None
        assert self.clock is not None
        # Fill screen with sky colour
        self.screen.fill((135, 206, 235))
        # Draw the ground to screen
        self.ground.draw_ground(self.screen)
        # Draw the agent
        self.agent.draw_agent(self.screen)
        # Update the screen
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    if HUMAN_PLAYING:
        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Hill climb")
        clock = pygame.time.Clock()
        human_play()
