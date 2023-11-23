from gymnasium.error import DependencyNotInstalled
try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed")
import ground
import agent
import pygame
from typing import Type

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
TIME_STEP = 1.0 / FPS
DIFFICULTY = 0  # Difficulty of terrain, max 100, min 100
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
            car.agent.shadow_dead = True

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

    # # # Generate a set number of world with the same ground
    # for i in range(0, NUMBER_OF_WORLDS):
    #     main_world = b2World(b2Vec2(0, GRAVITY), True)
    #     main_ground = ground.Ground(main_world)
    #     main_ground.cloneFrom(ground_template)
    #     main_ground.setBodies(main_world)
    #     grounds.append(main_ground)
    #     worlds.append(main_world)

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


if __name__ == "__main__":
    # Initialize world
    current_ground, current_agent, current_world = setup_world()

    while not current_agent.shadow_dead:
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
        current_world.Step(TIME_STEP, 10, 10)
        # Update Agent
        current_agent.update()
        # # Drive forward
        # current_agent.car.motor_on(forward=True)
        # Clear forces
        current_world.ClearForces()
        # Update render screen and fps
        pygame.display.flip()
        clock.tick(FPS)
    # Print final score
    print(f"Final score: {current_agent.score}")
    # Quit the game
    pygame.quit()
