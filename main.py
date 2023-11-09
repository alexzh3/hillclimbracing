from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")
import ground
import pygame

# collisionCategories represented in bits
WHEEL_CATEGORY = 0x0001
CHASSIS_CATEGORY = 0x0002
GRASS_CATEGORY = 0x0004
DIRT_CATEGORY = 0x0008
PERSON_CATEGORY = 0x0010

# collisionMasks, which category it collides with
WHEEL_MASK = GRASS_CATEGORY
CHASSIS_MASK = DIRT_CATEGORY
GRASS_MASK = WHEEL_CATEGORY | PERSON_CATEGORY
DIRT_MASK = CHASSIS_CATEGORY
PERSON_MASK = GRASS_CATEGORY

# Define constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCALE = 30  # Pixels per meter / Scale
FPS = 60  # frames per second
TIME_STEP = 1.0 / FPS
DIFFICULTY = 50  # Difficulty of terrain, max 100
panX = 0
panY = 0
GRAVITY = 10
WHEEL_SIZE = 17
PERSON_WIDTH = 15

# Game variables
NUMBER_OF_WORLDS = 100
grounds = []
worlds = []
HUMAN_PLAYING = False
SHOWING_GROUND = False

# Load in pictures
wheel_sprite = pygame.image.load("pictures/wheel.png")
head_sprite = pygame.image.load("pictures/head.png")
car_sprite = pygame.image.load("pictures/car.png")


# Contact listener for head and ground
class Contact_listener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Hill climb")
clock = pygame.time.Clock()


def initWorld():
    # Create ground template
    groundTemplate = ground.Ground()
    groundTemplate.randomizeGround()
    # Generate until we find a good ground
    while groundTemplate.groundTooSteep():
        groundTemplate = ground.Ground()
        groundTemplate.randomizeGround()
    # Generate a set number of world with the same ground
    for i in range(0, NUMBER_OF_WORLDS):
        tempWorld = b2World(b2Vec2(0, GRAVITY), True)
        tempGround = ground.Ground(tempWorld)
        tempGround.cloneFrom(groundTemplate)
        tempGround.setBodies(tempWorld)
        grounds.append(tempGround)
        worlds.append(tempWorld)

    otherWorld = b2World(b2Vec2(0, GRAVITY), True)
    tempGround = ground.Ground(otherWorld)
    tempGround.cloneFrom(groundTemplate)
    tempGround.setBodies(otherWorld)
    return tempGround


def draw(render_ground):
    # Clear the screen
    screen.fill((255, 255, 255))
    # Fill screen with sky colour
    screen.fill((135, 206, 235))
    # Draw the ground to screen
    print(len(grounds))
    render_ground.draw_ground()
    # Update the screen
    pygame.display.flip()


if __name__ == "__main__":
    # Initialize world
    temp_ground = initWorld()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Call the draw function
        if not SHOWING_GROUND:
            draw(temp_ground)
            SHOWING_GROUND = True
        # Limit frames per second
        clock.tick(FPS)
    # Quit the game
    pygame.quit()
