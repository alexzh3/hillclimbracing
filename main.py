import world
import ground
import pygame
from world import World
from ground import Ground
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

# collisionCategories i.e what it is
WHEEL_CATEGORY = 0x0001
CHASSIS_CATEGORY = 0x0002
GRASS_CATEGORY = 0x0004
DIRT_CATEGORY = 0x0008
PERSON_CATEGORY = 0x0010

# collisionMasks i.e what it collides with
WHEEL_MASK = GRASS_CATEGORY
CHASSIS_MASK = DIRT_CATEGORY
GRASS_MASK = WHEEL_CATEGORY | PERSON_CATEGORY
DIRT_MASK = CHASSIS_CATEGORY
PERSON_MASK = GRASS_CATEGORY

# Define constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCALE = 30  # Pixels per meter / Scale
FPS = 60  # frames per second
DIFFICULTY = 50 # Difficulty of terrain


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# From the world-class create object
class_world = World()
# Create Box2D world
world = class_world.world

# Create ground body
ground = world.CreateStaticBody(
    position=(SCREEN_WIDTH / 2 / PPM, 0),
    shapes=polygonShape(box=(SCREEN_WIDTH / PPM / 2, 1))
)


def draw():
    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw bodies
    for body in world.bodies:
        for fixture in body.fixtures:
            shape = fixture.shape
            vertices = [(body.transform * v) * PPM for v in shape.vertices]
            vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]  # Convert to Pygame coordinates
            pygame.draw.polygon(screen, (0, 0, 255), vertices)

    # Update the screen
    pygame.display.flip()


if __name__ == "__main__":
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Step the Box2D world
        world.Step(1.0 / FPS, 10, 10)

        # Call the draw function
        draw()

        # Limit frames per second
        clock.tick(FPS)

    # Quit the game
    pygame.quit()
